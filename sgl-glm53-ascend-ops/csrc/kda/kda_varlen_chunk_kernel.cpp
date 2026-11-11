// GLM-5.3 chunked KDA — MIX kernel, one AIC and two AIV per block.
//
// Specializes `kda_chunk_torch` (bench/kda_chunk/kda_python.py) to BT == 64,
// K == V == 128, H == HV. Banners map code to that reference: `══ NAME ══`
// opens a stage, `── expr ──` marks one expression.
//
// Design notes, measurements and traps: kda_varlen_chunk.md, same directory.
// Headings there are named by the code that points at them.

#include "kernel_operator.h"
#include "kda_varlen_chunk.h"
#include "utils.h"

#ifndef __CCE_KT_TEST__
#include <acl/acl.h>
#endif

namespace kda_chunk {

// Hardware event IDs, managed by hand as in the recurrent kernels.
constexpr int32_t EVENT_MTE2_V = 0;
constexpr int32_t EVENT_MTE2_V_B = 1;
constexpr int32_t EVENT_V_MTE2 = 0;
constexpr int32_t EVENT_V_MTE3 = 0;
constexpr int32_t EVENT_MTE3_V = 0;
constexpr int32_t EVENT_S_V = 0;
constexpr int32_t EVENT_V_S = 0;
constexpr int32_t EVENT_S_MTE3 = 0;

// The fp32 Cube fractal is 16 rows x 8 columns, 512 B.
constexpr uint32_t kFracRow = 16;
constexpr uint32_t kFracCol = 32 / sizeof(float);

__aicore__ inline constexpr uint32_t CeilAlign(uint32_t x, uint32_t a)
{
    return (x + a - 1) / a * a;
}

// L1 byte offsets. L1 is 512 KB and the largest operand here is 128x128 fp32
// (64 KB), so two fixed halves are enough.
constexpr uint32_t kL1AOffset = 0;
constexpr uint32_t kL1BOffset = 128 * 1024;

}  // namespace kda_chunk

// ── Cube: C = A @ B, or A^T @ B, or A @ B^T ───────────────────────────
//
// GM(ND) -> L1(Nz) -> L0A(Zz)/L0B(Zn) -> L0C(Nz) -> GM(ND). The `*RowStride`
// arguments default to 0, meaning packed. See kda_varlen_chunk.md.
template <bool A_TRANS, bool B_TRANS>
__aicore__ inline void CubeMatmul(
    const AscendC::GlobalTensor<float> &aGm,
    const AscendC::GlobalTensor<float> &bGm,
    const AscendC::GlobalTensor<float> &cGm,
    uint32_t M, uint32_t N, uint32_t K, uint32_t dstRowStride = 0,
    bool first = false, uint32_t aRowStride = 0, uint32_t bRowStride = 0)
{
    using namespace kda_chunk;

    const uint32_t aRows = A_TRANS ? K : M;
    const uint32_t aCols = A_TRANS ? M : K;
    const uint32_t bRows = B_TRANS ? N : K;
    const uint32_t bCols = B_TRANS ? K : N;

    const uint32_t aRowAlign = CeilAlign(aRows, kFracRow);
    const uint32_t aColAlign = CeilAlign(aCols, kFracCol);
    const uint32_t bRowAlign = CeilAlign(bRows, kFracRow);
    const uint32_t bColAlign = CeilAlign(bCols, kFracCol);
    const uint32_t mAlign = CeilAlign(M, kFracRow);
    const uint32_t nAlign = CeilAlign(N, kFracCol);
    const uint32_t kAlign = CeilAlign(K, kFracRow);

    AscendC::LocalTensor<float> a1(AscendC::TPosition::A1, kL1AOffset,
                                   aRowAlign * aColAlign);
    AscendC::LocalTensor<float> b1(AscendC::TPosition::B1, kL1BOffset,
                                   bRowAlign * bColAlign);
    AscendC::LocalTensor<float> a2(AscendC::TPosition::A2, 0,
                                   mAlign * CeilAlign(K, kFracCol));
    AscendC::LocalTensor<float> b2(AscendC::TPosition::B2, 0, kAlign * nAlign);
    AscendC::LocalTensor<float> c1(AscendC::TPosition::CO1, 0, mAlign * nAlign);

    // ── GM(ND) -> L1(Nz) ──────────────────────────────────────────────
    //
    // The three `!first` barriers release on-chip buffers the previous call
    // still reads; a chain without them faults the core. See the md.
    if (!first) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    }

    AscendC::Nd2NzParams ap;
    ap.ndNum = 1;
    ap.nValue = static_cast<uint16_t>(aRows);
    ap.dValue = static_cast<uint16_t>(aCols);
    ap.srcNdMatrixStride = 0;
    ap.srcDValue =
        static_cast<uint16_t>(aRowStride == 0 ? aCols : aRowStride);
    ap.dstNzC0Stride = static_cast<uint16_t>(aRowAlign);
    ap.dstNzNStride = 1;
    ap.dstNzMatrixStride = 0;
    AscendC::DataCopy(a1, aGm, ap);

    AscendC::Nd2NzParams bp;
    bp.ndNum = 1;
    bp.nValue = static_cast<uint16_t>(bRows);
    bp.dValue = static_cast<uint16_t>(bCols);
    bp.srcNdMatrixStride = 0;
    bp.srcDValue =
        static_cast<uint16_t>(bRowStride == 0 ? bCols : bRowStride);
    bp.dstNzC0Stride = static_cast<uint16_t>(bRowAlign);
    bp.dstNzNStride = 1;
    bp.dstNzMatrixStride = 0;
    AscendC::DataCopy(b1, bGm, bp);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);

    // ── L1(Nz) -> L0A(Zz) ─────────────────────────────────────────────
    if (!first) {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    }
    {
        AscendC::LoadData3DParamsV2<float> p;
        p.l1H = 1;
        p.l1W = static_cast<uint16_t>(aRowAlign);
        p.channelSize = static_cast<uint16_t>(aColAlign);
        p.kExtension = static_cast<uint16_t>(
            A_TRANS ? CeilAlign(M, kFracCol) : CeilAlign(K, kFracCol));
        p.mExtension = static_cast<uint16_t>(
            A_TRANS ? CeilAlign(K, kFracCol * 2) : mAlign);
        p.strideW = 1;
        p.strideH = 1;
        p.filterW = 1;
        p.filterH = 1;
        p.dilationFilterW = 1;
        p.dilationFilterH = 1;
        p.filterSizeW = false;
        p.filterSizeH = false;
        p.enTranspose = A_TRANS;
        p.fMatrixCtrl = false;
        AscendC::LoadData(a2, a1, p);
    }

    // ── L1(Nz) -> L0B(Zn) ─────────────────────────────────────────────
    //
    // B stored [N, K] is already Zn, so Load2D; [K, N] needs Load3Dv2's
    // transpose.
    if constexpr (B_TRANS) {
        AscendC::LoadData2DParams p;
        p.repeatTimes = static_cast<uint8_t>(bRowAlign / kFracRow);
        p.srcStride = 1;
        p.dstGap = 0;
        p.ifTranspose = false;
        const uint32_t frac = kFracRow * kFracCol;
        const uint32_t step = (bRowAlign / kFracRow) * frac;
        for (uint32_t i = 0; i < bColAlign / kFracCol; ++i) {
            AscendC::LoadData(b2[i * step], b1[i * step], p);
        }
    } else {
        AscendC::LoadData3DParamsV2<float> p;
        p.l1H = 1;
        p.l1W = static_cast<uint16_t>(bRowAlign);
        p.channelSize = static_cast<uint16_t>(bColAlign);
        p.kExtension = static_cast<uint16_t>(nAlign);
        p.mExtension = static_cast<uint16_t>(kAlign);
        p.strideW = 1;
        p.strideH = 1;
        p.filterW = 1;
        p.filterH = 1;
        p.dilationFilterW = 1;
        p.dilationFilterH = 1;
        p.filterSizeW = false;
        p.filterSizeH = false;
        p.enTranspose = true;
        p.fMatrixCtrl = false;
        AscendC::LoadData(b2, b1, p);
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

    // ── Mmad ──────────────────────────────────────────────────────────
    if (!first) {
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }
    AscendC::MmadParams mm;
    mm.m = static_cast<uint16_t>(M);
    mm.n = static_cast<uint16_t>(N);
    mm.k = static_cast<uint16_t>(K);
    mm.cmatrixInitVal = true;
    AscendC::Mmad(c1, a2, b2, mm);

    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

    // ── L0C(Nz) -> GM(ND) ─────────────────────────────────────────────
    AscendC::FixpipeParamsV220 fp;
    fp.nSize = static_cast<uint16_t>(N);
    fp.mSize = static_cast<uint16_t>(M);
    // `srcStride` is mSize rounded up to 16, in C0 units; under NZ2ND
    // `dstStride` is the destination row's element count.
    fp.srcStride = static_cast<uint16_t>(mAlign);
    fp.dstStride =
        static_cast<uint32_t>(dstRowStride == 0 ? N : dstRowStride);
    fp.ndNum = 1;
    fp.srcNdStride = 0;
    fp.dstNdStride = 0;
    AscendC::Fixpipe(cGm, c1, fp);
}

// TRACK_STATE  the reference's `track_state_indices` path: checkpoint the
//              state after `track_len` sequence-local tokens.
template <bool TRACK_STATE>
class KdaVarlenChunk {
public:
    __aicore__ inline void Init(const KdaChunkLaunchArgs &args,
                                uint32_t numHeads);
    __aicore__ inline void Process();

private:
    struct config {
        static constexpr uint32_t D = kda_chunk::kD;
        static constexpr uint32_t BT = kda_chunk::kBT;
        static constexpr uint32_t SB = kda_chunk::kRebaseRows;
        static constexpr uint32_t NB = kda_chunk::kRebaseBlocks;
    };

    // UB map, by hand at absolute addresses as in the recurrent kernels.
    // Only the AIVs touch it. Per-buffer notes in the md.
    struct UbLayout {
        static constexpr uint32_t D = config::D;
        static constexpr uint32_t BT = config::BT;
        static constexpr uint32_t ROW_F32 = D * sizeof(float);      // 512
        static constexpr uint32_t TILE_F32 = BT * ROW_F32;          // 32 KB
        static constexpr uint32_t TILE_BF16 = BT * D * sizeof(bfloat16_t);

        // ── the chunk's staged inputs ─────────────────────────────────
        static constexpr uint32_t QN_ADDR = 0x00000;                // [64,128]
        static constexpr uint32_t KN_ADDR = QN_ADDR + TILE_F32;
        static constexpr uint32_t GC_ADDR = KN_ADDR + TILE_F32;     // cumsum
        static constexpr uint32_t TMP_ADDR = GC_ADDR + TILE_F32;

        // ── per-row scalars ───────────────────────────────────────────
        static constexpr uint32_t BRAW_ADDR = TMP_ADDR + TILE_F32;
        static constexpr uint32_t BETA_ADDR = BRAW_ADDR + BT * sizeof(float);
        static constexpr uint32_t NRM_ADDR = BETA_ADDR + BT * sizeof(float);
        static constexpr uint32_t RED_ADDR =
            NRM_ADDR + 2 * BT * sizeof(float);
        // `b_dt_bias` for this head, loaded once per block.
        static constexpr uint32_t BIAS_ADDR = RED_ADDR + 64 * sizeof(float);
        // `lower_bound` as a tensor, for the gate's Div.
        static constexpr uint32_t LB_ADDR = BIAS_ADDR + ROW_F32;
        // Scalar/Vector exchange for `exp(A_log)`: no scalar exp here.
        static constexpr uint32_t SCR_ADDR = LB_ADDR + D * sizeof(float);
        // The second [64, 128] working tile.
        static constexpr uint32_t NORM_ADDR = SCR_ADDR + 64 * sizeof(float);
        // The bf16 staging tile, half a chunk, reused for q/k/a/v in turn.
        static constexpr uint32_t STAGE_ADDR = NORM_ADDR + TILE_F32;
        static constexpr uint32_t UB_END = STAGE_ADDR + TILE_BF16 / 2;

        // 192 KB in MIX mode, not 256; overrunning it faults the core.
        static_assert(UB_END <= 192 * 1024, "the chunk UB map must fit");
    };

    __aicore__ inline void RunAiv();
    __aicore__ inline void RunAic();

    // This block's workspace slice at element offset `off`. A member, not a
    // lambda: `operator[]` will not bind on a captured const member.
    __aicore__ inline AscendC::GlobalTensor<float> Slice(uint32_t off)
    {
        return wsGm[wsBase_ + static_cast<uint64_t>(off)];
    }

    // A [rows, cols] sub-block of a wider UB tile, packed into the
    // workspace. `srcRowStride - cols` must be a multiple of 8 floats.
    __aicore__ inline void StoreBlock(uint32_t off,
                                      const AscendC::LocalTensor<float> &src,
                                      uint32_t rows, uint32_t cols,
                                      uint32_t srcRowStride)
    {
        AscendC::DataCopyParams p{
            static_cast<uint16_t>(rows),
            static_cast<uint16_t>(cols * sizeof(float) / 32),
            static_cast<uint16_t>((srcRowStride - cols) * sizeof(float) / 32),
            0};
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
        AscendC::DataCopy(Slice(off), src, p);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
    }

    // AIV stages
    __aicore__ inline void LoadChunk(uint32_t head, int32_t tok0,
                                     uint32_t rows);
    __aicore__ inline void StageCast(
        const AscendC::GlobalTensor<bfloat16_t> &src, uint64_t base,
        const AscendC::LocalTensor<float> &dst, uint32_t rows);
    __aicore__ inline void PrepareFactors(uint32_t head, uint32_t rows);
    __aicore__ inline void WriteRebased(uint32_t rows);
    // X += I on a [BT, BT] fp32 tile. On Scalar because no vector form
    // reaches a stride-65 diagonal; see the md.
    __aicore__ inline void AddIdentity(
        const AscendC::LocalTensor<float> &x, uint32_t rows)
    {
        constexpr uint32_t BT = config::BT;
        AscendC::SetFlag<AscendC::HardEvent::V_S>(kda_chunk::EVENT_V_S);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(kda_chunk::EVENT_V_S);
        for (uint32_t r = 0; r < rows; ++r) {
            const uint32_t i = r * BT + r;
            x.SetValue(i, x.GetValue(i) + 1.0f);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(kda_chunk::EVENT_S_V);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(kda_chunk::EVENT_S_V);
        // The tile goes straight to MTE3, so the scalar writes need S_MTE3.
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(kda_chunk::EVENT_S_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(kda_chunk::EVENT_S_MTE3);
    }
    __aicore__ inline void BuildAuxOperands(uint32_t head, int32_t tok0,
                                            uint32_t rows);
    __aicore__ inline void FinishChunk(uint32_t head, int32_t tok0,
                                       uint32_t rows);
#if KDA_CHUNK_INVERSE_SQUARING
    __aicore__ inline void AdvanceInverse(uint32_t rows);
#else
    __aicore__ inline void SolveDiagBlocks(uint32_t rows);
    __aicore__ inline void SolveDiagBlockHelper();
    __aicore__ inline void SubstituteBlock(uint32_t r0, uint32_t rows);
#endif
    __aicore__ inline void RebaseKDecay(uint32_t r);
    __aicore__ inline void ApplyStateUpdate(uint32_t refRow, uint32_t rows);
    __aicore__ inline void CopyStateOut(uint32_t slot, uint32_t head);
    __aicore__ inline void RestoreState();
    __aicore__ inline void CopyWs(uint32_t dstOff, uint32_t srcOff,
                                  uint32_t count);

    AscendC::GlobalTensor<bfloat16_t> qGm;
    AscendC::GlobalTensor<bfloat16_t> kGm;
    AscendC::GlobalTensor<bfloat16_t> vGm;
    AscendC::GlobalTensor<bfloat16_t> aGm;
    AscendC::GlobalTensor<bfloat16_t> bGm;
    AscendC::GlobalTensor<float> aLogGm;
    AscendC::GlobalTensor<float> dtBiasGm;
    AscendC::GlobalTensor<bfloat16_t> outGm;
    AscendC::GlobalTensor<float> stateGm;
    AscendC::GlobalTensor<int32_t> indicesGm;
    AscendC::GlobalTensor<int32_t> startsGm;
    AscendC::GlobalTensor<int32_t> trackIndicesGm;
    AscendC::GlobalTensor<int32_t> trackLensGm;
    AscendC::GlobalTensor<float> wsGm;

    float scale_ = 0.0f;
    float lowerBound_ = 0.0f;
    uint32_t numReq_ = 0;
    uint32_t numHeads_ = 0;
    uint64_t wsBase_ = 0;      // this block's workspace slice, in elements

    AscendC::LocalTensor<float> qnLocal;
    AscendC::LocalTensor<float> knLocal;
    AscendC::LocalTensor<float> gcLocal;
    AscendC::LocalTensor<float> tmpLocal;
    AscendC::LocalTensor<bfloat16_t> stageLocal;
    AscendC::LocalTensor<float> brawLocal;
    AscendC::LocalTensor<float> betaLocal;
    AscendC::LocalTensor<float> nrmLocal;
    AscendC::LocalTensor<float> redLocal;
    AscendC::LocalTensor<float> biasLocal;
    AscendC::LocalTensor<float> lbRowLocal;
    AscendC::LocalTensor<float> scrLocal;
    AscendC::LocalTensor<float> normLocal;
};

template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::Init(
    const KdaChunkLaunchArgs &args, uint32_t numHeads)
{
    qGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.q));
    kGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.k));
    vGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.v));
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.a));
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.b));
    aLogGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.A_log));
    dtBiasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.dt_bias));
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args.out));
    stateGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.state));
    indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(args.indices));
    startsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(args.starts));
    wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.workspace));
    if constexpr (TRACK_STATE) {
        trackIndicesGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(args.track_indices));
        trackLensGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(args.track_lens));
    }

    scale_ = args.scale;
    lowerBound_ = args.lower_bound;
    numReq_ = args.num_req;
    numHeads_ = numHeads;

    using L = AscendC::LocalTensor<float>;
    using LB = AscendC::LocalTensor<bfloat16_t>;
    constexpr auto P = AscendC::TPosition::VECCALC;
    constexpr uint32_t D = config::D;
    constexpr uint32_t BT = config::BT;

    qnLocal = L(P, UbLayout::QN_ADDR, BT * D);
    knLocal = L(P, UbLayout::KN_ADDR, BT * D);
    gcLocal = L(P, UbLayout::GC_ADDR, BT * D);
    tmpLocal = L(P, UbLayout::TMP_ADDR, BT * D);
    stageLocal = LB(P, UbLayout::STAGE_ADDR, (BT / 2) * D);
    brawLocal = L(P, UbLayout::BRAW_ADDR, BT);
    betaLocal = L(P, UbLayout::BETA_ADDR, BT);
    nrmLocal = L(P, UbLayout::NRM_ADDR, 2 * BT);
    redLocal = L(P, UbLayout::RED_ADDR, 64);
    biasLocal = L(P, UbLayout::BIAS_ADDR, D);
    lbRowLocal = L(P, UbLayout::LB_ADDR, D);
    scrLocal = L(P, UbLayout::SCR_ADDR, 64);
    normLocal = L(P, UbLayout::NORM_ADDR, BT * D);
}

// ── One (sequence, head) ──────────────────────────────────────────────
//
// The reference's sequence and head axes collapse into the block index, head
// minor; its chunk loop stays a loop because the state carries across
// chunks. The AIV does everything elementwise and the AIC every matmul,
// alternating through the chunk via this block's workspace slice.
template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::Process()
{
    if ASCEND_IS_AIV {
        RunAiv();
    }
    if ASCEND_IS_AIC {
        RunAic();
    }
}


// ── one bf16 [rows, D] tensor, staged and cast to fp32 ───────────────
//
// Two half-chunk copies, since the staging tile is half a chunk. `V_MTE2` is
// the write-after-read on it: without it every Cast reads the last copy's
// rows. See the md.
template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::StageCast(
    const AscendC::GlobalTensor<bfloat16_t> &src, uint64_t base,
    const AscendC::LocalTensor<float> &dst, uint32_t rows)
{
    constexpr uint32_t D = config::D;
    constexpr uint32_t HALF = config::BT / 2;

    for (uint32_t r0 = 0; r0 < rows; r0 += HALF) {
        const uint32_t n = (rows - r0 < HALF) ? (rows - r0) : HALF;
        AscendC::DataCopyParams p{
            static_cast<uint16_t>(n),
            static_cast<uint16_t>(D * sizeof(bfloat16_t) / 32),
            static_cast<uint16_t>((numHeads_ - 1) * D * sizeof(bfloat16_t) / 32),
            0};
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
        AscendC::DataCopy(stageLocal, src[base + static_cast<uint64_t>(r0) *
                                                     numHeads_ * D], p);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kda_chunk::EVENT_MTE2_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kda_chunk::EVENT_MTE2_V);
        AscendC::Cast(dst[r0 * D], stageLocal, AscendC::RoundMode::CAST_NONE,
                      n * D);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

// ── b_q, b_k, b_a, b_v for one chunk ─────────────────────────────────
//
//     q_c, k_c, v_c, g_c, beta_c = rearrange(x[:, bos:eos], ...)
//
// Rows are strided by H in GM, so each tensor is one DataCopy with a row gap
// of (H-1) rows.
template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::LoadChunk(
    uint32_t head, int32_t tok0, uint32_t rows)
{
    const uint64_t base = static_cast<uint64_t>(tok0) * numHeads_ * config::D
                          + static_cast<uint64_t>(head) * config::D;

    StageCast(qGm, base, qnLocal, rows);
    StageCast(kGm, base, knLocal, rows);
    // The gate lands in `gc` and is turned into its cumsum downstream.
    StageCast(aGm, base, gcLocal, rows);

    // ── b_beta = load(p_b) ───────────────────────────────────────────
    //
    // One bf16 per row, strided by H, so Scalar gathers it: a DataCopy would
    // move one useful element per datablock.
    for (uint32_t r = 0; r < rows; ++r) {
        const uint64_t bRow =
            static_cast<uint64_t>(tok0 + r) * numHeads_ + head;
        brawLocal.SetValue(r, AscendC::ToFloat(bGm.GetValue(bRow)));
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(kda_chunk::EVENT_S_V);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(kda_chunk::EVENT_S_V);
}

// ── qn, kn, g, beta ──────────────────────────────────────────────────
//
//     qn = q / sqrt((q*q).sum(-1) + 1e-6) * scale
//     kn = k / sqrt((k*k).sum(-1) + 1e-6)
//     g = lower_bound * sigmoid(A_log.exp() * (a + dt_bias))
//     beta = sigmoid(b)
//     g_c = g.cumsum(-2)
//
// `kda_pre_torch`, for the whole chunk at once. The reference's association
// is kept; the md lists the four places that is load-bearing.
template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::PrepareFactors(
    uint32_t head, uint32_t rows)
{
    constexpr uint32_t D = config::D;

    // ── beta = 1 / (1 + exp(-b)) ─────────────────────────────────────
    AscendC::Muls(brawLocal, brawLocal, -1.0f, rows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(brawLocal, brawLocal, rows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(brawLocal, brawLocal, 1.0f, rows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(kda_chunk::EVENT_V_S);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(kda_chunk::EVENT_V_S);

    // ── the two L2 norms ─────────────────────────────────────────────
    //
    // `tl.sum` folds the two 64-lane halves before reducing; a plain 2x64
    // WholeReduceSum pair does not match it.
    AscendC::Mul(tmpLocal, qnLocal, qnLocal, rows * D);
    AscendC::PipeBarrier<PIPE_V>();
    ReduceRows128_AS<config::BT>(tmpLocal, redLocal, nrmLocal);
    AscendC::Mul(tmpLocal, knLocal, knLocal, rows * D);
    AscendC::PipeBarrier<PIPE_V>();
    ReduceRows128_AS<config::BT>(tmpLocal, redLocal, nrmLocal[config::BT]);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(kda_chunk::EVENT_V_S);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(kda_chunk::EVENT_V_S);

    // One Scalar visit for the chunk: beta's reciprocal and both square roots.
    for (uint32_t r = 0; r < rows; ++r) {
        betaLocal.SetValue(r, 1.0f / brawLocal.GetValue(r));
        // Scalar sqrt: Vector `Sqrt`/`Div` are 1 ulp on this hardware.
        const float qs = sqrt(nrmLocal.GetValue(r) + 1e-6f);
        const float ks = sqrt(nrmLocal.GetValue(config::BT + r) + 1e-6f);
        nrmLocal.SetValue(r, scale_ / qs);
        nrmLocal.SetValue(config::BT + r, 1.0f / ks);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(kda_chunk::EVENT_S_V);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(kda_chunk::EVENT_S_V);

    // A row at a time: the factor is per row and Muls takes a scalar.
    for (uint32_t r = 0; r < rows; ++r) {
        AscendC::Muls(qnLocal[r * D], qnLocal[r * D],
                      nrmLocal.GetValue(r), D);
        AscendC::Muls(knLocal[r * D], knLocal[r * D],
                      nrmLocal.GetValue(config::BT + r), D);
    }
    AscendC::PipeBarrier<PIPE_V>();

    // ── g = lower_bound / (1 + exp(-(decay * (a + dt_bias)))) ────────
    const float aLog = aLogGm.GetValue(head);
    for (uint32_t i = 0; i < 8; ++i) {
        scrLocal.SetValue(i, aLog);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(kda_chunk::EVENT_S_V);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(kda_chunk::EVENT_S_V);
    AscendC::Exp(scrLocal, scrLocal, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(kda_chunk::EVENT_V_S);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(kda_chunk::EVENT_V_S);
    const float decay = scrLocal.GetValue(0);

    for (uint32_t r = 0; r < rows; ++r) {
        AscendC::Add(gcLocal[r * D], gcLocal[r * D], biasLocal, D);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(gcLocal, gcLocal, -decay, rows * D);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(gcLocal, gcLocal, rows * D);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(gcLocal, gcLocal, 1.0f, rows * D);
    AscendC::PipeBarrier<PIPE_V>();
    // A real division, to match the reference; `lbRowLocal` is the numerator
    // broadcast over a row.
    for (uint32_t r = 0; r < rows; ++r) {
        AscendC::Div(gcLocal[r * D], lbRowLocal, gcLocal[r * D], D);
    }
    AscendC::PipeBarrier<PIPE_V>();

    // ── g_c = g.cumsum(-2) ───────────────────────────────────────────
    //
    // Along the token axis: each row adds the row above it, serially.
    for (uint32_t r = 1; r < rows; ++r) {
        AscendC::Add(gcLocal[r * D], gcLocal[r * D], gcLocal[(r - 1) * D], D);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

// ── the rebased row and column factors ───────────────────────────────
//
//     A[r, c] = sum_k kn[r,k] kn[c,k] exp(gc[r,k] - gc[c,k])
//
// written so the Cube can compute it:
//
//     row[r,k] = kn[r,k] exp(gc[r,k] - m[k])      per rebase block
//     col[c,k] = kn[c,k] exp(m[k] - gc[c,k])      same m
//     A[r, c] = sum_k row[r,k] col[c,k]
//
// m is gc of the block's midpoint row; see `kRebaseRows`. `kWsRowQ` is the
// same thing with qn, for Aqk.
template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::WriteRebased(uint32_t rows)
{
    using namespace kda_chunk;
    constexpr uint32_t D = config::D;
    constexpr uint32_t SB = config::SB;
    constexpr uint32_t NB = config::NB;

    for (uint32_t R = 0; R < NB; ++R) {
        const uint32_t r0 = R * SB;
        const uint32_t r1 = r0 + SB;
        const uint32_t mRow = (r0 + SB / 2) * D;

        // ── row factors: rows [r0, r1) of this block ──────────────────
        //
        // exp(gc[r] - m), one row at a time: the strided broadcast `Sub`
        // faults the vector core here.
        for (uint32_t r = r0; r < r1; ++r) {
            AscendC::Sub(tmpLocal[r * D], gcLocal[r * D], gcLocal[mRow], D);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(tmpLocal[r0 * D], tmpLocal[r0 * D], SB * D);
        AscendC::PipeBarrier<PIPE_V>();
        // The two share exp(gc - m), multiplied into kn then qn.
        AscendC::Mul(normLocal[r0 * D], knLocal[r0 * D], tmpLocal[r0 * D],
                     SB * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
        AscendC::DataCopy(Slice(kWsRowK + r0 * D), normLocal[r0 * D], SB * D);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
        AscendC::Mul(normLocal[r0 * D], qnLocal[r0 * D], tmpLocal[r0 * D],
                     SB * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
        AscendC::DataCopy(Slice(kWsRowQ + r0 * D), normLocal[r0 * D], SB * D);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);

        // ── column factors: rows [0, r1), the lower-triangular reach ──
        //
        // exp(m - gc[c]), a whole [r1, D] tile: the matmul reads it as B in
        // [N, K] orientation.
        for (uint32_t c = 0; c < r1; ++c) {
            AscendC::Sub(tmpLocal[c * D], gcLocal[mRow], gcLocal[c * D], D);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(tmpLocal, tmpLocal, r1 * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(tmpLocal, knLocal, tmpLocal, r1 * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
        AscendC::DataCopy(Slice(kWsColK + R * kTileBT_D), tmpLocal, r1 * D);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
    }
}

// ── the right operands of the auxiliary matmuls ──────────────────────
//
//     w = A @ (g_c.exp() * k_c)
//     u = A @ v_c
//     o = (q_i * g_i.exp()) @ S + ...
//     S = ... + ((g_i[-1] - g_i).exp() * k_i).T @ v_i
//
// beta is folded into the right operands rather than scaling the [BT, BT]
// inverse: (X * beta[None, :]) @ M == X @ (beta[:, None] * M).
template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::BuildAuxOperands(
    uint32_t head, int32_t tok0, uint32_t rows)
{
    using namespace kda_chunk;
    constexpr uint32_t D = config::D;
    constexpr uint32_t BT = config::BT;

    // exp(g_c), used by both w's operand and qg
    AscendC::Exp(tmpLocal, gcLocal, rows * D);
    AscendC::PipeBarrier<PIPE_V>();

    // beta * (exp(g_c) * k_c)
    AscendC::Mul(normLocal, tmpLocal, knLocal, rows * D);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t r = 0; r < rows; ++r) {
        AscendC::Muls(normLocal[r * D], normLocal[r * D],
                      betaLocal.GetValue(r), D);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
    AscendC::DataCopy(Slice(kWsKgBeta), normLocal, rows * D);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);

    // q_i * exp(g_i), the output's state term; no beta here
    AscendC::Mul(normLocal, tmpLocal, qnLocal, rows * D);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
    AscendC::DataCopy(Slice(kWsQg), normLocal, rows * D);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);

    // beta * v_c. v is re-staged here rather than held across the chunk:
    // keeping it resident would need another [64, 128] tile and the UB map
    // does not have one.
    const uint64_t vBase = static_cast<uint64_t>(tok0) * numHeads_ * D
                           + static_cast<uint64_t>(head) * D;
    StageCast(vGm, vBase, normLocal, rows);
    for (uint32_t r = 0; r < rows; ++r) {
        AscendC::Muls(normLocal[r * D], normLocal[r * D],
                      betaLocal.GetValue(r), D);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
    AscendC::DataCopy(Slice(kWsVBeta), normLocal, rows * D);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);

    // exp(g_i[-1] - g_i) * k_i, the state update's left operand. Its
    // transpose is the operand, which the matmul takes care of.
    const uint32_t lastRow = (rows - 1) * D;
    for (uint32_t r = 0; r < rows; ++r) {
        AscendC::Sub(tmpLocal[r * D], gcLocal[lastRow], gcLocal[r * D], D);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(tmpLocal, tmpLocal, rows * D);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(tmpLocal, tmpLocal, knLocal, rows * D);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
    AscendC::DataCopy(Slice(kWsKDecay), tmpLocal, rows * D);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
}

// ── v_i, o and the state ─────────────────────────────────────────────
//
//     v_i = u_i - w_i @ S
//     o = (q_i * g_i.exp()) @ S + Aqk @ v_i
//     S = S * g_i[-1].exp() + <state update from the Cube>
//
// Sums the Cube's five results, writes the output rows, advances the state.
template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::FinishChunk(
    uint32_t head, int32_t tok0, uint32_t rows)
{
    using namespace kda_chunk;
    constexpr uint32_t D = config::D;

    // ── o = state term + Aqk @ v_i ───────────────────────────────────
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
    AscendC::DataCopy(normLocal, Slice(kWsOut), rows * D);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
    AscendC::DataCopy(tmpLocal, Slice(kWsTmpBT_D), rows * D);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
    AscendC::Add(normLocal, normLocal, tmpLocal, rows * D);
    AscendC::PipeBarrier<PIPE_V>();

    // ── store(p_o, b_o) ─────────────────────────────────────────────
    //
    // Back to bf16, rows strided by H in GM.
    constexpr uint32_t HALF = config::BT / 2;
    const uint64_t outBase = static_cast<uint64_t>(tok0) * numHeads_ * D
                             + static_cast<uint64_t>(head) * D;
    for (uint32_t r0 = 0; r0 < rows; r0 += HALF) {
        const uint32_t n = (rows - r0 < HALF) ? (rows - r0) : HALF;
        AscendC::Cast(stageLocal, normLocal[r0 * D],
                      AscendC::RoundMode::CAST_RINT, n * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopyParams p{
            static_cast<uint16_t>(n),
            static_cast<uint16_t>(D * sizeof(bfloat16_t) / 32),
            0,
            static_cast<uint16_t>((numHeads_ - 1) * D *
                                  sizeof(bfloat16_t) / 32)};
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3);
        AscendC::DataCopy(
            outGm[outBase + static_cast<uint64_t>(r0) * numHeads_ * D],
            stageLocal, p);
        // The next half's Cast overwrites `stage`, so it waits for this
        // store to have read it.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V);
    }
}


#if KDA_CHUNK_INVERSE_SQUARING

// One squaring step's bookkeeping: move X @ (I + D) into X and D @ D into
// `kWsDpow`, adding the identity so the next product's operand is (I + D).
template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::AdvanceInverse(
    uint32_t rows)
{
    using namespace kda_chunk;
    constexpr uint32_t BT = config::BT;

    CopyWs(kWsX, kWsScratch, rows * BT);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
    AscendC::DataCopy(tmpLocal, Slice(kWsTmpBT_D), rows * BT);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
    AscendC::DataCopy(Slice(kWsDpow), tmpLocal, rows * BT);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
    AddIdentity(tmpLocal, rows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
    AscendC::DataCopy(Slice(kWsFactor), tmpLocal, rows * BT);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
}

#else  // blocked forward substitution

// ── one 32x32 diagonal inverse, by forward substitution ──────────────
//
//     for i in range(2, BC):
//         b_a = -Akk[i, :]  masked to columns [0, i)
//         b_a += sum_j b_a[j] * b_Ai[j, :]
//         b_Ai[i, :] = b_a
//
// `chunk_kda_fwd_kernel_diag_solve_npu`, on the block at (r0, r0) of the
// resident Araw tile. The row reduction becomes i-1 `Axpy` calls; the md
// covers why one Scalar visit per row is enough and why there are no PIPE_V
// barriers between them.
template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::SubstituteBlock(
    uint32_t r0, uint32_t rows)
{
    using namespace kda_chunk;
    constexpr uint32_t BT = config::BT;
    const uint32_t n = rows < kInvBlock ? rows : kInvBlock;

    // The block is on the diagonal, so its columns start at r0 as well.
    const uint32_t base = r0 * BT + r0;

    float coeff[kInvBlock];
    for (uint32_t i = 2; i < n; ++i) {
        const uint32_t rowI = base + i * BT;
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_V_S);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_V_S);
        for (uint32_t j = 1; j < i; ++j) {
            coeff[j] = tmpLocal.GetValue(rowI + j);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_S_V);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_S_V);
        // Row j is nonzero only on [0, j), so each covers j lanes.
        for (uint32_t j = 1; j < i; ++j) {
            AscendC::Axpy(tmpLocal[rowI], tmpLocal[base + j * BT], coeff[j],
                          j);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
}

// ── the worker AIV's share: C00, and the Cube's operands ─────────────
//
// Publishes B10 (the Cube's operand) and B11 (the helper's block), solves
// C00, and leaves rows [0, 32) of `kWsX` in their final form.
template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::SolveDiagBlocks(
    uint32_t rows)
{
    using namespace kda_chunk;
    constexpr uint32_t BT = config::BT;
    constexpr uint32_t NB = kInvBlock;

    // Reload Araw: the seeding block stored it to kWsDpow, but tmpLocal was
    // subsequently overwritten by the Aqk masking step.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2);
    AscendC::DataCopy(tmpLocal, Slice(kWsDpow), BT * BT);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);

    // ── B10 and B11, published before either is solved ───────────────
    //
    // Both packed, B11 because the helper has no UB copy of the chunk. B10
    // is already negated, so the Cube's +C11 B10 C00 is the reference's
    // -C11 Akk10 C00.
    StoreBlock(kWsInvB10, tmpLocal[NB * BT], NB, NB, BT);
    StoreBlock(kWsInvB11, tmpLocal[NB * BT + NB], NB, NB, BT);
    AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(kFlagAivToAiv);
    AscendC::CrossCoreWaitFlag(kFlagAivToAiv);

    // ── C00, solved here while the helper solves C11 ──────────────────
    SubstituteBlock(0, rows);
    AddIdentity(tmpLocal, NB);

    // ── rows [0, 32) of kWsX = [ C00 | 0 ] ───────────────────────────
    //
    // C00 is already in place; this clears (0, 32), which still holds Araw's
    // masked-out noise and which the auxiliary matmuls would read.
    for (uint32_t i = 0; i < NB; ++i) {
        AscendC::Duplicate(tmpLocal[i * BT + NB], 0.0f, NB);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
    AscendC::DataCopy(Slice(kWsX), tmpLocal, NB * BT);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
}

// ── the helper AIV's share: C11 ──────────────────────────────────────
//
// Reads B11 back from the workspace into (0, 32) of its own `tmp`, where
// that block sits in the [BT, BT] layout, and leaves rows [32, 64) of
// `kWsX` less the C10 hole the Cube fills.
template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::SolveDiagBlockHelper()
{
    using namespace kda_chunk;
    constexpr uint32_t BT = config::BT;
    constexpr uint32_t NB = kInvBlock;

    AscendC::CrossCoreSetFlag<0x1, PIPE_MTE3>(kFlagAivToAiv);
    AscendC::CrossCoreWaitFlag(kFlagAivToAiv);

    // B11 into (0, 32) of a BT-wide tile: 32 rows of 32, strided by BT.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2);
    {
        AscendC::DataCopyParams p{
            static_cast<uint16_t>(NB),
            static_cast<uint16_t>(NB * sizeof(float) / 32),
            0,
            static_cast<uint16_t>((BT - NB) * sizeof(float) / 32)};
        AscendC::DataCopy(tmpLocal[NB], Slice(kWsInvB11), p);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);

    // The C10 hole; the Cube overwrites it, so this is only to keep the
    // store from carrying garbage.
    for (uint32_t i = 0; i < NB; ++i) {
        AscendC::Duplicate(tmpLocal[i * BT], 0.0f, NB);
    }
    AscendC::PipeBarrier<PIPE_V>();

    // `SubstituteBlock`'s loop with the block at (0, 32) rather than on the
    // diagonal, so its (r0, r0) anchor does not apply.
    float coeff[kInvBlock];
    for (uint32_t i = 2; i < NB; ++i) {
        const uint32_t rowI = i * BT + NB;
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_V_S);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_V_S);
        for (uint32_t j = 1; j < i; ++j) {
            coeff[j] = tmpLocal.GetValue(rowI + j);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_S_V);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_S_V);
        for (uint32_t j = 1; j < i; ++j) {
            AscendC::Axpy(tmpLocal[rowI], tmpLocal[j * BT + NB], coeff[j], j);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // The identity, on a diagonal starting at column NB.
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_V_S);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_V_S);
    for (uint32_t r = 0; r < NB; ++r) {
        const uint32_t i = r * BT + NB + r;
        tmpLocal.SetValue(i, tmpLocal.GetValue(i) + 1.0f);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_S_MTE3);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_S_MTE3);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
    AscendC::DataCopy(Slice(kWsX + NB * BT), tmpLocal, NB * BT);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
}

#endif  // KDA_CHUNK_INVERSE_SQUARING

// exp(g_c[r-1] - g_c) * k_c over the first `r` rows: the state update's left
// operand, rebased on row r-1 instead of the chunk's last row. `r == rows`
// reproduces the chunk-final form.
template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::RebaseKDecay(uint32_t r)
{
    using namespace kda_chunk;
    constexpr uint32_t D = config::D;

    const uint32_t refRow = (r - 1) * D;
    for (uint32_t c = 0; c < r; ++c) {
        AscendC::Sub(tmpLocal[c * D], gcLocal[refRow], gcLocal[c * D], D);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(tmpLocal, tmpLocal, r * D);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(tmpLocal, tmpLocal, knLocal, r * D);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
    AscendC::DataCopy(Slice(kWsKDecay), tmpLocal, r * D);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
}

// ── S = S * g_c[refRow].exp() + increment ────────────────────────────
//
//     S = S * g_i[-1].exp()[:, None] + <increment>
//
// The decay is indexed by the key axis, the column in the transposed tile,
// so each [V, K] row takes the whole vector. One row at a time: the tile is
// 64 KB and no UB buffer here is larger than 32 KB.
template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::ApplyStateUpdate(
    uint32_t refRow, uint32_t rows)
{
    using namespace kda_chunk;
    constexpr uint32_t D = config::D;

    // exp(g_c[refRow]) into the first row of the second working tile.
    AscendC::Exp(normLocal, gcLocal[refRow * D], D);
    AscendC::PipeBarrier<PIPE_V>();

    // Rows of `tmp`, not `qn`/`kn`: the tracked-prefix path calls this and
    // then `RebaseKDecay`, which still needs `kn`.
    auto accLocal = tmpLocal;
    auto incLocal = tmpLocal[D];
    for (uint32_t row = 0; row < D; ++row) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
        AscendC::DataCopy(accLocal, Slice(kWsState + row * D), D);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
        AscendC::DataCopy(incLocal, Slice(kWsStateUpd + row * D), D);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
        AscendC::Mul(accLocal, accLocal, normLocal, D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(accLocal, accLocal, incLocal, D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
        AscendC::DataCopy(Slice(kWsState + row * D), accLocal, D);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
    }
}

// S -> initial_state_source[slot, head]. Verbatim: the pool slot is [V][K]
// in memory, which is the orientation `kWsState` keeps.
template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::CopyStateOut(
    uint32_t slot, uint32_t head)
{
    using namespace kda_chunk;
    constexpr uint32_t D = config::D;
    const uint64_t pstate =
        (static_cast<uint64_t>(slot) * numHeads_ + head) * D * D;
    for (uint32_t row = 0; row < D; ++row) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
        AscendC::DataCopy(tmpLocal, Slice(kWsState + row * D), D);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3);
        AscendC::DataCopy(stateGm[pstate + row * D], tmpLocal, D);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V);
    }
}

// Undo a tracked-prefix state update: the chunk's own has to run from the
// state as it was at the chunk's start.
template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::RestoreState()
{
    using namespace kda_chunk;
    constexpr uint32_t D = config::D;
    CopyWs(kWsState, kWsStateSave, D * D);
}

// Workspace-to-workspace copy, through UB because there is no GM-to-GM move.
template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::CopyWs(
    uint32_t dstOff, uint32_t srcOff, uint32_t count)
{
    using namespace kda_chunk;
    constexpr uint32_t CHUNKED = config::BT * config::D;
    for (uint32_t done = 0; done < count; done += CHUNKED) {
        const uint32_t n = (count - done < CHUNKED) ? (count - done) : CHUNKED;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2);
        AscendC::DataCopy(tmpLocal, Slice(srcOff + done), n);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_V_MTE3);
        AscendC::DataCopy(Slice(dstOff + done), tmpLocal, n);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_MTE3_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_MTE3_V);
    }
    // `tmp` is the staging buffer, so the caller's next Vector write to it
    // waits for the last store to have read it.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V);
}

// The AIV's half: everything elementwise, in the order the Cube needs it,
// with a handshake at each boundary. One chunk per pass of the loop.
template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::RunAiv()
{
    using namespace kda_chunk;
    constexpr uint32_t D = config::D;
    constexpr uint32_t BT = config::BT;

    // `GetBlockIdx()` on an AIV is a global AIV index (0 .. 2*blockDim-1),
    // not the Cube block, so the block is the index halved.
    const uint32_t aivIdx = AscendC::GetBlockIdx();
    const uint32_t blockIdx = aivIdx / 2;
    const uint32_t head = blockIdx % numHeads_;
    const uint32_t req = blockIdx / numHeads_;
    const bool worker = (AscendC::GetSubBlockIdx() == 0);
    if (req >= numReq_) {
        return;
    }
    wsBase_ = static_cast<uint64_t>(blockIdx) * kWorkspaceFloatsPerBlock;

    const int32_t bos = startsGm.GetValue(req);
    const int32_t eos = startsGm.GetValue(req + 1);
    const int32_t slot = indicesGm.GetValue(req);

    int32_t trackIndex = 0;
    int32_t trackLen = -1;
    if constexpr (TRACK_STATE) {
        trackIndex = trackIndicesGm.GetValue(req);
        if (trackIndex > 0) {
            trackLen = trackLensGm.GetValue(req);
        }
    }

    // Both AIVs: it sets the vector mask to all-ones, and the helper runs
    // vector work of its own under the substitution inverse.
    AscendC::InitSocState();

    if (worker) {
        // ── b_dt_bias, and lower_bound as a tensor for the gate's Div ──
        AscendC::DataCopy(biasLocal, dtBiasGm[head * D], D);
        AscendC::Duplicate(lbRowLocal, lowerBound_, D);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);

        // ── S = initial_state_source[slot] if slot > 0 else zeros ──────
        //
        // The slot is [K, V] with K contiguous, i.e. [V][K] in memory, which
        // is the orientation `kWsState` keeps, so the copy is verbatim.
        const uint64_t pstate =
            (static_cast<uint64_t>(slot) * numHeads_ + head) * D * D;
        for (uint32_t row = 0; row < D; ++row) {
            if (slot > 0) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2);
                AscendC::DataCopy(tmpLocal, stateGm[pstate + row * D], D);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
            } else {
                AscendC::Duplicate(tmpLocal, 0.0f, D);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
            AscendC::DataCopy(Slice(kWsState + row * D), tmpLocal, D);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
        }

        // ── track_len == 0 writes the initial state ───────────────────
        if constexpr (TRACK_STATE) {
            if (trackIndex > 0 && trackLen == 0) {
                CopyStateOut(static_cast<uint32_t>(trackIndex), head);
            }
        }
    }

    for (int32_t tok0 = bos; tok0 < eos; tok0 += static_cast<int32_t>(BT)) {
        const uint32_t rows = BT;

        if (worker) {
            LoadChunk(head, tok0, rows);
            PrepareFactors(head, rows);
            WriteRebased(rows);
        }

        // ══ A, Aqk on the Cube ══════════════════════════════════════════
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kFlagAivToAic);
        AscendC::CrossCoreWaitFlag(kFlagAicToAiv);

        if (worker) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
            // ── the masks and the negation ───────────────────────────────────────
            //
            //     A = A * beta_c[..., None]
            //     A = -A.masked_fill(triu(diagonal=0), 0)
            //     Aqk = Aqk.masked_fill(triu(diagonal=1), 0)
            //
            // ── Araw = -(A * beta).tril(-1) ──────────────────────────────────
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
            AscendC::DataCopy(tmpLocal, Slice(kWsA), BT * BT);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
            for (uint32_t r = 0; r < rows; ++r) {
                AscendC::Muls(tmpLocal[r * BT], tmpLocal[r * BT],
                              -betaLocal.GetValue(r), BT);
            }
            AscendC::PipeBarrier<PIPE_V>();
            // tril(-1): zero [r, BT) on row r.
            for (uint32_t r = 0; r < rows; ++r) {
                uint64_t mask[2] = { ~((1ULL << r) - 1ULL), 0ULL };
                AscendC::Duplicate<float>(tmpLocal[r * BT], 0.0f, mask, 1, 1, 1);
            }
            AscendC::ResetMask();

            // Araw: the squaring chain's seed, and the tile the substitution
            // path's `SolveDiagBlocks` reloads. The substitution path seeds
            // nothing else — the identity enters inside each diagonal solve,
            // and both halves of kWsX are written in full before the Cube
            // reads it.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
            AscendC::DataCopy(Slice(kWsDpow), tmpLocal, BT * BT);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);

#if KDA_CHUNK_INVERSE_SQUARING
            // I + Araw is the chain's first factor.
            AddIdentity(tmpLocal, BT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
            AscendC::DataCopy(Slice(kWsFactor), tmpLocal, BT * BT);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);

            // X seeds as the identity, not as I + Araw: that factor is the chain's
            // s == 0 product and seeding X with it applies it twice.
            AscendC::Duplicate(tmpLocal, 0.0f, BT * BT);
            AscendC::PipeBarrier<PIPE_V>();
            AddIdentity(tmpLocal, BT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
            AscendC::DataCopy(Slice(kWsX), tmpLocal, BT * BT);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
#endif

            // ── Aqk = Aqk.masked_fill(triu(1), 0) ───────────────────────────
            //
            // Last, because it overwrites the Araw the seeding above needs.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
            AscendC::DataCopy(tmpLocal, Slice(kWsAqk), BT * BT);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
            // Aqk triu(1): zero [r+1, BT) on row r.
            for (uint32_t r = 0; r + 1 < rows; ++r) {
                uint64_t mask[2] = { ~((1ULL << (r + 1)) - 1ULL), 0ULL };
                AscendC::Duplicate<float>(tmpLocal[r * BT], 0.0f, mask, 1, 1, 1);
            }
            AscendC::ResetMask();
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
            AscendC::DataCopy(Slice(kWsAqk), tmpLocal, BT * BT);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
        }

#if KDA_CHUNK_INVERSE_SQUARING
        // ══ the inverse: 6 squaring steps on the Cube ═══════════════════
        //
        // Between steps the AIV turns the Cube's two results back into the
        // next step's operands.
        for (uint32_t s = 0; s < kInverseSteps; ++s) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kFlagAivToAic);
            AscendC::CrossCoreWaitFlag(kFlagAicToAiv);
            if (worker) {
                if (s + 1 < kInverseSteps) {
                    AdvanceInverse(rows);
                } else {
                    // The last product is the inverse; no factor follows it.
                    CopyWs(kWsX, kWsScratch, rows * BT);
                }
            }
        }
        if (worker) {
            BuildAuxOperands(head, tok0, rows);
        }

        // ══ w, u, w @ S ════════════════════════════════════════════════
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kFlagAivToAic);
        AscendC::CrossCoreWaitFlag(kFlagAicToAiv);
#else
        // ══ the inverse: two 32x32 substitutions, then one Cube pair ════
        //
        //     C00, C11  forward substitution, one block per AIV
        //     C10 = -C11 B10 C00, two Cube matmuls
        //
        // One AIV -> AIC handoff, and no reply: the AIV goes straight on to
        // the auxiliary operands, which the Cube's two inversion matmuls
        // then overlap. See the md for why, and for the measurements.
        if (worker) {
            SolveDiagBlocks(rows);
        } else {
            SolveDiagBlockHelper();
        }
        // Mode 2 is a quorum: the AIC's wait clears only once *both* AIVs have
        // set this ID, so this one Set per core is what orders C00 and C11
        // against the Cube's C10 pair. Both AIVs must reach it unconditionally.
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kFlagAivToAic);

        if (worker) {
            BuildAuxOperands(head, tok0, rows);
        }

        // ══ w, u, w @ S ════════════════════════════════════════════════
        //
        // The aux ID, so the Cube takes the two forward messages in order
        // without the inversion needing a reply to close its pair.
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kFlagAivToAicAux);
        AscendC::CrossCoreWaitFlag(kFlagAicToAiv);
#endif

        if (worker) {
            // v_i = u_i - w_i @ S
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
            AscendC::DataCopy(normLocal, Slice(kWsU), rows * D);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(kda_chunk::EVENT_V_MTE2);
            AscendC::DataCopy(tmpLocal, Slice(kWsTmpBT_D), rows * D);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V);
            AscendC::Sub(normLocal, normLocal, tmpLocal, rows * D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_chunk::EVENT_V_MTE3);
            AscendC::DataCopy(Slice(kWsVNew), normLocal, rows * D);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_chunk::EVENT_MTE3_V);
        }

        // ══ o's two terms and the state increment ══════════════════════
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kFlagAivToAic);
        AscendC::CrossCoreWaitFlag(kFlagAicToAiv);

        if (worker) {
            FinishChunk(head, tok0, rows);
        }

        // ══ the tracked prefix state ═══════════════════════════════════
        //
        //     if i*BT < track_len <= (i+1)*BT
        //
        // The AIV rebuilds `kWsKDecay` against row r-1 and the Cube redoes
        // the increment over r rows.
        if constexpr (TRACK_STATE) {
            const int32_t consumed = tok0 - bos;
            if (trackIndex > 0 && consumed < trackLen &&
                trackLen <= consumed + static_cast<int32_t>(BT)) {
                const uint32_t r = static_cast<uint32_t>(trackLen - consumed);
                if (worker) {
                    CopyWs(kWsStateSave, kWsState, D * D);
                    RebaseKDecay(r);
                }
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kFlagAivToAic);
                AscendC::CrossCoreWaitFlag(kFlagAicToAiv);
                if (worker) {
                    ApplyStateUpdate(r - 1, rows);
                    CopyStateOut(static_cast<uint32_t>(trackIndex), head);
                    // The chunk's own update still has to run from the
                    // unmodified state, so put it back.
                    RestoreState();
                    RebaseKDecay(rows);
                }
                // The prefix increment landed in `kWsStateUpd`, overwriting
                // the chunk's own; the Cube redoes it over all BT rows.
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(kFlagAivToAic);
                AscendC::CrossCoreWaitFlag(kFlagAicToAiv);
            }
        }

        if (worker) {
            ApplyStateUpdate(rows - 1, rows);
        }
    }

    // ── store(p_h0, S) if slot > 0 ───────────────────────────────────
    if (worker && slot > 0) {
        CopyStateOut(static_cast<uint32_t>(slot), head);
    }
}

// The AIC's half of the chunk: eleven matmuls for the inverse, then the
// auxiliaries, the output and the state update. Every operand is read from
// and written to this block's workspace slice, and each group is bracketed by
// the handshake that orders it against the AIV's work on the same buffers.
template <bool TRACK_STATE>
__aicore__ inline void KdaVarlenChunk<TRACK_STATE>::RunAic()
{
    using namespace kda_chunk;
    constexpr uint32_t D = config::D;
    constexpr uint32_t BT = config::BT;
    constexpr uint32_t SB = config::SB;
    constexpr uint32_t NB = config::NB;

    const uint32_t blockIdx = AscendC::GetBlockIdx();
    const uint32_t head = blockIdx % numHeads_;
    const uint32_t req = blockIdx / numHeads_;
    if (req >= numReq_) {
        return;
    }
    const int32_t bos = startsGm.GetValue(req);
    const int32_t eos = startsGm.GetValue(req + 1);
    wsBase_ = static_cast<uint64_t>(blockIdx) * kWorkspaceFloatsPerBlock;

    for (int32_t tok0 = bos; tok0 < eos; tok0 += static_cast<int32_t>(BT)) {
        const uint32_t rows = BT;

        // ══ A, Aqk ══════════════════════════════════════════════════════
        //
        //     A[..., i] = einsum(k_c * (g_c - g_i).exp(), k_i)
        //     Aqk[..., j] = einsum(q_i * (g_i - g_j).exp(), k_j)
        //
        // One matmul per rebase block, over columns [0, r1). Columns past r1
        // are above the diagonal and the AIV masks them.
        AscendC::CrossCoreWaitFlag(kFlagAivToAic);
        for (uint32_t R = 0; R < NB; ++R) {
            const uint32_t r0 = R * SB;
            const uint32_t r1 = r0 + SB;
            // The result is a [SB, r1] block of a [BT, BT] tile, so the row
            // pitch is BT, not the block's own width.
            CubeMatmul<false, true>(Slice(kWsRowK + r0 * D),
                                    Slice(kWsColK + R * kTileBT_D),
                                    Slice(kWsA + r0 * BT), SB, r1, D, BT);
            CubeMatmul<false, true>(Slice(kWsRowQ + r0 * D),
                                    Slice(kWsColK + R * kTileBT_D),
                                    Slice(kWsAqk + r0 * BT), SB, r1, D, BT);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kFlagAicToAiv);

        // ══ TRIANGULAR INVERSE ══════════════════════════════════════════
        //
        //     for i in range(1, BT):
        //         A[..., i, :i] += (A[..., i, :, None] * A[..., :, :i]).sum(-2)
        //     A = A + eye(BT)
#if KDA_CHUNK_INVERSE_SQUARING
        //
        // Repeated squaring instead of that substitution: Araw is strictly
        // lower triangular so Araw^BT == 0, and every k < BT has a unique
        // binary expansion, hence
        //     (I - Araw)^-1 = sum_k Araw^k = prod_s (I + Araw^(2^s)).
        // Six steps for BT=64. The AIV adds the identity into each factor
        // before the product, so what the Cube multiplies is (I + Araw^(2^s)).
        for (uint32_t s = 0; s < kInverseSteps; ++s) {
            AscendC::CrossCoreWaitFlag(kFlagAivToAic);
            // X <- X @ (I + D), the factor built by the AIV
            CubeMatmul<false, false>(Slice(kWsX), Slice(kWsFactor),
                                     Slice(kWsScratch), BT, BT, BT);
            // D <- D @ D, squaring the nilpotent part. The last step's
            // factor is never used, so the squaring is skipped there.
            if (s + 1 < kInverseSteps) {
                CubeMatmul<false, false>(Slice(kWsDpow), Slice(kWsDpow),
                                         Slice(kWsTmpBT_D), BT, BT, BT);
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kFlagAicToAiv);
        }
#else
        //
        //     b_Ai10 = -dot(dot(b_Ai11, b_Akk10), b_Ai00)
        //
        // `chunk_kda_fwd_kernel_inter_solve_fused_npu`'s off-diagonal merge.
        // C00 and C11 are read in place out of `kWsX` and C10 lands in its
        // hole, hence the BT pitches; B10 arrives negated, so no sign flip.
        constexpr uint32_t NB = kInvBlock;
        // One wait: the mode-2 quorum already means both AIVs have stored
        // their half of kWsX before this clears.
        AscendC::CrossCoreWaitFlag(kFlagAivToAic);
        // C11 @ B10 — C11 at (32, 32) of kWsX, B10 packed
        CubeMatmul<false, false>(Slice(kWsX + NB * BT + NB),
                                 Slice(kWsInvB10), Slice(kWsInvTmp),
                                 NB, NB, NB, 0, false, BT, 0);
        // kWsInvTmp is the first matmul's Fixpipe output and the second's MTE2
        // input; CubeMatmul's internal barriers order on-chip buffers only.
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(EVENT_ID0);
        // (C11 B10) @ C00, into the hole at (32, 0) of kWsX
        CubeMatmul<false, false>(Slice(kWsInvTmp), Slice(kWsX),
                                 Slice(kWsX + NB * BT), NB, NB, NB, BT,
                                 false, 0, BT);
#endif

        // ══ AUXILIARIES, OUTPUT, STATE ══════════════════════════════════
        //
        //     w = A @ (g_c.exp() * k_c)
        //     u = A @ v_c
        //     v_i = u_i - w_i @ S
        //     o = (q_i * g_i.exp()) @ S + Aqk @ v_i
        //     S = S * g_i[-1].exp() + ((g_i[-1] - g_i).exp() * k_i).T @ v_i
        //
        // `A` is the inverse scaled by beta on the right, which the AIV
        // folds into the right operand: Ainv @ M == X @ (beta * M).
#if KDA_CHUNK_INVERSE_SQUARING
        AscendC::CrossCoreWaitFlag(kFlagAivToAic);
#else
        // The aux ID: the inversion's handoff came in on `kFlagAivToAic` and
        // was not replied to, so this is the second forward message.
        AscendC::CrossCoreWaitFlag(kFlagAivToAicAux);
#endif
        CubeMatmul<false, false>(Slice(kWsX), Slice(kWsKgBeta),
                                 Slice(kWsW), BT, D, BT);
        CubeMatmul<false, false>(Slice(kWsX), Slice(kWsVBeta),
                                 Slice(kWsU), BT, D, BT);
        // w_i @ S, landing where the AIV subtracts it from u. S is stored
        // transposed, which is exactly the B_TRANS orientation.
        CubeMatmul<false, true>(Slice(kWsW), Slice(kWsState),
                                Slice(kWsTmpBT_D), BT, D, D);
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kFlagAicToAiv);

        AscendC::CrossCoreWaitFlag(kFlagAivToAic);
        // (q_i * g_i.exp()) @ S, S transposed in the workspace
        CubeMatmul<false, true>(Slice(kWsQg), Slice(kWsState),
                                Slice(kWsOut), BT, D, D);
        // Aqk @ v_i
        CubeMatmul<false, false>(Slice(kWsAqk), Slice(kWsVNew),
                                 Slice(kWsTmpBT_D), BT, D, BT);
        // The state increment, transposed to match S:
        //     (kdecay.T @ vnew).T == vnew.T @ kdecay
        CubeMatmul<true, false>(Slice(kWsVNew), Slice(kWsKDecay),
                                Slice(kWsStateUpd), D, D, BT);
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kFlagAicToAiv);

        // ══ TRACKED PREFIX STATE ════════════════════════════════════════
        //
        //     S_r = S * g_r.exp() + ((g_r - g_i[:r]).exp() * k_i[:r]).T @ v_i[:r]
        //
        // Same shape as the chunk update with r rows instead of BT. Only the
        // chunk containing `track_len` runs it; the AIV decides and signals.
        if constexpr (TRACK_STATE) {
            const int32_t trackIndex = trackIndicesGm.GetValue(req);
            const int32_t trackLen = trackLensGm.GetValue(req);
            const int32_t consumed = tok0 - bos;
            if (trackIndex > 0 && consumed < trackLen &&
                trackLen <= consumed + static_cast<int32_t>(BT)) {
                const uint32_t r =
                    static_cast<uint32_t>(trackLen - consumed);
                AscendC::CrossCoreWaitFlag(kFlagAivToAic);
                CubeMatmul<true, false>(Slice(kWsVNew), Slice(kWsKDecay),
                                        Slice(kWsStateUpd), D, D, r);
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kFlagAicToAiv);

                // The prefix increment overwrote the chunk's own, so redo it
                // over all BT rows against the AIV's rebuilt `kWsKDecay`.
                AscendC::CrossCoreWaitFlag(kFlagAivToAic);
                CubeMatmul<true, false>(Slice(kWsVNew), Slice(kWsKDecay),
                                        Slice(kWsStateUpd), D, D, BT);
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(kFlagAicToAiv);
            }
        }
    }
}

// ── Entry points ──────────────────────────────────────────────────────
//
// Two kernels differing only in `TRACK_STATE`. The macro exists because a
// `__global__` kernel cannot be a template.
#define KDA_CHUNK_KERNEL(NAME, TRACK)                                   \
extern "C" __global__ __aicore__                                        \
void NAME(                                                              \
    GM_ADDR Q, GM_ADDR K, GM_ADDR V, GM_ADDR A, GM_ADDR B,              \
    GM_ADDR A_LOG, GM_ADDR DT_BIAS,                                     \
    GM_ADDR STATE, GM_ADDR INDICES, GM_ADDR STARTS, GM_ADDR OUT,        \
    GM_ADDR TRACK_INDICES, GM_ADDR TRACK_LENS, GM_ADDR WORKSPACE,       \
    float scale, float lowerBound,                                      \
    uint32_t numReq, uint32_t numHeads                                  \
) {                                                                     \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                  \
    KdaChunkLaunchArgs args{                                            \
        Q, K, V, A, B, A_LOG, DT_BIAS, STATE, INDICES, STARTS, OUT,     \
        TRACK_INDICES, TRACK_LENS, WORKSPACE,                           \
        scale, lowerBound, numReq, numHeads, TRACK};                    \
    KdaVarlenChunk<TRACK> op;                                           \
    op.Init(args, numHeads);                                            \
    op.Process();                                                       \
}

KDA_CHUNK_KERNEL(kda_chunk_plain, false)
KDA_CHUNK_KERNEL(kda_chunk_track, true)

#ifndef __CCE_KT_TEST__

// One block per (request, head). `blockDim` counts Cube blocks in
// KERNEL_TYPE_MIX_AIC_1_2, and each gets two AIVs.
void LaunchKdaVarlenChunk(const KdaChunkLaunchArgs &args, uint32_t num_heads,
                          void *stream)
{
    const uint32_t blockDim = args.num_req * num_heads;
    auto acl_stream = static_cast<aclrtStream>(stream);

#define KDA_CHUNK_CALL(KERNEL)                                          \
    KERNEL<<<blockDim, nullptr, acl_stream>>>(                          \
        args.q, args.k, args.v, args.a, args.b,                         \
        args.A_log, args.dt_bias,                                       \
        args.state, args.indices, args.starts, args.out,                \
        args.track_indices, args.track_lens, args.workspace,            \
        args.scale, args.lower_bound, args.num_req, num_heads)

    if (args.track_state) {
        KDA_CHUNK_CALL(kda_chunk_track);
    } else {
        KDA_CHUNK_CALL(kda_chunk_plain);
    }

#undef KDA_CHUNK_CALL
}

#endif
