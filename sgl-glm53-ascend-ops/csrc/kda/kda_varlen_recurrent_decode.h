// GLM-5.3 KDA varlen recurrence — decode kernel, shared across H=16/32/64.
//
// Device-side only. Included by the three decode translation units, each of
// which instantiates it for its own H and defines its launcher. See
// `kda_varlen_recurrent.h` for the shared contract and the reference semantics.
//
// ── What decode actually is ────────────────────────────────────────────
//
// Not "one token per request". The decode-side caller has two regimes and the
// speculative one dominates the profile:
//
//   forward_decode        T == numReq,              no intermediate
//   target verify         T == numReq * draftTok,   intermediate required
//
// Under EAGLE the verify path runs `draft_token_num` (4 in the deployed config)
// sequential recurrence steps per request and checkpoints the state after every
// one, because the post-verify commit has to be able to pick up at whichever
// step the draft was accepted to. Both regimes come through as a packed token
// stream split by `cu_seqlens`, so a single loop over `[bos, eos)` covers them;
// the step count is a runtime value, never a template parameter.
//
// ── Why the tiling is the opposite of prefill's ────────────────────────
//
// Prefill splits a head across 8 cores at BV=16, because it has one packed
// sequence and needs parallelism from somewhere. Decode has `numReq * H`
// independent work items before any value split, so it does not, and the split
// is pure overhead there:
//
//   * q, k, the gate row and beta are per (token, head) — shared by every
//     value tile of a head. Splitting a head 8 ways computes the gate, the two
//     L2 norms and the sigmoid 8 times over, and each of those is a
//     Vector->Scalar round trip, which this architecture pays dearly for
//     (no scalar `exp`).
//   * the per-step instruction count is fixed, so a narrow tile pays the same
//     PipeBarrier and issue latency for an eighth of the work. The prefill
//     consumer measures ~3000 cycles/token at BV=16 for ~162 repeat-iterations
//     of actual vector work — that ratio is the overhead a coarse tile
//     amortizes.
//
// So one core owns one (request, head): BV = 128, the whole value axis, with
// the 64 KB state tile resident in UB for the entire token loop. Measured on
// the Triton baseline the same preference shows up as a monotonic win for
// coarse tiles (BV 16/32/64 -> 85.2/52.6/40.3 us on the priority shape), and it
// stops there only because BV=128 does not compile under Triton at all — the
// [BV,BK] tile plus its live temporaries overflow UB in the BiShengIR pipeline
// ("requires 2383872 bits while 1572864 bits available"). An Ascend C kernel
// with a hand-laid map is not bound by that: the state tile and its reduction
// scratch are 96 KB of the 192 KB budget, and nothing else here is large.
//
// This holds even where it leaves the device half idle. The 512K shapes give
// the grid 32 and 16 blocks against 40 cores, and splitting their heads to fill
// it measures worse, not better (see `KDA_DEFINE_DECODE_WIDTH`). Those shapes
// are serial in the token index rather than short of parallelism, and a value
// split does not change the critical path — it just runs the per-token prologue
// several times.
//
// Keeping the state in UB across all steps is the other half of it. The
// recurrence is serial in the token index, so a per-step launch has to spill
// the state to GM and reload it; fusing the steps means one load, and the only
// GM traffic left is the checkpoints the caller actually asked for. On the
// priority shape four separate launches measure 78.4 us against 41.1 us fused.
//
// ── What this costs in traffic ────────────────────────────────────────
//
// Per (request, head): one 64 KB state load, then 64 KB per checkpoint. At
// H=32 with 2 requests and 4 steps that is 64 items x 320 KB = 20 MB, ~17 us
// at the ~1150 GB/s this device sustains. The checkpoints are 80% of it. That
// is the floor this kernel is working against, and it is a caller-side
// decision, not a tiling one: upstream's own ReplaySSM path
// (`enable_linear_replayssm_spec`) drops `intermediate_ssm` entirely and
// replays the ring at commit instead. Parity with the generic Triton kernel is
// what this file implements, so the checkpoints are written.
//
// Tracking is prefill-only, so the prefix-checkpoint store is compiled out and
// each H has two instantiations: plain, and save-intermediate.

#pragma once

#include "kernel_operator.h"
#include "kda_varlen_recurrent.h"
#include "utils.h"

namespace kda_decode {

// Hardware event IDs, managed by hand as in the prefill kernels. Avoid 6/7.
// Event ids are scoped per (source, destination) pipe pair, so the same id may
// appear against two different pairs.
//
// The checkpoint and the output get separate V_MTE3/MTE3_V ids so both can be
// in flight at once: the token loop issues the checkpoint and keeps going, and
// the two stores overlap the output reduction that follows them.
constexpr int32_t EVENT_MTE2_V_STATE = 0;
constexpr int32_t EVENT_MTE2_V_IN = 1;
constexpr int32_t EVENT_V_MTE3_OUT = 0;
constexpr int32_t EVENT_V_MTE3_MID = 1;
constexpr int32_t EVENT_MTE3_V_OUT = 0;
constexpr int32_t EVENT_MTE3_V_MID = 1;
constexpr int32_t EVENT_S_V = 0;
constexpr int32_t EVENT_V_S = 0;

// Tokens prepared per pass of the inner loop.
//
// The per-token inputs — the gate, beta, and the two L2 norms — are hoisted out
// of the recurrence and computed for a whole group at a time. The point is not
// the arithmetic, which is a few hundred lanes, but the Vector/Scalar
// round trips: beta needs a reciprocal and each norm needs a square root, both
// of which have to run on the Scalar unit to match upstream bit-for-bit, and
// every crossing drains a pipeline. Done per token that is four crossings each
// way; batched over a group it is one each way for the whole group, with all
// the Scalar work in the middle.
//
// Four is the deployed `draft_token_num`, so verify is exactly one group and
// plain decode is a one-row group. A longer sequence loops, which is what keeps
// the step count a runtime value.
constexpr uint32_t kStepGroup = 4;

}  // namespace kda_decode

// SAVE_INTERMEDIATE  checkpoint the state after every step (target verify).
//                    When set, the final state is *not* written back, matching
//                    the generic Triton kernel's `if not SAVE_INTERMEDIATE and
//                    state_index > 0` guard: the commit takes the accepted step
//                    out of the checkpoint buffer instead.
// H_TOTAL            heads per state slot; 16/32/64 for decode at TP4/TP2/TP1.
// V_TILES            cores per head, splitting the value axis. 1 unless the
//                    shape has too few (request, head) pairs to fill the
//                    device — see `PickVTiles` below.
template<bool SAVE_INTERMEDIATE, uint32_t H_TOTAL, uint32_t V_TILES>
class KdaVarlenRecurrentDecode {
public:
    __aicore__ inline void Init(
        GM_ADDR Q, GM_ADDR K, GM_ADDR V, GM_ADDR A,   // bf16 [1,T,H,D]
        GM_ADDR B,                                     // bf16 [1,T,H]
        GM_ADDR A_LOG,                                 // fp32 [H]
        GM_ADDR DT_BIAS,                               // fp32 [H*D]
        GM_ADDR STATE,                                 // fp32 [slots,H,K,V]
        GM_ADDR INDICES,                               // int32 [numReq]
        GM_ADDR STARTS,                                // int32 [numReq+1]
        GM_ADDR OUT,                                   // bf16 [1,T,H,D]
        GM_ADDR MID,                                   // fp32, K-contiguous
        uint32_t midStrideReq,                         // elements
        uint32_t midStrideStep,                        // elements
        uint32_t midStrideHead,                        // elements
        float scale,
        float lowerBound,
        uint32_t numReq);

    __aicore__ inline void Process();

private:
    struct config {
        static constexpr uint32_t D = static_cast<uint32_t>(config_D);
        // Normally the whole value axis on one core — a (request, head) is the
        // work item and there is no value split. Shapes with too few of those
        // to fill the device split the axis instead; see `PickVTiles`.
        static constexpr uint32_t NUM_V_TILES = V_TILES;
        static constexpr uint32_t BV = D / V_TILES;
        static constexpr uint32_t STATE_STRIDE = D;
        static constexpr uint32_t C = kda_decode::kStepGroup;

        // Vector repeat counts and strides are uint8 on this architecture.
        // BV is the repeat count of every state-wide instruction below and
        // STATE_STRIDE/8 is their repeat stride, so BV=128 sits inside the
        // limit but leaves no room: a wider tile would need a second pass.
        static_assert(BV <= 255, "BV is a uint8 repeat count");
        static_assert(STATE_STRIDE / 8 <= 255,
                      "the state row stride is a uint8 repeat stride");
        static_assert(BV % 8 == 0, "Brcb consumes 8 elements per repeat");
    };

    // UB map, laid out by hand at absolute addresses as in the prefill kernels.
    // The repeat strides in the token loop are derived from these offsets, so
    // moving one means revisiting the instruction that reads it.
    //
    // The state tile dominates: 128 rows x 128 fp32 = 64 KB, with a 32 KB
    // reduction scratch beside it. Everything after that is per-token rows and
    // scalars and is small; the whole map is ~116 KB of 192 KB.
    struct UbLayout {
        static constexpr uint32_t D = config::D;
        static constexpr uint32_t BV = config::BV;
        static constexpr uint32_t C = config::C;
        static constexpr uint32_t ROW_F32 = D * sizeof(float);     // 512
        static constexpr uint32_t ROW_BF16 = D * sizeof(bfloat16_t);

        // ── the resident state tile and its reduction scratch ──
        static constexpr uint32_t STATE_ADDR = 0x00000;
        static constexpr uint32_t STATE_SIZE =
            BV * config::STATE_STRIDE * sizeof(float);             // 64 KB
        // Holds the 64-lane partial products of a [BV, D] x [D] reduction:
        // the two halves of a row are folded in place, so this is half the
        // state tile, not the whole of it.
        static constexpr uint32_t MUL_REDUCE_ADDR = STATE_ADDR + STATE_SIZE;
        static constexpr uint32_t MUL_REDUCE_SIZE =
            BV * (D / 2) * sizeof(float);                          // 32 KB
        // The broadcast of v across datablocks, for the outer product.
        static constexpr uint32_t BRCB_ADDR = MUL_REDUCE_ADDR + MUL_REDUCE_SIZE;
        static constexpr uint32_t BRCB_SIZE = BV * 8 * sizeof(float);
        static constexpr uint32_t REDUCE_ADDR = BRCB_ADDR + BRCB_SIZE;
        static constexpr uint32_t TMP_BV_ADDR =
            REDUCE_ADDR + BV * sizeof(float);
        static constexpr uint32_t OUT_ADDR = TMP_BV_ADDR + BV * sizeof(float);

        // ── per-token inputs, one group of C tokens at a time ──
        // Hoisted out of the recurrence: the normalized q and k, exp(gate),
        // and the staged bf16 rows they are built from.
        static constexpr uint32_t QN_ADDR = OUT_ADDR + Align32(BV * sizeof(bfloat16_t));
        static constexpr uint32_t KN_ADDR = QN_ADDR + C * ROW_F32;
        static constexpr uint32_t GD_ADDR = KN_ADDR + C * ROW_F32;
        static constexpr uint32_t QB_ADDR = GD_ADDR + C * ROW_F32;
        static constexpr uint32_t KB_ADDR = QB_ADDR + C * ROW_BF16;
        static constexpr uint32_t AB_ADDR = KB_ADDR + C * ROW_BF16;
        static constexpr uint32_t VB_ADDR = AB_ADDR + C * ROW_BF16;

        // ── scalars and scratch ──
        // beta: the 1 + exp(-b) denominators Vector builds, for Scalar to take
        // the reciprocal of. A whole 64-lane repeat, because Exp works a
        // datablock at a time and the tail lanes must not be garbage.
        static constexpr uint32_t BRAW_ADDR = VB_ADDR + C * ROW_BF16;
        // the dt_bias row for this head, loaded once
        static constexpr uint32_t BIAS_ADDR = BRAW_ADDR + 64 * sizeof(float);
        // The group's squares, and the folded half-sums of q and k. Sized for
        // the whole group rather than one row, so the norm reduction runs as
        // one strided instruction per stage instead of a loop over tokens.
        static constexpr uint32_t TMP_ADDR = BIAS_ADDR + ROW_F32;
        static constexpr uint32_t RED_ADDR = TMP_ADDR + C * ROW_F32;
        // The group's squared-norm sums: q's at lane r, k's at lane 8+r, so
        // one `Brcb` covers both and `WholeReduceSum` can land each set
        // contiguously. The gap is what keeps the two sets in separate
        // datablocks. One Scalar visit takes all the square roots.
        static constexpr uint32_t NRM_ADDR = RED_ADDR + 2 * C * 64 * sizeof(float);
        // One repeat of lower_bound, the gate division's numerator: `Div`
        // takes both operands as tensors here and `Divs` is unsupported, so a
        // scalar numerator has to be materialized. A src0 repeat stride of 0
        // then reuses this one repeat for every row.
        static constexpr uint32_t LB_ADDR = NRM_ADDR + 64 * sizeof(float);
        // Scalar/Vector exchange slot: this architecture has no scalar `exp`,
        // so exp(A_log) is computed on the Vector unit and read back here.
        static constexpr uint32_t SCR_ADDR = LB_ADDR + 64 * sizeof(float);
        // The broadcast q/k norms, one row per token of the group, for the same
        // `Div` reason. A row per token rather than one shared row so the
        // group's division is a single instruction over C*D lanes.
        static constexpr uint32_t NORM_ADDR = SCR_ADDR + 64 * sizeof(float);
        static constexpr uint32_t UB_END = NORM_ADDR + C * ROW_F32;

        static_assert(UB_END <= 192 * 1024, "the decode UB map must fit");
    };

    AscendC::GlobalTensor<bfloat16_t> qGm;
    AscendC::GlobalTensor<bfloat16_t> kGm;
    AscendC::GlobalTensor<bfloat16_t> vGm;
    AscendC::GlobalTensor<bfloat16_t> aGm;
    AscendC::GlobalTensor<bfloat16_t> bGm;
    AscendC::GlobalTensor<float> aLogGm;
    AscendC::GlobalTensor<float> dtBiasGm;
    AscendC::GlobalTensor<bfloat16_t> outGm;
    AscendC::GlobalTensor<float> stateGm;
    AscendC::GlobalTensor<float> midGm;
    AscendC::GlobalTensor<int32_t> indicesGm;
    AscendC::GlobalTensor<int32_t> startsGm;

    float scale_ = 0.0f;
    float lowerBound_ = 0.0f;
    uint32_t numReq_ = 0;
    uint32_t midStrideReq_ = 0;
    uint32_t midStrideStep_ = 0;
    uint32_t midStrideHead_ = 0;

    AscendC::LocalTensor<float> stateLocal;
    AscendC::LocalTensor<float> mulReduceLocal;
    AscendC::LocalTensor<float> brcbLocal;
    AscendC::LocalTensor<float> reduceLocal;
    AscendC::LocalTensor<float> tmpBVLocal;
    AscendC::LocalTensor<bfloat16_t> outLocal;

    AscendC::LocalTensor<float> qnLocal;
    AscendC::LocalTensor<float> knLocal;
    AscendC::LocalTensor<float> gdLocal;
    AscendC::LocalTensor<bfloat16_t> qbLocal;
    AscendC::LocalTensor<bfloat16_t> kbLocal;
    AscendC::LocalTensor<bfloat16_t> abLocal;
    AscendC::LocalTensor<bfloat16_t> vbLocal;

    AscendC::LocalTensor<float> brawLocal;
    AscendC::LocalTensor<float> nrmLocal;
    AscendC::LocalTensor<float> biasLocal;
    AscendC::LocalTensor<float> tmpLocal;
    AscendC::LocalTensor<float> redLocal;
    AscendC::LocalTensor<float> lbLocal;
    AscendC::LocalTensor<float> scrLocal;
    AscendC::LocalTensor<float> normLocal;

    // The current group's scalars, held in registers between the one Scalar
    // visit that produces them and the Vector work that consumes them: beta as
    // the `Muls` factor in the recurrence, the norms as `Duplicate` fills.
    // Keeping them here rather than in UB is what makes that visit the group's
    // only pipe crossing.
    float beta_[config::C] = {};
    float qNorm_[config::C] = {};
    float kNorm_[config::C] = {};

    // exp(A_log[head]), hoisted per head.
    __aicore__ inline float HeadDecay(uint32_t head);
    // Fill qn/kn/gd/bt for `rows` tokens starting at GM token `tok0` of `head`.
    __aicore__ inline void PrepareGroup(uint32_t head, uint32_t tok0,
                                        uint32_t rows, float decay);
    __aicore__ inline void StoreTile(uint64_t gmOffset);
    __aicore__ inline void RunItem(uint32_t req, uint32_t head, uint32_t iv);
};

template<bool SAVE_INTERMEDIATE, uint32_t H_TOTAL, uint32_t V_TILES>
__aicore__ inline void
KdaVarlenRecurrentDecode<SAVE_INTERMEDIATE, H_TOTAL, V_TILES>::Init(
    GM_ADDR Q, GM_ADDR K, GM_ADDR V, GM_ADDR A,
    GM_ADDR B,
    GM_ADDR A_LOG,
    GM_ADDR DT_BIAS,
    GM_ADDR STATE,
    GM_ADDR INDICES,
    GM_ADDR STARTS,
    GM_ADDR OUT,
    GM_ADDR MID,
    uint32_t midStrideReq,
    uint32_t midStrideStep,
    uint32_t midStrideHead,
    float scale,
    float lowerBound,
    uint32_t numReq)
{
    qGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(Q));
    kGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(K));
    vGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(V));
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A));
    bGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(B));
    aLogGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A_LOG));
    dtBiasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(DT_BIAS));
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(OUT));
    stateGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(STATE));
    indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(INDICES));
    startsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(STARTS));
    if constexpr (SAVE_INTERMEDIATE) {
        midGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(MID));
    }

    scale_ = scale;
    lowerBound_ = lowerBound;
    numReq_ = numReq;
    midStrideReq_ = midStrideReq;
    midStrideStep_ = midStrideStep;
    midStrideHead_ = midStrideHead;

    constexpr uint32_t D = config::D;
    constexpr uint32_t BV = config::BV;
    constexpr uint32_t C = config::C;
    using L = AscendC::LocalTensor<float>;
    using LB = AscendC::LocalTensor<bfloat16_t>;
    constexpr auto P = AscendC::TPosition::VECCALC;

    stateLocal     = L(P, UbLayout::STATE_ADDR, BV * config::STATE_STRIDE);
    mulReduceLocal = L(P, UbLayout::MUL_REDUCE_ADDR, BV * D / 2);
    brcbLocal      = L(P, UbLayout::BRCB_ADDR, BV * 8);
    reduceLocal    = L(P, UbLayout::REDUCE_ADDR, BV);
    tmpBVLocal     = L(P, UbLayout::TMP_BV_ADDR, BV);
    outLocal       = LB(P, UbLayout::OUT_ADDR, BV);

    qnLocal = L(P, UbLayout::QN_ADDR, C * D);
    knLocal = L(P, UbLayout::KN_ADDR, C * D);
    gdLocal = L(P, UbLayout::GD_ADDR, C * D);
    qbLocal = LB(P, UbLayout::QB_ADDR, C * D);
    kbLocal = LB(P, UbLayout::KB_ADDR, C * D);
    abLocal = LB(P, UbLayout::AB_ADDR, C * D);
    vbLocal = LB(P, UbLayout::VB_ADDR, C * D);

    brawLocal = L(P, UbLayout::BRAW_ADDR, 64);
    nrmLocal  = L(P, UbLayout::NRM_ADDR, 64);
    biasLocal = L(P, UbLayout::BIAS_ADDR, D);
    tmpLocal  = L(P, UbLayout::TMP_ADDR, C * D);
    redLocal  = L(P, UbLayout::RED_ADDR, 2 * C * 64);
    lbLocal   = L(P, UbLayout::LB_ADDR, 64);
    scrLocal  = L(P, UbLayout::SCR_ADDR, 64);
    normLocal = L(P, UbLayout::NORM_ADDR, C * D);
}

// `decay = exp(A_log[head])` is a scalar, but this architecture has no scalar
// `exp`: the only `exp` is the Vector instruction. So A_log goes out to UB on
// the Scalar pipe, through `Exp`, and back — two real hardware events, since
// `PipeBarrier<PIPE_V>` does not order Vector against Scalar.
template<bool SAVE_INTERMEDIATE, uint32_t H_TOTAL, uint32_t V_TILES>
__aicore__ inline float
KdaVarlenRecurrentDecode<SAVE_INTERMEDIATE, H_TOTAL, V_TILES>::HeadDecay(uint32_t head)
{
    // `Exp` works a whole 8-float datablock at a time, so all eight lanes are
    // written rather than just the one that is read back — UB arrives
    // uninitialized and exp() of a garbage bit pattern is an avoidable inf.
    const float aLog = aLogGm.GetValue(head);
    for (uint32_t i = 0; i < 8; ++i) {
        scrLocal.SetValue(i, aLog);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(kda_decode::EVENT_S_V);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(kda_decode::EVENT_S_V);
    AscendC::Exp(scrLocal, scrLocal, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(kda_decode::EVENT_V_S);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(kda_decode::EVENT_V_S);
    return scrLocal.GetValue(0);
}

// ── Per-token inputs ──────────────────────────────────────────────────
//
// The quantities the recurrence consumes but does not produce, for a group of
// `rows` tokens of one head. Mirrors `_prepare_inputs` in kda_prefill_npu.py
// and the prefill producer that replaces it:
//
//     gate = lower_bound * sigmoid(decay * (a + dt_bias))
//     GD   = exp(gate)
//     BT   = sigmoid(b)
//     QN   = q / sqrt(sum(q*q) + 1e-6) * scale
//     KN   = k / sqrt(sum(k*k) + 1e-6)
//
// Written as sigmoid(x) = 1 / (1 + exp(-x)), matching upstream's own spelling,
// so the only reassociation anywhere here is in the normalizing division.
// Unlike prefill this runs on the core that will consume it, so nothing round
// trips through GM and there is no notification to pay for.
template<bool SAVE_INTERMEDIATE, uint32_t H_TOTAL, uint32_t V_TILES>
__aicore__ inline void
KdaVarlenRecurrentDecode<SAVE_INTERMEDIATE, H_TOTAL, V_TILES>::PrepareGroup(
    uint32_t head, uint32_t tok0, uint32_t rows, float decay)
{
    constexpr uint32_t D = config::D;
    constexpr uint32_t C = config::C;

    // Rows are strided by H in GM, so one DataCopy per tensor for the whole
    // group with a row gap of (H-1) rows. blockLen/gaps are in 32 B
    // datablocks: a 128-lane bf16 row is 256 B = 8 blocks.
    AscendC::DataCopyParams rowParams{
        static_cast<uint16_t>(rows),
        static_cast<uint16_t>(D * sizeof(bfloat16_t) / 32),
        static_cast<uint16_t>((H_TOTAL - 1) * D * sizeof(bfloat16_t) / 32),
        0};
    const uint64_t base = static_cast<uint64_t>(tok0) * H_TOTAL * D
                          + static_cast<uint64_t>(head) * D;

    AscendC::DataCopy(qbLocal, qGm[base], rowParams);
    AscendC::DataCopy(kbLocal, kGm[base], rowParams);
    AscendC::DataCopy(abLocal, aGm[base], rowParams);
    AscendC::DataCopy(vbLocal, vGm[base], rowParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kda_decode::EVENT_MTE2_V_IN);

    // b is [1,T,H]: one bf16 per row, strided by H. One element per 32 B block
    // is below the DMA's useful granularity, so Scalar gathers the group and
    // hands it to Vector in one go. The tail of a short group gets a finite
    // placeholder so the 64-lane sequence below never sees uninitialized UB.
    for (uint32_t r = 0; r < rows; ++r) {
        const uint64_t bRow = static_cast<uint64_t>(tok0 + r) * H_TOTAL + head;
        brawLocal.SetValue(r, AscendC::ToFloat(bGm.GetValue(bRow)));
    }
    for (uint32_t r = rows; r < 64; ++r) {
        brawLocal.SetValue(r, 0.0f);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(kda_decode::EVENT_S_V);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(kda_decode::EVENT_S_V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kda_decode::EVENT_MTE2_V_IN);

    // ── beta's denominator: 1 + exp(-b) ──
    // The reciprocal that finishes it is Scalar work and is deferred to the
    // single Scalar visit below, with the square roots.
    AscendC::Muls(brawLocal, brawLocal, -1.0f, 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(brawLocal, brawLocal, 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(brawLocal, brawLocal, 1.0f, 64);
    AscendC::PipeBarrier<PIPE_V>();

    // ── GD = exp(lower_bound * sigmoid(decay * (a + dt_bias))) ──
    //
    // Whole group at a time. The dt_bias row is 128 lanes but a Vector repeat
    // covers 64, so each pass runs twice with a repeat stride of one row
    // (16 datablocks) and a src1 stride of 0 to hold bias in place.
    AscendC::Cast(gdLocal, abLocal, AscendC::RoundMode::CAST_NONE, rows * D);
    AscendC::PipeBarrier<PIPE_V>();
    const AscendC::BinaryRepeatParams biasParams{
        1, 1, 1,
        static_cast<uint8_t>(D * sizeof(float) / 32),
        static_cast<uint8_t>(D * sizeof(float) / 32),
        0};
    AscendC::Add(gdLocal, gdLocal, biasLocal, 64,
                 static_cast<uint8_t>(rows), biasParams);
    AscendC::Add(gdLocal[64], gdLocal[64], biasLocal[64], 64,
                 static_cast<uint8_t>(rows), biasParams);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(gdLocal, gdLocal, -decay, rows * D);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(gdLocal, gdLocal, rows * D);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(gdLocal, gdLocal, 1.0f, rows * D);
    AscendC::PipeBarrier<PIPE_V>();
    // gate = lower_bound / (1 + exp(-decay*x)), the numerator held in place by
    // a src0 repeat stride of 0 so one repeat of `lbLocal` covers every row.
    const AscendC::BinaryRepeatParams lbParams{
        1, 1, 1,
        static_cast<uint8_t>(D * sizeof(float) / 32),
        0,
        static_cast<uint8_t>(D * sizeof(float) / 32)};
    AscendC::Div(gdLocal, lbLocal, gdLocal, 64,
                 static_cast<uint8_t>(rows), lbParams);
    AscendC::Div(gdLocal[64], lbLocal, gdLocal[64], 64,
                 static_cast<uint8_t>(rows), lbParams);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(gdLocal, gdLocal, rows * D);

    // ── q, k: the squared norms for the whole group ──
    //
    // All the Vector work first, so the group needs one Scalar visit rather
    // than one per token. The sums land in `nrmLocal` two per token, q then k.
    AscendC::Cast(qnLocal, qbLocal, AscendC::RoundMode::CAST_NONE, rows * D);
    AscendC::Cast(knLocal, kbLocal, AscendC::RoundMode::CAST_NONE, rows * D);
    AscendC::PipeBarrier<PIPE_V>();

    // sum(x*x) over 128 lanes, in upstream's summation order.
    //
    // `tl.sum` over 128 fp32 lanes is bit-identical to `torch.sum(-1)` and to
    // neither a 2x64 nor a 4x32 decomposition. Of thirteen candidate trees
    // measured against it, exactly one matched on every row: add the two
    // 64-lane halves elementwise, then reduce the 64 partials as a binary tree
    // over adjacent lanes — which is what `WholeReduceSum` over one 64-lane
    // repeat does. A 2x64 `WholeReduceSum` pair differs on ~28% of rows. The
    // norm divides q and k on every step, so a last-bit difference here
    // compounds; the order has to match.
    //
    // Each stage runs once for the whole group rather than once per token: the
    // fold is a strided `Add` whose repeat stride walks rows (16 datablocks)
    // while its operands are the two halves of each row, and the reduction is
    // one `WholeReduceSum` over all 2*rows partial sets. q's sums land at
    // lanes [0, rows) of `nrmLocal` and k's at [8, 8+rows).
    const uint8_t rowStride = static_cast<uint8_t>(D * sizeof(float) / 32);
    const AscendC::BinaryRepeatParams foldParams{1, 1, 1, 8, rowStride,
                                                 rowStride};
    AscendC::Mul(tmpLocal, qnLocal, qnLocal, rows * D);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(redLocal, tmpLocal, tmpLocal[D / 2], D / 2,
                 static_cast<uint8_t>(rows), foldParams);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(tmpLocal, knLocal, knLocal, rows * D);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(redLocal[C * 64], tmpLocal, tmpLocal[D / 2], D / 2,
                 static_cast<uint8_t>(rows), foldParams);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum<float>(nrmLocal, redLocal, 64,
                                   static_cast<uint8_t>(rows), 1, 1, 8);
    AscendC::WholeReduceSum<float>(nrmLocal[8], redLocal[C * 64], 64,
                                   static_cast<uint8_t>(rows), 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();

    // ── the group's Scalar visit ──
    //
    // Everything that has to run on the Scalar unit, in one crossing: beta's
    // reciprocal and both square roots per token. Both are Scalar rather than
    // Vector for the same reason — `Div` and `Sqrt` are documented at 1 ulp
    // here (the 0-ulp algorithms are 950-series only) while upstream's are
    // IEEE-exact, and beta and the norms feed the state on every step, so a
    // last-bit difference compounds along the sequence. The Vector `Sqrt`
    // roughly doubles the surviving output mismatches on its own.
    //
    // The norms go back to UB because they are needed as 128-lane tensors;
    // beta stays in registers because the recurrence consumes it as a `Muls`
    // factor.
    AscendC::SetFlag<AscendC::HardEvent::V_S>(kda_decode::EVENT_V_S);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(kda_decode::EVENT_V_S);
    for (uint32_t r = 0; r < rows; ++r) {
        beta_[r] = 1.0f / brawLocal.GetValue(r);
        // sum(x*x) + 1e-6, then sqrt, matching upstream's epsilon placement.
        qNorm_[r] = sqrt(nrmLocal.GetValue(r) + 1e-6f);
        kNorm_[r] = sqrt(nrmLocal.GetValue(8 + r) + 1e-6f);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(kda_decode::EVENT_S_V);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(kda_decode::EVENT_S_V);

    // ── normalize, and fold `scale` into q ──
    //
    // Upstream is `q = q / norm` then `q *= scale`, and `k = k / norm`. Keep
    // that order and keep both as real divisions: folding them into
    // `scale / qNorm`, or into a reciprocal multiply, reassociates and costs a
    // bf16 ulp on a few output elements.
    //
    // `Div` takes both operands as tensors and `Divs` is unsupported, so the
    // norms have to be broadcast first. `Brcb` spreads one datablock of 8
    // across 8 datablocks, which is exactly the group's q norms (and, 8 lanes
    // on, its k norms) turned into one 64-lane row each; a strided `Copy`
    // then fills each token's row from its block, and the division runs once
    // over the whole group.
    // norms have to be broadcast first — but from the registers the Scalar
    // visit left them in, not back out of UB, which would be another crossing.
    // One `Duplicate` per token per tensor, then one division for the group.
    for (uint32_t r = 0; r < rows; ++r) {
        AscendC::Duplicate(normLocal[r * D], qNorm_[r], D);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(qnLocal, qnLocal, normLocal, rows * D);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(qnLocal, qnLocal, scale_, rows * D);
    for (uint32_t r = 0; r < rows; ++r) {
        AscendC::Duplicate(normLocal[r * D], kNorm_[r], D);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(knLocal, knLocal, normLocal, rows * D);
    AscendC::PipeBarrier<PIPE_V>();
}

// The state tile in the same orientation on both sides — V-major with the D
// values of K contiguous — so this is a straight contiguous copy of BV whole
// rows. One core owns the entire 64 KB span, so concurrent stores from other
// cores cannot share a datablock with it.
template<bool SAVE_INTERMEDIATE, uint32_t H_TOTAL, uint32_t V_TILES>
__aicore__ inline void
KdaVarlenRecurrentDecode<SAVE_INTERMEDIATE, H_TOTAL, V_TILES>::StoreTile(
    uint64_t gmOffset)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(kda_decode::EVENT_V_MTE3_OUT);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(kda_decode::EVENT_V_MTE3_OUT);
    AscendC::DataCopy(stateGm[gmOffset], stateLocal, config::BV * config::D);
    // MTE3 reads `stateLocal` directly, and this is the last thing the block
    // does, so the copy has to be seen out before the kernel exits.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(kda_decode::EVENT_MTE3_V_OUT);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(kda_decode::EVENT_MTE3_V_OUT);
}

// ── One (request, head) ───────────────────────────────────────────────
template<bool SAVE_INTERMEDIATE, uint32_t H_TOTAL, uint32_t V_TILES>
__aicore__ inline void
KdaVarlenRecurrentDecode<SAVE_INTERMEDIATE, H_TOTAL, V_TILES>::RunItem(
    uint32_t req, uint32_t head, uint32_t iv)
{
    constexpr uint32_t D = config::D;
    constexpr uint32_t BV = config::BV;
    constexpr uint32_t C = config::C;
    constexpr uint32_t STATE_STRIDE = config::STATE_STRIDE;

    const int32_t bos = startsGm.GetValue(req);
    const int32_t eos = startsGm.GetValue(req + 1);
    const int32_t slot = indicesGm.GetValue(req);

    // pstate points at this core's slice of state[slot, head]. The slot is
    // [V, K] with K contiguous, so value rows are the outer axis and tile `iv`
    // owns a contiguous run of BV of them — at V_TILES == 1 that is the whole
    // D*D span.
    const uint64_t pstate =
        (static_cast<uint64_t>(slot) * H_TOTAL + head) * D * D
        + static_cast<uint64_t>(iv) * BV * D;

    if (slot > 0) {
        AscendC::DataCopy(stateLocal, stateGm[pstate], BV * D);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
            kda_decode::EVENT_MTE2_V_STATE);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            kda_decode::EVENT_MTE2_V_STATE);
    } else {
        AscendC::Duplicate(stateLocal, 0.0f, BV * STATE_STRIDE);
    }
    AscendC::PipeBarrier<PIPE_V>();

    // Per-head constants. `Div` takes both operands as tensors here and `Divs`
    // is unsupported, so the gate's scalar numerator has to be materialized;
    // `Reciprocal` would avoid it but is documented as missing the
    // double-ten-thousandth bar for float, and the gate multiplies the state
    // once per step, so that error would compound.
    AscendC::DataCopy(biasLocal, dtBiasGm[head * D], D);
    AscendC::Duplicate(lbLocal, lowerBound_, 64);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kda_decode::EVENT_MTE2_V_IN);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kda_decode::EVENT_MTE2_V_IN);
    const float decay = HeadDecay(head);

    // Repeat descriptors for the state-wide instructions. These encode the UB
    // map: a state row is STATE_STRIDE floats, so consecutive rows are
    // STATE_STRIDE/8 datablocks apart.
    const AscendC::BinaryRepeatParams decay_params{
        1, 1, 1, STATE_STRIDE / 8, STATE_STRIDE / 8, 0};
    const AscendC::BinaryRepeatParams mul_reduce_params{
        1, 1, 1, 8, STATE_STRIDE / 8, 0};
    const AscendC::BinaryRepeatParams outer_params{
        1, 1, 0, STATE_STRIDE / 8, 0, 1};

    // The checkpoint row for this (request, head). Indexed by the request's
    // position in the batch, as the generic Triton kernel indexes it (`i_n`),
    // not by the state slot.
    uint64_t pmid = 0;
    if constexpr (SAVE_INTERMEDIATE) {
        pmid = static_cast<uint64_t>(req) * midStrideReq_
               + static_cast<uint64_t>(head) * midStrideHead_
               + static_cast<uint64_t>(iv) * BV * D;
    }

    for (int32_t tok0 = bos; tok0 < eos; tok0 += C) {
        const uint32_t rows = (tok0 + static_cast<int32_t>(C) <= eos)
                                  ? C
                                  : static_cast<uint32_t>(eos - tok0);
        PrepareGroup(head, static_cast<uint32_t>(tok0), rows, decay);

        for (uint32_t r = 0; r < rows; ++r) {
            const int32_t token = tok0 + static_cast<int32_t>(r);
            const uint64_t row = static_cast<uint64_t>(token) * H_TOTAL + head;
            auto qRow = qnLocal[r * D];
            auto kRow = knLocal[r * D];
            auto gRow = gdLocal[r * D];

            // state *= exp(gate)[None, :]
            AscendC::Mul(stateLocal, stateLocal, gRow, 64, BV, decay_params);
            AscendC::Mul(stateLocal[64], stateLocal[64], gRow[64],
                         64, BV, decay_params);
            AscendC::PipeBarrier<PIPE_V>();

            // v -= tl.sum(state * k[None, :], axis=1)
            AscendC::Mul(mulReduceLocal, stateLocal, kRow,
                         64, BV, mul_reduce_params);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst(mulReduceLocal, stateLocal[64], kRow[64],
                               64, BV, mul_reduce_params);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(reduceLocal, mulReduceLocal,
                                           64, BV, 1, 1, 8);

            // v is indexed by the value axis, so only this tile's slice of the
            // row is wanted. q, k and the gate are indexed by the key axis and
            // every tile needs all of them.
            AscendC::Cast(tmpBVLocal, vbLocal[r * D + iv * BV],
                          AscendC::RoundMode::CAST_NONE, BV);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(tmpBVLocal, tmpBVLocal, reduceLocal, BV);
            AscendC::PipeBarrier<PIPE_V>();

            // v *= beta. Already in a register from the group's Scalar visit,
            // so this costs no pipe crossing.
            AscendC::Muls(tmpBVLocal, tmpBVLocal, beta_[r], BV);
            AscendC::PipeBarrier<PIPE_V>();

            // state += k[None, :] * v[:, None]
            AscendC::Brcb(brcbLocal, tmpBVLocal, BV / 8, {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst(stateLocal, kRow, brcbLocal,
                               64, BV, outer_params);
            AscendC::MulAddDst(stateLocal[64], kRow[64], brcbLocal,
                               64, BV, outer_params);
            AscendC::PipeBarrier<PIPE_V>();

            // The per-step checkpoint, taken here — after the state update,
            // before the output — where the Triton kernel takes it. Skipped
            // when the slot is the padding row, matching its
            // `mask_h & (state_index > 0)`.
            //
            // Issued and left to drain: 64 KB is by far the largest transfer
            // in the loop, and the output reduction below reads `stateLocal`
            // without writing it, so MTE3 can walk the tile while Vector works
            // on it. The wait is deferred to the top of the next step, which is
            // the first point that writes the tile again.
            bool midInFlight = false;
            if constexpr (SAVE_INTERMEDIATE) {
                if (slot > 0) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                        kda_decode::EVENT_V_MTE3_MID);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                        kda_decode::EVENT_V_MTE3_MID);
                    AscendC::DataCopy(
                        midGm[pmid + static_cast<uint64_t>(token - bos) *
                                         midStrideStep_],
                        stateLocal, BV * D);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                        kda_decode::EVENT_MTE3_V_MID);
                    midInFlight = true;
                }
            }

            // out = tl.sum(state * q[None, :], axis=1)
            AscendC::Mul(mulReduceLocal, stateLocal, qRow,
                         64, BV, mul_reduce_params);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst(mulReduceLocal, stateLocal[64], qRow[64],
                               64, BV, mul_reduce_params);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(reduceLocal, mulReduceLocal,
                                           64, BV, 1, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Cast(outLocal, reduceLocal,
                          AscendC::RoundMode::CAST_RINT, BV);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                kda_decode::EVENT_V_MTE3_OUT);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                kda_decode::EVENT_V_MTE3_OUT);
            AscendC::DataCopy(outGm[row * D + iv * BV], outLocal, BV);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                kda_decode::EVENT_MTE3_V_OUT);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                kda_decode::EVENT_MTE3_V_OUT);

            // Collect the checkpoint before the next step's decay touches the
            // tile. Also before leaving the loop, so the final-state store and
            // the kernel exit cannot race it.
            if (midInFlight) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                    kda_decode::EVENT_MTE3_V_MID);
            }
        }
    }

    // The final state goes back only when nothing checkpointed it, matching
    // the generic kernel's `if not SAVE_INTERMEDIATE and state_index > 0`:
    // under verify the commit reads the accepted step out of the checkpoint
    // buffer, and the source slot must stay at its pre-verify value.
    if constexpr (!SAVE_INTERMEDIATE) {
        if (slot > 0) {
            StoreTile(pstate);
        }
    }
}

template<bool SAVE_INTERMEDIATE, uint32_t H_TOTAL, uint32_t V_TILES>
__aicore__ inline void
KdaVarlenRecurrentDecode<SAVE_INTERMEDIATE, H_TOTAL, V_TILES>::Process()
{
    AscendC::InitSocState();

    // One block per (request, head, value tile), with the value tile minor and
    // the head next, so consecutive blocks walk a contiguous run of GM in both
    // the state pool and the output. The grid is sized to exactly
    // `numReq * H_TOTAL * V_TILES` by the launcher, but a padded or replayed
    // graph can hand over a smaller `numReq` than the captured grid, so the
    // bound is checked rather than assumed.
    const uint32_t blockIdx = AscendC::GetBlockIdx();
    const uint32_t iv = blockIdx % V_TILES;
    const uint32_t head = (blockIdx / V_TILES) % H_TOTAL;
    const uint32_t req = blockIdx / (V_TILES * H_TOTAL);
    if (req >= numReq_) {
        return;
    }
    RunItem(req, head, iv);
}

// One instantiation of one decode width. `SAVE_INTERMEDIATE` and `V_TILES` are
// template parameters rather than runtime values because the first changes the
// store schedule inside the innermost loop and removes the final state write,
// and the second sets every repeat count and UB offset in the kernel.
#define KDA_DEFINE_DECODE_KERNEL(NAME, SAVE, HTOT, VT)                  \
extern "C" __global__ __vector__                                        \
void NAME(                                                              \
    GM_ADDR Q, GM_ADDR K, GM_ADDR V, GM_ADDR A, GM_ADDR B,              \
    GM_ADDR A_LOG, GM_ADDR DT_BIAS,                                     \
    GM_ADDR STATE, GM_ADDR INDICES, GM_ADDR STARTS, GM_ADDR OUT,        \
    GM_ADDR MID,                                                        \
    uint32_t MID_STRIDE_REQ, uint32_t MID_STRIDE_STEP,                  \
    uint32_t MID_STRIDE_HEAD,                                           \
    float scale, float lowerBound, uint32_t numReq                      \
) {                                                                     \
    KdaVarlenRecurrentDecode<SAVE, HTOT, VT> op;                        \
    op.Init(Q, K, V, A, B, A_LOG, DT_BIAS,                              \
            STATE, INDICES, STARTS, OUT, MID,                           \
            MID_STRIDE_REQ, MID_STRIDE_STEP, MID_STRIDE_HEAD,           \
            scale, lowerBound, numReq);                                 \
    op.Process();                                                       \
}

// Both instantiations of one decode width.
//
// V_TILES is 1 everywhere. It is a template parameter rather than a constant
// because the question of whether to split a head across cores is a real one at
// low concurrency — the profiled 512K shapes give the device 16 and 32 blocks
// against 40 vector cores, so most of it is idle — and the parameter is what
// makes the answer measurable. The answer turned out to be no:
//
//   512K DP4TP2 R1 (32 blocks)   V_TILES=1 23.8us   V_TILES=2 30.2us
//   512K DP2TP4 R1 (16 blocks)   V_TILES=1 19.7us   V_TILES=4 19.6us
//
// Splitting costs more than the idle cores save, because every tile of a head
// repeats that head's whole per-token prologue: the same gate, the same two L2
// norms, the same beta, each of which crosses the Vector/Scalar boundary. Those
// shapes are not short of cores so much as serial in the token index, and a
// value split does nothing about that.
#define KDA_DEFINE_DECODE_WIDTH(HTOT)                                   \
    KDA_DEFINE_DECODE_KERNEL(kda_decode_k##HTOT,      false, HTOT, 1)   \
    KDA_DEFINE_DECODE_KERNEL(kda_decode_k##HTOT##_mid, true, HTOT, 1)

// The launcher body, identical across widths: one block per (request, head).
// Decode work scales with `numReq`, so the block count does too — a fixed grid
// would either leave cores idle at low concurrency or need an inner loop.
//
// `args.mid` is null outside the verify regime and the stride fields are then
// meaningless, but the plain kernel ignores all four, so they are passed
// either way to keep one call expression.
#define KDA_LAUNCH_DECODE(HTOT)                                         \
    do {                                                                \
        const uint32_t blockDim = args.num_req * (HTOT);                \
        auto acl_stream = static_cast<aclrtStream>(stream);             \
        if (args.mid != nullptr) {                                      \
            KDA_DECODE_CALL(kda_decode_k##HTOT##_mid);                  \
        } else {                                                        \
            KDA_DECODE_CALL(kda_decode_k##HTOT);                        \
        }                                                               \
    } while (0)

#define KDA_DECODE_CALL(KERNEL)                                         \
    KERNEL<<<blockDim, nullptr, acl_stream>>>(                          \
        args.q, args.k, args.v, args.a, args.b,                         \
        args.A_log, args.dt_bias,                                       \
        args.state, args.indices, args.starts, args.out,                \
        args.mid, args.mid_stride_req, args.mid_stride_step,            \
        args.mid_stride_head,                                           \
        args.scale, args.lower_bound, args.num_req)
