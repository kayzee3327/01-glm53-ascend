// GLM-5.3 KDA varlen recurrence — decode kernel, H=32 (TP2).
//
// Specializes `_glm_kda_varlen_recurrent_kernel` from
// sglang/srt/hardware_backend/npu/attention/glm53/kda_recurrent_npu.py to
// H == HV == 32, K == V == 128.
//
// `Process` is banner-commented against that kernel: a `══ NAME ══` banner
// opens a stage and lists the Triton lines it covers, and a `── expr ──`
// banner marks the instructions computing that one expression. Code between
// two banners implements the upper one. Triton's token body is split across
// two stages — PROLOGUE for the state-independent lines, which this kernel
// batches over a group of tokens, and RECURRENCE for the dependency chain,
// which it runs one token at a time in upstream's order.

#include "kernel_operator.h"
#include "kda_varlen_recurrent.h"

#ifndef __CCE_KT_TEST__
#include <acl/acl.h>
#endif

namespace kda_decode_h32 {

// Hardware event IDs, managed by hand as in the prefill kernels. Avoid 6/7.
constexpr int32_t EVENT_MTE2_V_STATE = 0;
constexpr int32_t EVENT_MTE2_V_IN = 1;
constexpr int32_t EVENT_V_MTE2_IN = 2;
constexpr int32_t EVENT_V_MTE3_OUT = 0;
constexpr int32_t EVENT_V_MTE3_MID = 1;
constexpr int32_t EVENT_MTE3_V_OUT = 0;
constexpr int32_t EVENT_MTE3_V_MID = 1;
constexpr int32_t EVENT_S_V = 0;
constexpr int32_t EVENT_V_S = 0;

// Tokens whose gate, beta and L2 norms are prepared per pass of the token
// loop. Batching them is what makes the Scalar reciprocal and square roots one
// pipe crossing per group instead of four per token. Four is the deployed
// `draft_token_num`, so verify is one group and plain decode a one-row group.
constexpr uint32_t kStepGroup = 4;

// Heads per state slot: the row stride of every q/k/v/a/b read and the head
// stride of the state pool. TP2 on a 64-head model, so this is the shape, not
// a parameter — the host rejects anything else before the H=32 launcher.
constexpr uint32_t kHeads = 32;

}  // namespace kda_decode_h32

// SAVE_INTERMEDIATE  Triton's `SAVE_INTERMEDIATE`: checkpoint the state after
//                    every step and skip the final state write back.
template<bool SAVE_INTERMEDIATE>
class KdaVarlenRecurrentDecodeH32 {
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
        // The state tile's value rows. One core owns a head's whole value
        // axis, so this is D; it stays a separate name because it counts rows
        // where D counts a row's lanes.
        static constexpr uint32_t BV = D;
        static constexpr uint32_t STATE_STRIDE = D;
        static constexpr uint32_t C = kda_decode_h32::kStepGroup;
        static constexpr uint32_t H_TOTAL = kda_decode_h32::kHeads;

        // Vector repeat counts and strides are uint8 on this architecture, so
        // BV=128 sits inside the limit but leaves no room for a wider tile.
        static_assert(BV <= 255, "BV is a uint8 repeat count");
        static_assert(STATE_STRIDE / 8 <= 255,
                      "the state row stride is a uint8 repeat stride");
        static_assert(BV % 8 == 0, "Brcb consumes 8 elements per repeat");
    };

    // UB map, laid out by hand at absolute addresses as in the prefill
    // kernels. The repeat strides in the token loop are derived from these
    // offsets, so moving one means revisiting the instruction that reads it.
    // ~116 KB of 192 KB, dominated by the 64 KB state tile.
    struct UbLayout {
        static constexpr uint32_t D = config::D;
        static constexpr uint32_t BV = config::BV;
        static constexpr uint32_t C = config::C;
        static constexpr uint32_t ROW_F32 = D * sizeof(float);     // 512
        static constexpr uint32_t ROW_BF16 = D * sizeof(bfloat16_t);

        // ── the resident state tile, `b_h` ──────────────────────────────
        static constexpr uint32_t STATE_ADDR = 0x00000;
        static constexpr uint32_t STATE_SIZE =
            BV * config::STATE_STRIDE * sizeof(float);             // 64 KB
        // Partial products of the two `tl.sum(b_h * x[None, :], axis=1)`
        // reductions, half a row wide because the halves fold in place.
        static constexpr uint32_t MUL_REDUCE_ADDR = STATE_ADDR + STATE_SIZE;
        static constexpr uint32_t MUL_REDUCE_SIZE =
            BV * (D / 2) * sizeof(float);                          // 32 KB
        // v broadcast across datablocks, the `b_v[:, None]` outer-product term.
        static constexpr uint32_t BRCB_ADDR = MUL_REDUCE_ADDR + MUL_REDUCE_SIZE;
        static constexpr uint32_t BRCB_SIZE = BV * 8 * sizeof(float);
        static constexpr uint32_t REDUCE_ADDR = BRCB_ADDR + BRCB_SIZE;
        static constexpr uint32_t TMP_BV_ADDR =
            REDUCE_ADDR + BV * sizeof(float);
        static constexpr uint32_t OUT_ADDR = TMP_BV_ADDR + BV * sizeof(float);

        // ── per-token inputs, one group of C tokens at a time ───────────
        // `b_q`/`b_k` normalized, `tl.exp(b_g)`, and the staged bf16 rows.
        static constexpr uint32_t QN_ADDR = OUT_ADDR + Align32(BV * sizeof(bfloat16_t));
        static constexpr uint32_t KN_ADDR = QN_ADDR + C * ROW_F32;
        static constexpr uint32_t GD_ADDR = KN_ADDR + C * ROW_F32;
        static constexpr uint32_t QB_ADDR = GD_ADDR + C * ROW_F32;
        static constexpr uint32_t KB_ADDR = QB_ADDR + C * ROW_BF16;
        static constexpr uint32_t AB_ADDR = KB_ADDR + C * ROW_BF16;
        static constexpr uint32_t VB_ADDR = AB_ADDR + C * ROW_BF16;

        // ── scalars and scratch ─────────────────────────────────────────
        // beta's `1 + tl.exp(-b_beta)` denominators. A whole 64-lane repeat,
        // because Exp works a datablock at a time and the tail lanes must not
        // be garbage.
        static constexpr uint32_t BRAW_ADDR = VB_ADDR + C * ROW_BF16;
        // `b_dt_bias` for this head, loaded once
        static constexpr uint32_t BIAS_ADDR = BRAW_ADDR + 64 * sizeof(float);
        // `b_q * b_q` / `b_k * b_k` and their folded half-sums, group-sized so
        // each reduction stage is one strided instruction, not a token loop.
        static constexpr uint32_t TMP_ADDR = BIAS_ADDR + ROW_F32;
        static constexpr uint32_t RED_ADDR = TMP_ADDR + C * ROW_F32;
        // The `tl.sum(b_q * b_q)` / `tl.sum(b_k * b_k)` results: q's at lane
        // r, k's at lane 8+r. The gap keeps the two sets in separate
        // datablocks so one `WholeReduceSum` can land each contiguously.
        static constexpr uint32_t NRM_ADDR = RED_ADDR + 2 * C * 64 * sizeof(float);
        // `lower_bound`, the gate division's numerator. `Div` takes both
        // operands as tensors and `Divs` is unsupported, so the scalar has to
        // be materialized; a src0 repeat stride of 0 reuses this one repeat.
        static constexpr uint32_t LB_ADDR = NRM_ADDR + 64 * sizeof(float);
        // Scalar/Vector exchange slot for `decay = tl.exp(b_A_log)`: this
        // architecture has no scalar `exp`.
        static constexpr uint32_t SCR_ADDR = LB_ADDR + 64 * sizeof(float);
        // The broadcast norms, one row per token, for the same `Div` reason —
        // a row each so the group's division is one instruction over C*D lanes.
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
};

template<bool SAVE_INTERMEDIATE>
__aicore__ inline void
KdaVarlenRecurrentDecodeH32<SAVE_INTERMEDIATE>::Init(
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

// ── One (request, head) ───────────────────────────────────────────────
//
// Triton's grid is (value tile, request, head group); this one is (request,
// head), so both of its outer axes collapse. Its `i_v` ranges over two BV=64
// tiles because `block_v` is capped at 64 on the host, while one core here
// takes the whole value axis at BV=128 — legal because nothing in the step
// reduces along that axis, and measured faster (see the instantiations below).
// `i_h` and `i_hv` collapse into one `head` because the host requires H == HV,
// its `BHV` head loop is 1, and the mask_k/mask_v/mask_h guards vanish because
// BK == BV == K == V == 128.
template<bool SAVE_INTERMEDIATE>
__aicore__ inline void
KdaVarlenRecurrentDecodeH32<SAVE_INTERMEDIATE>::Process()
{
    AscendC::InitSocState();

    constexpr uint32_t D = config::D;
    constexpr uint32_t BV = config::BV;
    constexpr uint32_t C = config::C;
    constexpr uint32_t STATE_STRIDE = config::STATE_STRIDE;
    constexpr uint32_t H_TOTAL = config::H_TOTAL;

    // ── i_n, i_nhv = program_id(1), program_id(2) ────────────────────────
    //
    // Flattened into one block index, head minor. A padded or replayed graph
    // can pass a smaller `numReq` than the captured grid, so the bound is
    // checked rather than assumed.
    const uint32_t blockIdx = AscendC::GetBlockIdx();
    const uint32_t head = blockIdx % H_TOTAL;
    const uint32_t req = blockIdx / H_TOTAL;
    if (req >= numReq_) {
        return;
    }

    // ── bos, eos = cu_seqlens[i_n], cu_seqlens[i_n + 1] ──────────────────
    const int32_t bos = startsGm.GetValue(req);
    const int32_t eos = startsGm.GetValue(req + 1);
    const int32_t slot = indicesGm.GetValue(req);

    // ── p_h0 = h0_source + state_index*S0 + i_hv*S1 + o_k*SK + o_v*SV ────
    //
    // The slot is [V, K] with K contiguous and this core owns every value row,
    // so `o_k`/`o_v` cover the whole contiguous D*D span.
    const uint64_t pstate =
        (static_cast<uint64_t>(slot) * H_TOTAL + head) * D * D;

    // ── b_h = load(p_h0) if state_index > 0 else zeros([BV, BK]) ─────────
    if (slot > 0) {
        AscendC::DataCopy(stateLocal, stateGm[pstate], BV * D);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
            kda_decode_h32::EVENT_MTE2_V_STATE);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            kda_decode_h32::EVENT_MTE2_V_STATE);
    } else {
        AscendC::Duplicate(stateLocal, 0.0f, BV * STATE_STRIDE);
    }
    AscendC::PipeBarrier<PIPE_V>();

    // ── b_dt_bias = load(p_dt_bias) ──────────────────────────────────────
    //
    // Plus lower_bound as a tensor, which the gate's `Div` needs.
    AscendC::DataCopy(biasLocal, dtBiasGm[head * D], D);
    AscendC::Duplicate(lbLocal, lowerBound_, 64);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(kda_decode_h32::EVENT_MTE2_V_IN);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(kda_decode_h32::EVENT_MTE2_V_IN);

    // ── decay = tl.exp(b_A_log) ──────────────────────────────────────────
    //
    // Hoisted out of the token loop. No scalar `exp` here, so it goes out to
    // UB, through the Vector `Exp`, and back; both crossings are real events
    // because `PipeBarrier<PIPE_V>` does not order Vector against Scalar. All
    // eight lanes are written because `Exp` works a whole datablock and exp()
    // of uninitialized UB is an avoidable inf.
    const float aLog = aLogGm.GetValue(head);
    for (uint32_t i = 0; i < 8; ++i) {
        scrLocal.SetValue(i, aLog);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(kda_decode_h32::EVENT_S_V);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(kda_decode_h32::EVENT_S_V);
    AscendC::Exp(scrLocal, scrLocal, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(kda_decode_h32::EVENT_V_S);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(kda_decode_h32::EVENT_V_S);
    const float decay = scrLocal.GetValue(0);

    // Repeat descriptors for the three state-wide broadcasts below: the gate
    // multiply, the two reductions, and the outer product. A state row is
    // STATE_STRIDE floats, so consecutive rows are STATE_STRIDE/8 datablocks
    // apart.
    const AscendC::BinaryRepeatParams decay_params{
        1, 1, 1, STATE_STRIDE / 8, STATE_STRIDE / 8, 0};
    const AscendC::BinaryRepeatParams mul_reduce_params{
        1, 1, 1, 8, STATE_STRIDE / 8, 0};
    const AscendC::BinaryRepeatParams outer_params{
        1, 1, 0, STATE_STRIDE / 8, 0, 1};

    // ── p_mid = intermediate + i_n*M0 + i_hv*M2 + o_k*MK + o_v*MV ────────
    //
    // Indexed by the request's position in the batch (`i_n`), not by the state
    // slot. The `i*M1` step term is added per token inside the loop.
    uint64_t pmid = 0;
    if constexpr (SAVE_INTERMEDIATE) {
        pmid = static_cast<uint64_t>(req) * midStrideReq_
               + static_cast<uint64_t>(head) * midStrideHead_;
    }

    // The group's scalars, in registers between the Scalar visit that produces
    // them and the Vector work that consumes them. Keeping them out of UB is
    // what makes that visit the group's only pipe crossing.
    float beta[C] = {};
    float qNorm[C] = {};
    float kNorm[C] = {};

    // ── for i in range(seq_len) ──────────────────────────────────────────
    //
    // Strip-mined by C: the group's tokens share one PROLOGUE pass, then
    // RECURRENCE walks them individually.
    for (int32_t tok0 = bos; tok0 < eos; tok0 += C) {
        const uint32_t rows = (tok0 + static_cast<int32_t>(C) <= eos)
                                  ? C
                                  : static_cast<uint32_t>(eos - tok0);

        // ══ PROLOGUE ════════════════════════════════════════════════════
        //
        // The state-independent half of Triton's token body, lifted to run
        // once per group of `rows` tokens:
        //
        //     b_beta = 1 / (1 + tl.exp(-b_beta))
        //     b_g    = lower_bound / (1 + tl.exp(-(decay * (b_a + b_dt_bias))))
        //     b_q    = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6) * scale
        //     b_k    = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        //
        // Each keeps upstream's spelling and association; the Scalar visit and
        // the real divisions below are bit-exactness requirements.

        // ── b_q, b_k, b_a, b_v = load(p_q), load(p_k), ... ──────────────
        //
        // Rows are strided by H in GM, so one DataCopy each with a row gap of
        // (H-1) rows. blockLen/gaps are in 32 B datablocks: a 128-lane bf16
        // row is 256 B = 8 blocks.
        AscendC::DataCopyParams rowParams{
            static_cast<uint16_t>(rows),
            static_cast<uint16_t>(D * sizeof(bfloat16_t) / 32),
            static_cast<uint16_t>((H_TOTAL - 1) * D * sizeof(bfloat16_t) / 32),
            0};
        const uint64_t base = static_cast<uint64_t>(tok0) * H_TOTAL * D
                              + static_cast<uint64_t>(head) * D;

        // The staged rows are reused every group, and the previous group's
        // recurrence reads `vbLocal` to its last token, so MTE2 must not
        // refill them until those reads retire. Only the second group onward
        // can race; the first has nothing in flight.
        if (tok0 > bos) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                kda_decode_h32::EVENT_V_MTE2_IN);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                kda_decode_h32::EVENT_V_MTE2_IN);
        }

        AscendC::DataCopy(qbLocal, qGm[base], rowParams);
        AscendC::DataCopy(kbLocal, kGm[base], rowParams);
        AscendC::DataCopy(abLocal, aGm[base], rowParams);
        AscendC::DataCopy(vbLocal, vGm[base], rowParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
            kda_decode_h32::EVENT_MTE2_V_IN);

        // ── b_beta = load(p_b) ──────────────────────────────────────────
        //
        // One bf16 per row, strided by H, so a DataCopy would move one useful
        // element per 32 B datablock; Scalar gathers the group instead. The
        // tail of a short group gets a placeholder so the 64-lane sequence
        // below never sees uninitialized UB.
        for (uint32_t r = 0; r < rows; ++r) {
            const uint64_t bRow =
                static_cast<uint64_t>(tok0 + r) * H_TOTAL + head;
            brawLocal.SetValue(r, AscendC::ToFloat(bGm.GetValue(bRow)));
        }
        for (uint32_t r = rows; r < 64; ++r) {
            brawLocal.SetValue(r, 0.0f);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(kda_decode_h32::EVENT_S_V);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(kda_decode_h32::EVENT_S_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            kda_decode_h32::EVENT_MTE2_V_IN);

        // ── b_beta: 1 + tl.exp(-b_beta) ─────────────────────────────────
        //
        // Denominator only; the `1 /` is Scalar work, deferred to the single
        // visit below.
        AscendC::Muls(brawLocal, brawLocal, -1.0f, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(brawLocal, brawLocal, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(brawLocal, brawLocal, 1.0f, 64);
        AscendC::PipeBarrier<PIPE_V>();

        // ── b_g = lower_bound / (1 + tl.exp(-(decay * x))) ──────────────
        //    where x = b_a + b_dt_bias, then tl.exp(b_g)
        //
        // A Vector repeat covers 64 of the row's 128 lanes, so each pass runs
        // twice with a repeat stride of one row (16 datablocks) and a src1
        // stride of 0 to hold bias in place.
        AscendC::Cast(gdLocal, abLocal, AscendC::RoundMode::CAST_NONE,
                      rows * D);
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
        // The `lower_bound / ...` division: a src0 repeat stride of 0 reuses
        // the one repeat of `lbLocal` for every row.
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

        // ── tl.sum(b_q * b_q), tl.sum(b_k * b_k) ────────────────────────
        //
        // Landing in `nrmLocal` two per token. All the Vector work first, so
        // the group needs one Scalar visit rather than one per token.
        AscendC::Cast(qnLocal, qbLocal, AscendC::RoundMode::CAST_NONE,
                      rows * D);
        AscendC::Cast(knLocal, kbLocal, AscendC::RoundMode::CAST_NONE,
                      rows * D);
        AscendC::PipeBarrier<PIPE_V>();

        // The reduction tree is fixed, not a choice: `tl.sum` over 128 fp32
        // lanes matches only fold-the-two-64-lane-halves then one 64-lane
        // `WholeReduceSum`. A 2x64 `WholeReduceSum` pair differs on ~28% of
        // rows, and the norm divides q and k every step, so it compounds.
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

        // ── 1 / (...) and both tl.sqrt(... + 1e-6) ──────────────────────
        //
        // The group's one Scalar visit. On Scalar because the Vector
        // `Div`/`Sqrt` are 1 ulp here (0-ulp is 950-series only) while
        // upstream's are IEEE-exact, and both feed the state every step. The
        // Vector `Sqrt` alone roughly doubles the surviving mismatches.
        AscendC::SetFlag<AscendC::HardEvent::V_S>(kda_decode_h32::EVENT_V_S);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(kda_decode_h32::EVENT_V_S);
        for (uint32_t r = 0; r < rows; ++r) {
            beta[r] = 1.0f / brawLocal.GetValue(r);
            // The epsilon goes inside the sqrt, as upstream places it.
            qNorm[r] = sqrt(nrmLocal.GetValue(r) + 1e-6f);
            kNorm[r] = sqrt(nrmLocal.GetValue(8 + r) + 1e-6f);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(kda_decode_h32::EVENT_S_V);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(kda_decode_h32::EVENT_S_V);

        // ── b_q = b_q / norm; b_q *= scale; b_k = b_k / norm ────────────
        //
        // Keep that order and keep both as real divisions: folding them into
        // `scale / qNorm`, or into a reciprocal multiply, reassociates and
        // costs a bf16 ulp on a few output elements. `Divs` is unsupported, so
        // the norms are broadcast from the registers the Scalar visit left
        // them in.
        for (uint32_t r = 0; r < rows; ++r) {
            AscendC::Duplicate(normLocal[r * D], qNorm[r], D);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(qnLocal, qnLocal, normLocal, rows * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(qnLocal, qnLocal, scale_, rows * D);
        for (uint32_t r = 0; r < rows; ++r) {
            AscendC::Duplicate(normLocal[r * D], kNorm[r], D);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(knLocal, knLocal, normLocal, rows * D);
        AscendC::PipeBarrier<PIPE_V>();

        // ══ RECURRENCE ══════════════════════════════════════════════════
        //
        // The state-dependent half, one token at a time and in Triton's order:
        //
        //     b_h *= tl.exp(b_g[None, :])
        //     b_v -= tl.sum(b_h * b_k[None, :], axis=1)
        //     b_v *= b_beta
        //     b_h += b_k[None, :] * b_v[:, None]
        //     b_o  = tl.sum(b_h * b_q[None, :], axis=1)
        //
        // This order is the dependency chain, so it is the one part of the
        // kernel that cannot be regrouped.
        for (uint32_t r = 0; r < rows; ++r) {
            const int32_t token = tok0 + static_cast<int32_t>(r);
            const uint64_t row = static_cast<uint64_t>(token) * H_TOTAL + head;
            auto qRow = qnLocal[r * D];
            auto kRow = knLocal[r * D];
            auto gRow = gdLocal[r * D];

            // ── b_h *= tl.exp(b_g[None, :]) ─────────────────────────────
            AscendC::Mul(stateLocal, stateLocal, gRow, 64, BV, decay_params);
            AscendC::Mul(stateLocal[64], stateLocal[64], gRow[64],
                         64, BV, decay_params);
            AscendC::PipeBarrier<PIPE_V>();

            // ── b_v -= tl.sum(b_h * b_k[None, :], axis=1) ───────────────
            AscendC::Mul(mulReduceLocal, stateLocal, kRow,
                         64, BV, mul_reduce_params);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst(mulReduceLocal, stateLocal[64], kRow[64],
                               64, BV, mul_reduce_params);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::WholeReduceSum<float>(reduceLocal, mulReduceLocal,
                                           64, BV, 1, 1, 8);

            AscendC::Cast(tmpBVLocal, vbLocal[r * D],
                          AscendC::RoundMode::CAST_NONE, BV);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(tmpBVLocal, tmpBVLocal, reduceLocal, BV);
            AscendC::PipeBarrier<PIPE_V>();

            // ── b_v *= b_beta ───────────────────────────────────────────
            //
            // From a register, so no pipe crossing.
            AscendC::Muls(tmpBVLocal, tmpBVLocal, beta[r], BV);
            AscendC::PipeBarrier<PIPE_V>();

            // ── b_h += b_k[None, :] * b_v[:, None] ──────────────────────
            AscendC::Brcb(brcbLocal, tmpBVLocal, BV / 8, {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::MulAddDst(stateLocal, kRow, brcbLocal,
                               64, BV, outer_params);
            AscendC::MulAddDst(stateLocal[64], kRow[64], brcbLocal,
                               64, BV, outer_params);
            AscendC::PipeBarrier<PIPE_V>();

            // ── tl.store(p_mid, b_h, mask_h & (state_index > 0)) ────────
            //
            // Same position in the step as upstream: after the update, before
            // the output. Issued with its wait deferred to end-of-step, so
            // MTE3 can walk the 64 KB tile while the output reduction below
            // reads `stateLocal` without writing it.
            //
            // The overlap this buys is smaller than it looks, and measurement
            // says it is not worth restructuring for: `SetFlag` executes only
            // once every prior instruction in its source pipe has finished, so
            // the output store's own `SetFlag<V_MTE3>` below subsumes this
            // store and its `WaitFlag<MTE3_V>` already drains the checkpoint.
            // Batching the output store out of this loop removes that and
            // measured no faster (verify R2 36.98us vs 37.58us, medians of 3),
            // because MTE3 is only ~8% of the block's time.
            bool midInFlight = false;
            if constexpr (SAVE_INTERMEDIATE) {
                if (slot > 0) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                        kda_decode_h32::EVENT_V_MTE3_MID);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                        kda_decode_h32::EVENT_V_MTE3_MID);
                    AscendC::DataCopy(
                        midGm[pmid + static_cast<uint64_t>(token - bos) *
                                         midStrideStep_],
                        stateLocal, BV * D);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                        kda_decode_h32::EVENT_MTE3_V_MID);
                    midInFlight = true;
                }
            }

            // ── b_o = tl.sum(b_h * b_q[None, :], axis=1); store(p_o) ────
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
                kda_decode_h32::EVENT_V_MTE3_OUT);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                kda_decode_h32::EVENT_V_MTE3_OUT);
            AscendC::DataCopy(outGm[row * D], outLocal, BV);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                kda_decode_h32::EVENT_MTE3_V_OUT);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                kda_decode_h32::EVENT_MTE3_V_OUT);

            // Collect the checkpoint before the next step's gate multiply
            // touches the tile, and before the final store or the kernel exit
            // can race it.
            if (midInFlight) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                    kda_decode_h32::EVENT_MTE3_V_MID);
            }
        }
    }

    // ── if not SAVE_INTERMEDIATE and state_index > 0: store(p_h0, b_h) ───
    //
    // Under verify the commit reads the accepted step out of the checkpoint
    // buffer, so the source slot must keep its pre-verify value. Both sides are
    // V-major with K contiguous, making this a contiguous copy of BV rows; one
    // core owns the whole span, so no other core shares a datablock.
    if constexpr (!SAVE_INTERMEDIATE) {
        if (slot > 0) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                kda_decode_h32::EVENT_V_MTE3_OUT);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                kda_decode_h32::EVENT_V_MTE3_OUT);
            AscendC::DataCopy(stateGm[pstate], stateLocal, BV * D);
            // MTE3 reads `stateLocal` directly, and this is the last thing the
            // block does, so the copy has to be seen out before it exits.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                kda_decode_h32::EVENT_MTE3_V_OUT);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                kda_decode_h32::EVENT_MTE3_V_OUT);
        }
    }
}

// ── Entry points ──────────────────────────────────────────────────────
//
// Two kernels differing only in `SAVE_INTERMEDIATE`, a template parameter
// rather than a runtime flag because it changes the store schedule inside the
// innermost loop. The macro exists because a `__global__` kernel cannot be a
// template and the nineteen-parameter signature has to match the launcher
// exactly.
#define KDA_H32_KERNEL(NAME, SAVE)                                      \
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
    KdaVarlenRecurrentDecodeH32<SAVE> op;                               \
    op.Init(Q, K, V, A, B, A_LOG, DT_BIAS,                              \
            STATE, INDICES, STARTS, OUT, MID,                           \
            MID_STRIDE_REQ, MID_STRIDE_STEP, MID_STRIDE_HEAD,           \
            scale, lowerBound, numReq);                                 \
    op.Process();                                                       \
}

// One head per core, not split across cores along the value axis. Splitting was
// a template parameter until it measured a loss, because every tile repeats
// that head's whole prologue and these shapes are serial in the token index
// rather than short of cores (`msprof op`, medians of 3, idle card):
//
//   512K DP4TP2 R1 (32 blocks)   1 tile 23.8us   2 tiles 30.2us
//   512K DP2TP4 R1 (16 blocks)   1 tile 19.7us   4 tiles 19.6us
//
// Three MTE3 restructurings were tried and reverted, all bit-exact, none
// faster (`msprof op`, medians of 3, idle card): batching the per-token output
// store into one strided copy per group, overlapping the final 64 KB state
// store with the last token's output reduction, and deferring the output
// store's `MTE3_V` wait. decode B1 8.48us either way; verify R2 36.98 against
// 37.58. The reason is in the pipe profile — on decode B1, MTE3 is 0.58us of a
// 7.06us block, so the whole target is ~8% and the pipes barely overlap
// (vec 3.34 + scalar 2.25 + mte2 1.17 + mte3 0.58 ≈ aiv_time 7.06).
//
// Watch the bench numbers for this kernel: ~120us of the ~129us an eager
// decode B1 call takes is launch overhead, so bench deltas there say nothing
// about the kernel. Profile it.
//
// The remaining concentration is Scalar: 32% of decode B1, and ~4.9us of
// scalar/vector stall on verify R2. That is the group's one Scalar visit for
// beta's reciprocal and the two `sqrt`s, which bit-exactness pins to the
// Scalar unit.
//
// The symbol names are the ones the shared header used to emit for this width,
// so profiles taken before and after the fork still line up.
KDA_H32_KERNEL(kda_decode_k32,     false)
KDA_H32_KERNEL(kda_decode_k32_mid, true)

#ifndef __CCE_KT_TEST__

// One block per (request, head), so the grid scales with `numReq`.
//
// `args.mid` is null outside the verify regime and the stride fields are then
// meaningless, but the plain kernel ignores all four, so they are passed either
// way to keep one call expression.
void LaunchKdaDecodeH32(const KdaLaunchArgs &args, void *stream)
{
    const uint32_t blockDim = args.num_req * kda_decode_h32::kHeads;
    auto acl_stream = static_cast<aclrtStream>(stream);

#define KDA_H32_CALL(KERNEL)                                            \
    KERNEL<<<blockDim, nullptr, acl_stream>>>(                           \
        args.q, args.k, args.v, args.a, args.b,                          \
        args.A_log, args.dt_bias,                                        \
        args.state, args.indices, args.starts, args.out,                 \
        args.mid, args.mid_stride_req, args.mid_stride_step,             \
        args.mid_stride_head,                                            \
        args.scale, args.lower_bound, args.num_req)

    if (args.mid != nullptr) {
        KDA_H32_CALL(kda_decode_k32_mid);
    } else {
        KDA_H32_CALL(kda_decode_k32);
    }

#undef KDA_H32_CALL
}

#endif
