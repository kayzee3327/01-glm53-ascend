// GLM-5.3 KDA varlen recurrence — prefill kernel, H=4 (TP16).
//
// One packed sequence (numReq==1) and a long serial token loop. See
// `kda_varlen_recurrent.h` for the shared contract and the reference semantics;
// the host file validates everything before this launcher is reached.
//
// ── Why this kernel has two kinds of block ─────────────────────────────
//
// The recurrence is serial in the token index, so the only parallelism is
// across (head, value tile) — 32 work items at H=4, BV=16. That leaves 8 of a
// 910B's 40 vector cores with nothing to do, while the per-token inputs the
// recurrence consumes (the gate, beta, and the q/k L2 normalization) are
// entirely independent across tokens and are *not* serial at all.
//
// Upstream splits exactly along that line, but across two launches: a parallel
// Triton `_prepare_inputs` writing four FP32 tensors, then the serial
// `_prepared_recurrent` reading them. Measured on-device, the prepare half is
// ~19% of the prepared path's time (H4/T4096: 1.16 ms of 6.61 ms) and it is
// pure latency, because the recurrence cannot start until all of it lands.
//
// So this kernel runs both halves at once in one grid:
//
//     blocks  0..31   consumers — the serial recurrence, one state tile each
//     blocks 32..39   producers — prepare per-token inputs, running ahead
//
// Consumers keep `prepared_recurrent.cpp`'s block decomposition unchanged,
// which is what makes its UB map, transpose lists and token loop reusable:
//
//     iv   = blockIdx % NUM_V_TILES
//     head = blockIdx / NUM_V_TILES
//     state pointer = (slot * H_TOTAL + head) * D * D + iv * BV * D
//
// The state slot is K-contiguous (see `kda_varlen_recurrent.h`), so a tile of
// BV value rows is a contiguous span and V is the axis that strides by D.
//
// Producers take two per head and split the token stream into chunks, even
// chunks to one and odd to the other, so a head's inputs are always being
// prepared from two cores while its 8 consumers read them.
//
// ── What this costs and why it is safe ────────────────────────────────
//
// Communication is through GM — the four FP32 intermediates, laid out exactly
// as upstream lays them out (row = token * H + head), so the consumer's
// indexing is the same expression it would use against the Triton output.
// Notification is `IBSet`/`IBWait`, one bank per (producer, consumer lane, ring
// slot). Measured cost: ~61 cycles per chunk in the consumer's `IBWait` and
// ~122 in the producer's 8 `IBSet`s, against ~3000 cycles per token of real
// consumer work. At 16 tokens per chunk that is noise.
//
// Three hazards worth naming, all of them load-bearing:
//
//   * `IBWait` *clears* the flag it consumes, so one `IBSet` can be awaited by
//     exactly one `IBWait`. A producer therefore signals each of its 8
//     consumers on its own bank; it does not broadcast once.
//   * `IBSet` blocks while the flag is still set, which is what bounds producer
//     run-ahead — but with a single bank per pair that bound is one chunk,
//     which would serialize producer and consumer. Hence the small ring of
//     `eventID`s: `kSyncRing` chunks of slack per pair.
//   * `IBSet`/`IBWait` call `GetTPipePtr()->FetchEventID()` internally, so a
//     `TPipe` must exist even though every buffer here is at a hand-laid
//     absolute address. Without one it is a null dereference that surfaces as
//     "MPU address access is invalid" on the waiting cores. The `TPipe` below
//     never gets an `InitBuffer`, so it reserves no UB and the absolute map is
//     unaffected.
//
// The grid must not exceed the physical core count or a wave of blocked
// consumers can exclude the producers they wait on; `kda_prefill::kBlockDim`
// static_asserts this.
//
// ── Divergence from `prepared_recurrent` ──────────────────────────────
//
// The consumer half is that kernel, with two changes: the chunk-boundary
// `IBWait` is hoisted to just before the prefetch of the next token (so the
// software pipeline stays full and the wait is not paid per token), and the
// inputs come from this kernel's own producers rather than from Triton.
//
// The producer half is new, and it is the one place bit-exactness against
// upstream is not guaranteed: Triton's `q / sqrt(sum(q*q) + 1e-6)` is a
// division, and the Ascend sequence for it reassociates slightly. Everything
// else — the `+1e-6` inside the sqrt, `scale` applied after normalization, the
// L2 (not RMS) reading — is matched operation for operation.

#include "kernel_operator.h"
#include "kda_varlen_recurrent.h"
#include "utils.h"

#ifndef __CCE_KT_TEST__
#include <acl/acl.h>
#endif

// ── Device-side ────────────────────────────────────────────────────────

// Hardware event IDs, managed by hand as in `prepared_recurrent`. Avoid 6/7.
// Producer and consumer never run on the same core, so the two sets may reuse
// ids freely; they are kept distinct anyway to keep traces readable.
namespace {

// Consumer, lifted from `prepared_recurrent.cpp`.
constexpr int32_t EVENT_MTE2_V_S = 0;
constexpr int32_t EVENT_MTE2_V_QKVD_0 = 1;
constexpr int32_t EVENT_MTE2_V_QKVD_1 = 2;
constexpr int32_t EVENT_V_S_BETA = 0;
constexpr int32_t EVENT_S_V = 2;
constexpr int32_t EVENT_V_MTE2 = 3;
constexpr int32_t EVENT_MTE3_V = 4;
constexpr int32_t EVENT_MTE3_V_TRACK = 0;
constexpr int32_t EVENT_V_MTE3 = 5;

// Producer. Event ids are scoped per (source, destination) pipe pair, so these
// may repeat ids used above for a different pair.
constexpr int32_t EVENT_MTE2_V_PROD = 1;
constexpr int32_t EVENT_V_MTE3_PROD = 2;
constexpr int32_t EVENT_MTE3_V_PROD = 3;
constexpr int32_t EVENT_MTE3_S_PROD = 4;
constexpr int32_t EVENT_V_MTE2_PROD = 5;
// This architecture has no scalar `exp`, so every sigmoid and every exp in the
// producer round-trips through the Vector unit: Scalar writes the operand to UB,
// Vector computes, Scalar reads the result back. Both directions need a real
// hardware event — `PipeBarrier<PIPE_V>` does not order Vector against Scalar.
constexpr int32_t EVENT_S_V_PROD = 0;
constexpr int32_t EVENT_V_S_PROD = 0;

}  // namespace

// TRACK_STATE   compile out the prefix-checkpoint store when unused.
// H_TOTAL       heads per state slot; 4 for prefill at TP16.
template<bool TRACK_STATE, uint32_t H_TOTAL>
class KdaVarlenRecurrentPrefillH4 {
public:
    __aicore__ inline void Init(
        GM_ADDR Q, GM_ADDR K, GM_ADDR V, GM_ADDR A,   // bf16 [1,T,H,D]
        GM_ADDR B,                                     // bf16 [1,T,H]
        GM_ADDR A_LOG,                                 // fp32 [H]
        GM_ADDR DT_BIAS,                               // fp32 [H*D]
        GM_ADDR STATE,                                 // fp32, K-contiguous
        GM_ADDR INDICES,                               // int32 [numReq]
        GM_ADDR STARTS,                                // int32 [numReq+1]
        GM_ADDR OUT,                                   // bf16 [1,T,H,D]
        GM_ADDR TRACK_INDICES, GM_ADDR TRACK_LENS,     // int32 [numReq]
        GM_ADDR QN, GM_ADDR KN, GM_ADDR GDECAY,        // fp32 [1,T,H,D]
        GM_ADDR BETA,                                  // fp32 [1,T,H]
        GM_ADDR SYNC,                                  // int32, zeroed
        float scale,
        float lowerBound,
        uint32_t numReq,
        uint32_t numTokens);

    __aicore__ inline void Process();

private:
    struct config {
        static constexpr uint32_t D = static_cast<uint32_t>(config_D);
        static constexpr uint32_t BV = 16;
        static constexpr uint32_t NUM_V_TILES = D / BV;   // 8
        static constexpr uint32_t NUM_REQ = 1;
        static constexpr uint32_t STATE_STRIDE = 128;

        static constexpr uint32_t NUM_CONSUMERS = kda_prefill::kNumConsumers;
        static constexpr uint32_t NUM_PRODUCERS = kda_prefill::kNumProducers;
        static constexpr uint32_t CHUNK = kda_prefill::kChunkTokens;
        static constexpr uint32_t RING = kda_prefill::kSyncRing;

        // Consumer lanes fed by one producer, and producers per head.
        static constexpr uint32_t LANES_PER_HEAD = NUM_V_TILES;            // 8
        static constexpr uint32_t PRODUCERS_PER_HEAD =
            NUM_PRODUCERS / (NUM_CONSUMERS / NUM_V_TILES);                 // 2

        static_assert(NUM_CONSUMERS == NUM_V_TILES * NUM_REQ * 4,
                      "32 consumers = 8 value tiles x 4 heads");
        static_assert(PRODUCERS_PER_HEAD >= 1,
                      "each head needs at least one producer");
    };

    // UB map. The consumer half is `prepared_recurrent`'s map at the same
    // addresses, deliberately: the token loop's repeat strides are derived from
    // these offsets. The producer half and the sync scratch go in the 57 KB
    // hole that map leaves between the state tile (8 KB at 0x0) and 0x10100.
    //
    // Producer and consumer never coexist on a core, so the two regions could
    // overlap; they are kept disjoint so that a misrouted block corrupts
    // nothing and traces stay legible.
    struct UbLayout {
        // ── shared ──
        // 32 B of private scratch for the IBSet/IBWait handshake. Every core
        // needs its own; this one is read and written by the sync API only.
        static constexpr uint32_t SYNC_ADDR = 0x02000;
        static constexpr uint32_t SYNC_SIZE = 32;

        // ── consumer (from prepared_recurrent.cpp) ──
        static constexpr uint32_t STATE_ADDR = 0x00000;
        static constexpr uint32_t STATE_SIZE =
            config::BV * config::STATE_STRIDE * sizeof(float);
        static_assert(STATE_ADDR + STATE_SIZE <= SYNC_ADDR,
                      "the state tile must not reach the sync scratch");

        static constexpr uint32_t Q0_ADDR     = 0x10100;
        static constexpr uint32_t Q1_ADDR     = 0x10500;
        static constexpr uint32_t K0_ADDR     = 0x10900;
        static constexpr uint32_t K1_ADDR     = 0x10D00;
        static constexpr uint32_t DECAY0_ADDR = 0x11100;
        static constexpr uint32_t DECAY1_ADDR = 0x11500;
        static constexpr uint32_t V0_ADDR     = 0x11900;
        static constexpr uint32_t V1_ADDR     = 0x11920;

        static constexpr uint32_t BRCB_ADDR       = 0x12000;
        static constexpr uint32_t REDUCE_ADDR     = 0x12300;
        static constexpr uint32_t OUT_ADDR        = 0x20000;
        static constexpr uint32_t MUL_REDUCE_ADDR = 0x20200;
        static constexpr uint32_t TMP_BV_ADDR     = 0x21440;

        // ── producer ──
        // One chunk of C tokens, each a 128-lane row. Staged bf16 inputs are
        // cast up to fp32 in place of being recomputed, and the four outputs
        // are built contiguously so each leaves in one DataCopy.
        static constexpr uint32_t C = config::CHUNK;
        static constexpr uint32_t D_ = config::D;
        static constexpr uint32_t ROW_F32 = D_ * sizeof(float);    // 512
        static constexpr uint32_t ROW_BF16 = D_ * sizeof(bfloat16_t);

        static constexpr uint32_t P_BASE = 0x02100;                 // above SYNC
        // bf16 staging for q/k/a of the whole chunk.
        static constexpr uint32_t P_QB_ADDR = P_BASE;                      // C*256
        static constexpr uint32_t P_KB_ADDR = P_QB_ADDR + C * ROW_BF16;
        static constexpr uint32_t P_AB_ADDR = P_KB_ADDR + C * ROW_BF16;
        // fp32 outputs for the whole chunk.
        static constexpr uint32_t P_QN_ADDR = P_AB_ADDR + C * ROW_BF16;
        static constexpr uint32_t P_KN_ADDR = P_QN_ADDR + C * ROW_F32;
        static constexpr uint32_t P_GD_ADDR = P_KN_ADDR + C * ROW_F32;
        // beta, a chunk at a time: the 1 + exp(-b) denominators built by
        // Vector, and the sigmoid itself, which Scalar divides into place.
        static constexpr uint32_t P_BRAW_ADDR = P_GD_ADDR + C * ROW_F32;
        static constexpr uint32_t P_BT_ADDR = P_BRAW_ADDR + 64;
        // per-head constants, loaded once: the dt_bias row.
        static constexpr uint32_t P_BIAS_ADDR = P_BT_ADDR + 64;

        // ── normalization scratch, sized for a whole chunk ──
        //
        // Rows carry no dependency on one another, so the normalization runs
        // as one strided instruction per step across all `rows` rather than a
        // per-row sequence. That costs the chunk-wide buffers below (16 KB of
        // the 57 KB hole) and buys back 2*C Vector/Scalar round trips, which
        // is what the per-row form actually spent its time on.
        //
        //   P_SQ   squares of one tensor's chunk; folded, then reused by k
        //   P_RED  the folded 64-lane half-sums, q's rows then k's
        //   P_NSQ  sum(x*x) per row, q then k, as the reduce leaves them
        //   P_QS   the roots, Scalar's output and `Brcb`'s input
        //   P_QBC  one datablock per row, all eight lanes that row's norm
        static constexpr uint32_t P_SQ_ADDR = P_BIAS_ADDR + ROW_F32;
        static constexpr uint32_t P_RED_ADDR = P_SQ_ADDR + C * ROW_F32;
        // Two tensors x C rows x 64 lanes.
        static constexpr uint32_t P_NSQ_ADDR =
            P_RED_ADDR + 2 * C * (D_ / 2) * sizeof(float);
        // `Brcb` consumes 8 elements per repeat, so its source is rounded up
        // to a whole repeat; the tail lanes are never broadcast into a row
        // that the strided `Div` below actually visits.
        static constexpr uint32_t BRCB_SRC = (C + 7) / 8 * 8 * sizeof(float);
        static constexpr uint32_t P_QS_ADDR =
            P_NSQ_ADDR + 2 * C * sizeof(float);
        static constexpr uint32_t P_KS_ADDR = P_QS_ADDR + BRCB_SRC;
        static constexpr uint32_t P_QBC_ADDR = P_KS_ADDR + BRCB_SRC;
        static constexpr uint32_t P_KBC_ADDR = P_QBC_ADDR + C * 32;

        // One repeat of lower_bound, the gate division's numerator. `Div`
        // takes both operands as tensors, so the scalar has to be broadcast;
        // a src0 repeat stride of 0 then reuses this one repeat for all rows.
        static constexpr uint32_t P_LB_ADDR = P_KBC_ADDR + C * 32;
        // Scalar/Vector exchange slot. This architecture has no scalar `exp`,
        // so exp(A_log) is computed on the Vector unit and read back here.
        static constexpr uint32_t P_SCR_ADDR = P_LB_ADDR + 256;
        static constexpr uint32_t P_END = P_SCR_ADDR + 256;

        // The producer region lives entirely in the hole below Q0.
        static_assert(P_END <= Q0_ADDR,
                      "producer buffers must fit under the consumer's map");

        static constexpr uint32_t UB_END =
            TMP_BV_ADDR + config::BV * sizeof(float);
        static_assert(UB_END <= 192 * 1024);
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
    AscendC::GlobalTensor<int32_t> indicesGm;
    AscendC::GlobalTensor<int32_t> startsGm;
    AscendC::GlobalTensor<int32_t> trackIndicesGm;
    AscendC::GlobalTensor<int32_t> trackLensGm;

    // The hoisted per-token inputs, shared between the two halves.
    AscendC::GlobalTensor<float> qnGm;
    AscendC::GlobalTensor<float> knGm;
    AscendC::GlobalTensor<float> decayGm;
    AscendC::GlobalTensor<float> betaGm;
    AscendC::GlobalTensor<int32_t> syncGm;

    float scale_ = 0.0f;
    float lowerBound_ = 0.0f;
    uint32_t numTokens_ = 0;
    // Row stride of the beta scratch, which is [H, betaStride_] rather than
    // upstream's [T, H]. See the store in `RunProducer` for why.
    uint32_t betaStride_ = 0;

    AscendC::LocalTensor<int32_t> syncLocal;

    // Consumer buffers.
    AscendC::LocalTensor<float> stateLocal;
    AscendC::LocalTensor<float> qLocal[2];
    AscendC::LocalTensor<float> kLocal[2];
    AscendC::LocalTensor<float> decayLocal[2];
    AscendC::LocalTensor<bfloat16_t> vLocal[2];
    AscendC::LocalTensor<bfloat16_t> outLocal;

    __aicore__ inline void LoadStateTile(
        const AscendC::LocalTensor<float>& state,
        uint64_t gmOffset);
    __aicore__ inline void StoreStateTile(
        const AscendC::LocalTensor<float>& state,
        uint64_t gmOffset);
    __aicore__ inline void SaveTrackedState(uint64_t gmOffset);

    __aicore__ inline void RunConsumer(uint32_t blockIdx);
    __aicore__ inline void RunProducer(uint32_t blockIdx);

    // Block until `chunk` is published, then make its beta line readable by
    // this core's Scalar unit. A member function rather than a local lambda
    // because a lambda in this dialect is a host function and cannot call
    // `__aicore__` APIs.
    __aicore__ inline void WaitChunk(uint32_t blockIdx, uint32_t head,
                                     uint64_t betaBase, uint32_t chunk);

    // Bank identity for one (producer, consumer lane, chunk). `eventID` carries
    // the consumer lane and the ring slot; `blockIdx` (the API's argument, not
    // ours) carries the producing core. Both sides must compute this the same
    // way, so it lives in one place.
    static __aicore__ inline int32_t SyncEventId(uint32_t consumerBlock,
                                                 uint32_t chunk)
    {
        const uint32_t slot = (chunk / config::PRODUCERS_PER_HEAD) %
                              config::RING;
        return static_cast<int32_t>(consumerBlock * config::RING + slot);
    }
};

template<bool TRACK_STATE, uint32_t H_TOTAL>
__aicore__ inline void KdaVarlenRecurrentPrefillH4<TRACK_STATE, H_TOTAL>::Init(
    GM_ADDR Q, GM_ADDR K, GM_ADDR V, GM_ADDR A,
    GM_ADDR B,
    GM_ADDR A_LOG,
    GM_ADDR DT_BIAS,
    GM_ADDR STATE,
    GM_ADDR INDICES,
    GM_ADDR STARTS,
    GM_ADDR OUT,
    GM_ADDR TRACK_INDICES, GM_ADDR TRACK_LENS,
    GM_ADDR QN, GM_ADDR KN, GM_ADDR GDECAY, GM_ADDR BETA,
    GM_ADDR SYNC,
    float scale,
    float lowerBound,
    uint32_t numReq,
    uint32_t numTokens)
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
    if constexpr (TRACK_STATE) {
        trackIndicesGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(TRACK_INDICES));
        trackLensGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(TRACK_LENS));
    }
    qnGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(QN));
    knGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(KN));
    decayGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GDECAY));
    betaGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(BETA));
    syncGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(SYNC));

    scale_ = scale;
    lowerBound_ = lowerBound;
    numTokens_ = numTokens;
    // Rounded up to a whole chunk so each chunk store is 64 B-aligned and
    // owned by one core. Must match the host's allocation.
    betaStride_ = (numTokens + config::CHUNK - 1) / config::CHUNK *
                  config::CHUNK;

    constexpr uint32_t BV = config::BV;
    constexpr uint32_t D = config::D;

    syncLocal = AscendC::LocalTensor<int32_t>(
        AscendC::TPosition::VECCALC, UbLayout::SYNC_ADDR, 8);

    stateLocal = AscendC::LocalTensor<float>(
        AscendC::TPosition::VECCALC, UbLayout::STATE_ADDR,
        BV * config::STATE_STRIDE);
    qLocal[0] = AscendC::LocalTensor<float>(
        AscendC::TPosition::VECCALC, UbLayout::Q0_ADDR, D);
    qLocal[1] = AscendC::LocalTensor<float>(
        AscendC::TPosition::VECCALC, UbLayout::Q1_ADDR, D);
    kLocal[0] = AscendC::LocalTensor<float>(
        AscendC::TPosition::VECCALC, UbLayout::K0_ADDR, D);
    kLocal[1] = AscendC::LocalTensor<float>(
        AscendC::TPosition::VECCALC, UbLayout::K1_ADDR, D);
    decayLocal[0] = AscendC::LocalTensor<float>(
        AscendC::TPosition::VECCALC, UbLayout::DECAY0_ADDR, D);
    decayLocal[1] = AscendC::LocalTensor<float>(
        AscendC::TPosition::VECCALC, UbLayout::DECAY1_ADDR, D);
    vLocal[0] = AscendC::LocalTensor<bfloat16_t>(
        AscendC::TPosition::VECCALC, UbLayout::V0_ADDR, BV);
    vLocal[1] = AscendC::LocalTensor<bfloat16_t>(
        AscendC::TPosition::VECCALC, UbLayout::V1_ADDR, BV);
    outLocal = AscendC::LocalTensor<bfloat16_t>(
        AscendC::TPosition::VECCALC, UbLayout::OUT_ADDR, BV);
}

// ── State tile movement ────────────────────────────────────────────────
//
// GM holds each state slot as [V, K] with K contiguous: the pool is allocated
// [slots, H, V, K] and handed to the kernel as a transposed view, so stride(K)
// is 1. The token loop wants [V, K] too, so the reduction over K runs along
// contiguous lanes — the orientations agree and a tile is a contiguous span.
//
// `prepared_recurrent` moves this tile through a [128, 24] staging buffer and
// a pair of `TransDataTo5HD` transposes, having read the layout as [K, V]. The
// two transposes are exact inverses, so a load/store round trip is unaffected
// and its outputs still match; what it leaves in GM is the state transposed.
// Here both directions are a plain `DataCopy`, which is also one MTE
// instruction instead of 24 and frees the 12 KB staging buffer.

template<bool TRACK_STATE, uint32_t H_TOTAL>
__aicore__ inline void
KdaVarlenRecurrentPrefillH4<TRACK_STATE, H_TOTAL>::LoadStateTile(
    const AscendC::LocalTensor<float>& state,
    uint64_t gmOffset)
{
    // GM and UB hold the tile in the same orientation — V-major with the D
    // values of K contiguous — so this is a straight contiguous copy of BV
    // whole rows. `gmOffset` already points at row `iv * BV`.
    AscendC::DataCopy(state, stateGm[gmOffset], config::BV * config::D);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V_S);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V_S);
}

template<bool TRACK_STATE, uint32_t H_TOTAL>
__aicore__ inline void
KdaVarlenRecurrentPrefillH4<TRACK_STATE, H_TOTAL>::StoreStateTile(
    const AscendC::LocalTensor<float>& state,
    uint64_t gmOffset)
{
    // The mirror of `LoadStateTile`: same orientation on both sides, so the
    // tile leaves as BV contiguous rows. Each tile owns a disjoint, 64 B
    // aligned span of GM (BV * D floats = 8 KB), so concurrent stores from the
    // 8 consumers of a head cannot share a datablock.
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);

    AscendC::DataCopy(stateGm[gmOffset], state, config::BV * config::D);
}

template<bool TRACK_STATE, uint32_t H_TOTAL>
__aicore__ inline void
KdaVarlenRecurrentPrefillH4<TRACK_STATE, H_TOTAL>::SaveTrackedState(
    uint64_t gmOffset)
{
    // Contains its own V_MTE3 handshake.
    StoreStateTile(stateLocal, gmOffset);

    // MTE3 reads `stateLocal` directly now, so the token loop must not resume
    // writing it until the checkpoint has left.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V_TRACK);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V_TRACK);
}

// ── Producer ──────────────────────────────────────────────────────────
//
// Mirrors `_prepare_inputs` in kda_prefill_npu.py, one chunk of C tokens at a
// time for a single head:
//
//     decay = exp(A_log)                      (per head, scalar)
//     gate  = lower_bound * sigmoid(decay * (a + dt_bias))
//     GDECAY = exp(gate)
//     BETA   = sigmoid(b)
//     QN     = q / sqrt(sum(q*q) + 1e-6) * scale
//     KN     = k / sqrt(sum(k*k) + 1e-6)
//
// Written as sigmoid(x) = 1 / (1 + exp(-x)), matching upstream's own spelling,
// so the only reassociation is in the normalizing division.
template<bool TRACK_STATE, uint32_t H_TOTAL>
__aicore__ inline void
KdaVarlenRecurrentPrefillH4<TRACK_STATE, H_TOTAL>::RunProducer(
    uint32_t blockIdx)
{
    constexpr uint32_t D = config::D;
    constexpr uint32_t C = config::CHUNK;
    constexpr uint32_t PPH = config::PRODUCERS_PER_HEAD;
    constexpr uint32_t LANES = config::LANES_PER_HEAD;

    const uint32_t prod = blockIdx - config::NUM_CONSUMERS;  // 0..NUM_PRODUCERS-1
    const uint32_t head = prod / PPH;
    const uint32_t phase = prod % PPH;

    const int32_t bos = startsGm.GetValue(0);
    const int32_t eos = startsGm.GetValue(1);
    // An empty sequence yields zero chunks and the loop below never runs, so
    // no `IBSet` is skipped while a consumer waits on it. There is deliberately
    // no early `return` here: `IBWait` has no timeout, so a producer that exits
    // before signalling would hang every consumer of its head rather than
    // reporting an error. The consumer's own `bos < eos` guard keeps the other
    // side of that invariant.
    const int32_t seqLen = eos - bos;
    const uint32_t numChunks =
        seqLen > 0 ? (static_cast<uint32_t>(seqLen) + C - 1) / C : 0u;

    AscendC::LocalTensor<bfloat16_t> qbLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_QB_ADDR, C * D);
    AscendC::LocalTensor<bfloat16_t> kbLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_KB_ADDR, C * D);
    AscendC::LocalTensor<bfloat16_t> abLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_AB_ADDR, C * D);
    AscendC::LocalTensor<float> qnLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_QN_ADDR, C * D);
    AscendC::LocalTensor<float> knLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_KN_ADDR, C * D);
    AscendC::LocalTensor<float> gdLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_GD_ADDR, C * D);
    AscendC::LocalTensor<float> brawLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_BRAW_ADDR, C);
    AscendC::LocalTensor<float> btLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_BT_ADDR, C);
    AscendC::LocalTensor<float> biasLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_BIAS_ADDR, D);
    AscendC::LocalTensor<float> sqLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_SQ_ADDR, C * D);
    AscendC::LocalTensor<float> redLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_RED_ADDR, 2 * C * (D / 2));
    AscendC::LocalTensor<float> nsqLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_NSQ_ADDR, 2 * C);
    AscendC::LocalTensor<float> qsLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_QS_ADDR, C);
    AscendC::LocalTensor<float> ksLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_KS_ADDR, C);
    AscendC::LocalTensor<float> qbcLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_QBC_ADDR, C * 8);
    AscendC::LocalTensor<float> kbcLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_KBC_ADDR, C * 8);
    AscendC::LocalTensor<float> lbLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_LB_ADDR, 64);
    AscendC::LocalTensor<float> scrLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_SCR_ADDR, 64);

    // ── Per-head constants, computed once ──
    //
    // `decay = exp(A_log[head])` is a scalar, but this architecture has no
    // scalar `exp`: the only `exp` is the Vector instruction. So A_log goes out
    // to UB on the Scalar pipe, through `Exp`, and back — two real hardware
    // events, since `PipeBarrier<PIPE_V>` does not order Vector against Scalar.
    AscendC::DataCopy(biasLocal, dtBiasGm[head * D], D);
    // `Exp` works a whole 8-float datablock at a time, so all eight lanes are
    // written rather than just the one that is read back — UB arrives
    // uninitialized and exp() of a garbage bit pattern is an avoidable inf.
    const float aLog = aLogGm.GetValue(head);
    for (uint32_t i = 0; i < 8; ++i) {
        scrLocal.SetValue(i, aLog);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_S_V_PROD);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_S_V_PROD);
    AscendC::Exp(scrLocal, scrLocal, 8);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_V_S_PROD);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_V_S_PROD);
    const float decay = scrLocal.GetValue(0);

    // The gate's division numerator, broadcast once. `Div` takes both operands
    // as tensors on this architecture and `Divs` is unsupported, so a scalar
    // numerator has to be materialized. `Reciprocal` would avoid this but is
    // documented as missing the double-ten-thousandth bar for float, and the
    // gate it would feed multiplies the state once per token, so the error
    // would compound across the sequence.
    AscendC::Duplicate(lbLocal, lowerBound_, 64);
    // The dt_bias row is read by Vector in the loop below.
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V_PROD);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V_PROD);

    for (uint32_t chunk = phase; chunk < numChunks; chunk += PPH) {
        const uint32_t tok0 = static_cast<uint32_t>(bos) + chunk * C;
        const uint32_t rows =
            (tok0 + C <= static_cast<uint32_t>(eos)) ? C
                                                     : static_cast<uint32_t>(eos) - tok0;

        // Rows are strided by H in GM, so one DataCopy per tensor per chunk
        // with a row gap of (H-1) rows. blockLen/gaps are in 32 B datablocks:
        // a 128-lane bf16 row is 256 B = 8 blocks, and the gap skips H-1 rows.
        AscendC::DataCopyParams rowParams{
            static_cast<uint16_t>(rows),
            static_cast<uint16_t>(D * sizeof(bfloat16_t) / 32),
            static_cast<uint16_t>((H_TOTAL - 1) * D * sizeof(bfloat16_t) / 32),
            0};
        const uint64_t base = static_cast<uint64_t>(tok0) * H_TOTAL * D
                              + static_cast<uint64_t>(head) * D;

        // Vector must be done reading last iteration's staging before MTE2
        // refills it. Paired with the Set at the bottom of the loop.
        if (chunk != phase) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2_PROD);
        }
        AscendC::DataCopy(qbLocal, qGm[base], rowParams);
        AscendC::DataCopy(kbLocal, kGm[base], rowParams);
        AscendC::DataCopy(abLocal, aGm[base], rowParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V_PROD);

        // b is [1,T,H]: one fp32 per row once converted, strided by H in GM.
        // One element per 32 B block is below the DMA's useful granularity, so
        // Scalar gathers the chunk and hands it to Vector in one go.
        for (uint32_t r = 0; r < rows; ++r) {
            const uint64_t bRow =
                static_cast<uint64_t>(tok0 + r) * H_TOTAL + head;
            brawLocal.SetValue(r, AscendC::ToFloat(bGm.GetValue(bRow)));
        }
        // Tail of a short final chunk: a finite placeholder, so the Vector
        // sequence below can run over the whole C-wide buffer.
        for (uint32_t r = rows; r < C; ++r) {
            brawLocal.SetValue(r, 0.0f);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_S_V_PROD);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_S_V_PROD);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V_PROD);

        // ── beta = sigmoid(b) = 1 / (1 + exp(-b)) ──
        //
        // The whole C-wide buffer, not just `rows`: the store below is one
        // fixed-size chunk, and on a short final chunk the tail lanes would
        // otherwise carry uninitialized UB through Exp. The padding is written
        // to GM and never read, but it must not be inf or NaN.
        AscendC::Muls(brawLocal, brawLocal, -1.0f, C);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(brawLocal, brawLocal, C);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(brawLocal, brawLocal, 1.0f, C);

        // The reciprocal runs on Scalar, one token at a time, rather than as a
        // Vector `Div`. `Div` is documented at 1 ulp here: its 0-ulp algorithms
        // (DivAlgo::PRECISION_0ULP_*) are 950-series only, so this hardware
        // gets DivAlgo::INTRINSIC. Upstream's division is IEEE-exact, and beta
        // scales the state update on every step of the recurrence, so a
        // last-bit difference compounds along the sequence. One division per
        // token is cheap next to the 128-lane work around it.
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_V_S_PROD);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_V_S_PROD);
        for (uint32_t r = 0; r < C; ++r) {
            btLocal.SetValue(r, 1.0f / brawLocal.GetValue(r));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_S_V_PROD);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_S_V_PROD);

        // ── GDECAY = exp(lower_bound * sigmoid(decay * (a + dt_bias))) ──
        //
        // Whole chunk at a time. The dt_bias row is 128 lanes but a Vector
        // repeat covers 64, so the add runs twice with a repeat stride of one
        // row (16 datablocks) and a src1 stride of 0 to hold bias in place.
        AscendC::PipeBarrier<PIPE_V>();
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
        // -(decay * x), ready for exp
        AscendC::Muls(gdLocal, gdLocal, -decay, rows * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(gdLocal, gdLocal, rows * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(gdLocal, gdLocal, 1.0f, rows * D);
        AscendC::PipeBarrier<PIPE_V>();
        // gate = lower_bound / (1 + exp(-decay*x)), with the numerator held in
        // place by a src0 repeat stride of 0 so one repeat of `lbLocal` covers
        // every row. Two passes, 64 lanes each, as above.
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

        // ── q, k: L2 normalize the whole chunk; q also takes `scale` ──
        //
        // The norm is a per-row scalar, but the rows are independent, so this
        // runs as one strided instruction per step over all `rows` instead of
        // a per-row sequence. What that saves is not vector issue slots — it
        // is the Vector/Scalar handshake. The root has to be taken on Scalar
        // (see below), and a per-row form pays a V_S and an S_V for every row,
        // 2*C full pipeline drains per chunk, each one also serializing the
        // row's own six vector instructions behind it. Batched, the chunk
        // costs exactly one round trip whatever C is.
        //
        // Every step below is elementwise or a 64-lane reduce over the same
        // 64 lanes as before, so no value's arithmetic changes — only how
        // many instructions carry it. That matters because this kernel is
        // bit-exact against Triton and the summation order is load-bearing.
        //
        // Strides are in datablocks (32 B = 8 floats): a 128-lane fp32 row is
        // 16, a 64-lane half-row is 8.
        constexpr uint8_t ROW_BLK = D * sizeof(float) / 32;        // 16
        constexpr uint8_t HALF_BLK = (D / 2) * sizeof(float) / 32; // 8
        const uint8_t reps = static_cast<uint8_t>(rows);

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(qnLocal, qbLocal, AscendC::RoundMode::CAST_NONE,
                      rows * D);
        AscendC::Cast(knLocal, kbLocal, AscendC::RoundMode::CAST_NONE,
                      rows * D);
        AscendC::PipeBarrier<PIPE_V>();

        // sum(x*x) over 128 lanes, in upstream's summation order.
        //
        // `tl.sum` over 128 fp32 lanes is bit-identical to `torch.sum(-1)`
        // and to neither a 2x64 nor a 4x32 decomposition. Of thirteen
        // candidate trees measured against it, exactly one matched on every
        // row: add the two 64-lane halves elementwise, then reduce the 64
        // partials as a binary tree over adjacent lanes. A 2x64
        // `WholeReduceSum` pair differs on ~28% of rows. The norm divides q
        // and k on every step of a recurrence, so a last-bit difference here
        // compounds along the sequence; the order has to match.
        //
        // The fold is the same elementwise add as before, issued once with a
        // repeat per row: 64 lanes of dst per repeat (stride 8 datablocks)
        // drawn from a 128-lane row (stride 16), with the upper half reached
        // by starting src1 one half-row in.
        const AscendC::BinaryRepeatParams foldParams{
            1, 1, 1, HALF_BLK, ROW_BLK, ROW_BLK};
        AscendC::Mul(sqLocal, qnLocal, qnLocal, rows * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(redLocal, sqLocal, sqLocal[D / 2], D / 2, reps,
                     foldParams);
        AscendC::PipeBarrier<PIPE_V>();
        // k reuses the squares buffer, so the fold above must have read it.
        AscendC::Mul(sqLocal, knLocal, knLocal, rows * D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(redLocal[C * (D / 2)], sqLocal, sqLocal[D / 2], D / 2,
                     reps, foldParams);
        AscendC::PipeBarrier<PIPE_V>();
        // One repeat per row, each reducing that row's 64 folded partials.
        // `mask` fixes the reduction tree, so a repeat here is the same tree
        // the two-repeat per-row form used; only the count changes. q's norms
        // land packed at `nsqLocal[0]`, k's at `nsqLocal[C]`.
        AscendC::WholeReduceSum<float>(nsqLocal, redLocal, 64, reps, 1, 1,
                                       HALF_BLK);
        AscendC::WholeReduceSum<float>(nsqLocal[C], redLocal[C * (D / 2)], 64,
                                       reps, 1, 1, HALF_BLK);

        // Vector result read by Scalar: a real event, not a PipeBarrier. One
        // round trip for the chunk, which is the point of the batching.
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_V_S_PROD);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_V_S_PROD);
        // sum(x*x) + 1e-6, then sqrt, matching upstream's epsilon placement.
        // The root stays on the Scalar unit: the Vector `Sqrt` is documented
        // at 1 ulp and measurably further from `tl.sqrt` here, roughly
        // doubling the surviving output mismatches.
        for (uint32_t r = 0; r < rows; ++r) {
            qsLocal.SetValue(r, sqrt(nsqLocal.GetValue(r) + 1e-6f));
            ksLocal.SetValue(r, sqrt(nsqLocal.GetValue(C + r) + 1e-6f));
        }
        // `Brcb` reads a whole 8-element repeat, so a short final chunk would
        // otherwise broadcast uninitialized UB. Those datablocks are never
        // divided by — the `Div`s below repeat only `rows` times — but a
        // finite filler keeps the buffer free of stray NaN.
        const uint8_t brcbReps = static_cast<uint8_t>((rows + 7) / 8);
        for (uint32_t r = rows; r < brcbReps * 8u; ++r) {
            qsLocal.SetValue(r, 1.0f);
            ksLocal.SetValue(r, 1.0f);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_S_V_PROD);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_S_V_PROD);

        // Upstream is `q = q / norm` then `q *= scale`, and `k = k / norm`.
        // Keep that order and keep both as real divisions: folding them into
        // `scale / qNorm`, or into a reciprocal multiply, reassociates and
        // costs a bf16 ulp on a few output elements.
        //
        // `Div` takes both operands as tensors, so each row's norm has to
        // occupy lanes. `Brcb` fills one datablock per row with it, and a
        // src1 block stride of 0 then makes a repeat reuse that single
        // datablock across all 64 of its lanes — so the divisor tensor is C
        // datablocks rather than C full rows, and no per-row `Duplicate` is
        // needed. src1 advances one datablock per repeat, dst and src0 a row.
        AscendC::Brcb(qbcLocal, qsLocal, brcbReps, {1, 8});
        AscendC::Brcb(kbcLocal, ksLocal, brcbReps, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        const AscendC::BinaryRepeatParams normParams{
            1, 1, 0, ROW_BLK, ROW_BLK, 1};
        AscendC::Div(qnLocal, qnLocal, qbcLocal, 64, reps, normParams);
        AscendC::Div(qnLocal[D / 2], qnLocal[D / 2], qbcLocal, 64, reps,
                     normParams);
        AscendC::Div(knLocal, knLocal, kbcLocal, 64, reps, normParams);
        AscendC::Div(knLocal[D / 2], knLocal[D / 2], kbcLocal, 64, reps,
                     normParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(qnLocal, qnLocal, scale_, rows * D);

        // Publish the chunk. The four stores must land before any consumer is
        // told the chunk exists, hence V_MTE3 before and MTE3_S after.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3_PROD);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3_PROD);
        // Vector is done with the bf16 staging; MTE2 may refill it next round.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2_PROD);

        AscendC::DataCopyParams outParams{
            static_cast<uint16_t>(rows),
            static_cast<uint16_t>(D * sizeof(float) / 32),
            0,
            static_cast<uint16_t>((H_TOTAL - 1) * D * sizeof(float) / 32)};
        AscendC::DataCopy(qnGm[base], qnLocal, outParams);
        AscendC::DataCopy(knGm[base], knLocal, outParams);
        AscendC::DataCopy(decayGm[base], gdLocal, outParams);
        // BETA is relaid as [H, Tpad] rather than upstream's [T, H]. In the
        // upstream layout one chunk of one head is `rows` fp32 strided by H,
        // and DMA writes whole 32 B datablocks, so storing head h's token t
        // would also clobber the other three heads at that token — written
        // concurrently by six other cores. Here a head owns a contiguous run
        // and `Tpad` is a multiple of CHUNK, so every chunk is its own 64 B
        // line and no two cores ever touch the same datablock. The padding
        // past `rows` is written with stale values and never read.
        AscendC::DataCopy(betaGm[static_cast<uint64_t>(head) * betaStride_ +
                                 chunk * C],
                          btLocal, C);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_MTE3_S_PROD);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_MTE3_S_PROD);

        // Notify this head's consumers, one bank each — IBWait consumes the
        // flag, so a single IBSet cannot serve all eight.
        #pragma unroll
        for (uint32_t lane = 0; lane < LANES; ++lane) {
            const uint32_t consumer = head * LANES + lane;
            AscendC::IBSet<true>(syncGm, syncLocal,
                                 static_cast<int32_t>(blockIdx),
                                 SyncEventId(consumer, chunk));
        }

        // MTE3 has to finish reading the fp32 outputs before the next
        // iteration's Vector work overwrites them.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V_PROD);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V_PROD);
    }

    // The last iteration's V_MTE2 has no matching Wait; drain it so the core
    // does not exit with a flag still set.
    if (numChunks > phase) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2_PROD);
    }
}

template<bool TRACK_STATE, uint32_t H_TOTAL>
__aicore__ inline void
KdaVarlenRecurrentPrefillH4<TRACK_STATE, H_TOTAL>::WaitChunk(
    uint32_t blockIdx, uint32_t head, uint64_t betaBase, uint32_t chunk)
{
    constexpr uint32_t PPH = config::PRODUCERS_PER_HEAD;
    constexpr uint32_t C = config::CHUNK;

    const uint32_t prodBase = config::NUM_CONSUMERS + head * PPH;
    AscendC::IBWait<true>(
        syncGm, syncLocal,
        static_cast<int32_t>(prodBase + chunk % PPH),
        SyncEventId(blockIdx, chunk));

    // Beta is the one producer output this core reads with the Scalar unit
    // rather than by DataCopy, and on this architecture the Scalar path goes
    // through a per-core DCache that is not coherent with another core's
    // writes. Invalidate the line now that the chunk has landed; one chunk is
    // exactly one 64 B line, so this is one instruction per 16 tokens, and the
    // Vector-side inputs (qn/kn/gdecay, all DMA) need nothing.
    AscendC::DataCacheCleanAndInvalid<
        float, AscendC::CacheLine::SINGLE_CACHE_LINE,
        AscendC::DcciDst::CACHELINE_OUT>(betaGm[betaBase + chunk * C]);
}

// ── Consumer ──────────────────────────────────────────────────────────
//
// `prepared_recurrent.cpp`'s token loop. The only structural change is the
// chunk-boundary IBWait, placed immediately before the prefetch of the token it
// guards so the double-buffered pipeline keeps running.
template<bool TRACK_STATE, uint32_t H_TOTAL>
__aicore__ inline void
KdaVarlenRecurrentPrefillH4<TRACK_STATE, H_TOTAL>::RunConsumer(
    uint32_t blockIdx)
{
    constexpr uint32_t NUM_V_TILES = config::NUM_V_TILES;
    constexpr uint32_t NUM_REQ = config::NUM_REQ;
    constexpr uint32_t STATE_STRIDE = config::STATE_STRIDE;
    constexpr uint32_t D = config::D;
    constexpr uint32_t BV = config::BV;
    constexpr uint32_t C = config::CHUNK;
    constexpr uint32_t PPH = config::PRODUCERS_PER_HEAD;

    const uint32_t iv   = blockIdx % NUM_V_TILES;
    const uint32_t req  = (blockIdx / NUM_V_TILES) % NUM_REQ;
    const uint32_t head = blockIdx / (NUM_V_TILES * NUM_REQ);

    const int32_t bos = startsGm.GetValue(req);
    const int32_t eos = startsGm.GetValue(req + 1);
    const int32_t slot = indicesGm.GetValue(req);

    // Beta rows for this head, [H, betaStride_] with betaStride_ a multiple of
    // the chunk size, so one chunk is one 64 B cache line.
    const uint64_t betaBase = static_cast<uint64_t>(head) * betaStride_;

    // pstate points at: state[slot, head, iv*BV, 0], where the state slot is
    // [V, K] with K contiguous. V is the major axis, so a tile of BV value
    // rows starts BV*D floats in, not BV.
    const uint64_t pstate =
        (static_cast<uint64_t>(slot) * H_TOTAL + head) * D * D
        + static_cast<uint64_t>(iv) * BV * D;

    if (slot > 0) {
        LoadStateTile(stateLocal, pstate);
    } else {
        AscendC::Duplicate(stateLocal, 0.0f, BV * STATE_STRIDE);
    }
    AscendC::PipeBarrier<PIPE_V>();

    int32_t trackSlot = 0;
    int32_t trackLen = 0;
    uint64_t ptrack = 0;

    if constexpr (TRACK_STATE) {
        trackSlot = trackIndicesGm.GetValue(req);
        trackLen = trackLensGm.GetValue(req);
        if (trackSlot > 0) {
            // Same GM layout as pstate: [slot, head, V, K].
            ptrack =
                (static_cast<uint64_t>(trackSlot) * H_TOTAL + head) * D * D
                + static_cast<uint64_t>(iv) * BV * D;
            if (trackLen == 0) {
                SaveTrackedState(ptrack);
            }
        }
    }

    AscendC::LocalTensor<float> tmpMulReduceLocal(
        AscendC::TPosition::VECCALC, UbLayout::MUL_REDUCE_ADDR, BV * D / 2);
    AscendC::LocalTensor<float> reduceLocal(
        AscendC::TPosition::VECCALC, UbLayout::REDUCE_ADDR, BV);
    AscendC::LocalTensor<float> tmpBVLocal(
        AscendC::TPosition::VECCALC, UbLayout::TMP_BV_ADDR, BV);
    AscendC::LocalTensor<float> tmpBrcbLocal(
        AscendC::TPosition::VECCALC, UbLayout::BRCB_ADDR, BV * 8);

    // Prefetch first token into buf[0], once its chunk is published.
    if (bos < eos) {
        WaitChunk(blockIdx, head, betaBase, 0);
        const uint64_t row0 = static_cast<uint64_t>(bos) * H_TOTAL + head;
        AscendC::DataCopy(qLocal[0], qnGm[row0 * D], D);
        AscendC::DataCopy(kLocal[0], knGm[row0 * D], D);
        AscendC::DataCopy(decayLocal[0], decayGm[row0 * D], D);
        AscendC::DataCopy(vLocal[0], vGm[row0 * D + iv * BV], BV);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V_QKVD_0);
    }

    AscendC::BinaryRepeatParams decay_params{
        1, 1, 1, STATE_STRIDE / 8, STATE_STRIDE / 8, 0};
    AscendC::BinaryRepeatParams mul_reduce_params{
        1, 1, 1, 8, STATE_STRIDE / 8, 0};
    AscendC::BinaryRepeatParams outer_params{
        1, 1, 0, STATE_STRIDE / 8, 0, 1};

    for (int32_t token = bos; token < eos; ++token)
    {
        bool pos2 = token > bos;
        bool posn_1 = token + 1 < eos;

        const uint64_t row = static_cast<uint64_t>(token) * H_TOTAL + head;
        const uint32_t cur = (token - bos) & 1;
        const uint32_t nxt = 1 - cur;
        int32_t cur_buf_e = cur == 0 ? EVENT_MTE2_V_QKVD_0 : EVENT_MTE2_V_QKVD_1;
        int32_t nxt_buf_e = nxt == 0 ? EVENT_MTE2_V_QKVD_0 : EVENT_MTE2_V_QKVD_1;

        if (pos2) {
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_V_S_BETA);
        }
        const float betaValue =
            betaGm.GetValue(betaBase + static_cast<uint64_t>(token - bos));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_S_V);

        // Ensure Vector is done with buf[nxt] (used in the previous iteration)
        // before MTE2 overwrites it with the prefetch.
        if (pos2) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2);
        }

        // Prefetch next token into buf[nxt]. If it opens a new chunk, wait for
        // that chunk first — this is the only place the consumer blocks on a
        // producer, and it costs one barrier per C tokens rather than per token.
        if (posn_1) {
            const uint32_t nextOffset = static_cast<uint32_t>(token + 1 - bos);
            if (nextOffset % C == 0) {
                WaitChunk(blockIdx, head, betaBase, nextOffset / C);
            }
            const uint64_t nextRow =
                static_cast<uint64_t>(token + 1) * H_TOTAL + head;
            AscendC::DataCopy(qLocal[nxt], qnGm[nextRow * D], D);
            AscendC::DataCopy(kLocal[nxt], knGm[nextRow * D], D);
            AscendC::DataCopy(decayLocal[nxt], decayGm[nextRow * D], D);
            AscendC::DataCopy(vLocal[nxt], vGm[nextRow * D + iv * BV], BV);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(nxt_buf_e);
        }

        // Wait for all current-buffer loads (issued before the loop or as
        // prefetch at the end of the previous iteration).
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(cur_buf_e);

        // state *= decay[None, :]
        AscendC::Mul(stateLocal, stateLocal, decayLocal[cur],
                     64, BV, decay_params);
        AscendC::Mul(stateLocal[64], stateLocal[64], decayLocal[cur][64],
                     64, BV, decay_params);
        AscendC::PipeBarrier<PIPE_V>();

        // v -= tl.sum(state * k[None, :], axis=1)
        AscendC::Mul(tmpMulReduceLocal, stateLocal, kLocal[cur],
                     64, BV, mul_reduce_params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::MulAddDst(tmpMulReduceLocal, stateLocal[64], kLocal[cur][64],
                           64, BV, mul_reduce_params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float>(reduceLocal, tmpMulReduceLocal,
                                       64, BV, 1, 1, 8);

        AscendC::Cast(tmpBVLocal, vLocal[cur], AscendC::RoundMode::CAST_NONE, BV);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(tmpBVLocal, tmpBVLocal, reduceLocal, BV);
        AscendC::PipeBarrier<PIPE_V>();

        // v *= beta
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_S_V);
        AscendC::Muls(tmpBVLocal, tmpBVLocal, betaValue, BV);
        if (posn_1) {
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_V_S_BETA);
        }

        // state += k[None, :] * v[:, None]
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(tmpBrcbLocal, tmpBVLocal, 2, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();

        // First 64 columns of all 16 rows, then the last 64.
        AscendC::MulAddDst(stateLocal, kLocal[cur], tmpBrcbLocal,
                           64, BV, outer_params);
        AscendC::MulAddDst(stateLocal[64], kLocal[cur][64], tmpBrcbLocal,
                           64, BV, outer_params);
        AscendC::PipeBarrier<PIPE_V>();

        // out = tl.sum(state * q[None, :], axis=1)
        AscendC::Mul(tmpMulReduceLocal, stateLocal, qLocal[cur],
                     64, BV, mul_reduce_params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::MulAddDst(tmpMulReduceLocal, stateLocal[64], qLocal[cur][64],
                           64, BV, mul_reduce_params);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WholeReduceSum<float>(reduceLocal, tmpMulReduceLocal,
                                       64, BV, 1, 1, 8);

        // Signal that Vector is done reading buf[cur]; the next iteration's
        // prefetch into buf[cur] waits on this.
        if (posn_1) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2);
        }

        if (pos2) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V);
        }
        if constexpr (TRACK_STATE) {
            if (trackSlot > 0 &&
                static_cast<int64_t>(token) - bos + 1 == trackLen) {
                SaveTrackedState(ptrack);
            }
        }

        AscendC::Cast(outLocal, reduceLocal, AscendC::RoundMode::CAST_RINT, BV);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3);
        AscendC::DataCopy(outGm[row * D + iv * BV], outLocal, BV);
        if (posn_1) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V);
        }
    }

    if (slot > 0) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3);
        StoreStateTile(stateLocal, pstate);
    }
}

template<bool TRACK_STATE, uint32_t H_TOTAL>
__aicore__ inline void
KdaVarlenRecurrentPrefillH4<TRACK_STATE, H_TOTAL>::Process()
{
    AscendC::InitSocState();

    const uint32_t blockIdx = AscendC::GetBlockIdx();
    if (blockIdx >= config::NUM_CONSUMERS) {
        RunProducer(blockIdx);
    } else {
        RunConsumer(blockIdx);
    }
}

#define DEFINE_KERNEL(NAME, TRACK, HTOT)                                \
extern "C" __global__ __vector__                                        \
void NAME(                                                              \
    GM_ADDR Q, GM_ADDR K, GM_ADDR V, GM_ADDR A, GM_ADDR B,              \
    GM_ADDR A_LOG, GM_ADDR DT_BIAS,                                     \
    GM_ADDR STATE, GM_ADDR INDICES, GM_ADDR STARTS, GM_ADDR OUT,        \
    GM_ADDR TRACK_INDICES, GM_ADDR TRACK_LENS,                          \
    GM_ADDR QN, GM_ADDR KN, GM_ADDR GDECAY, GM_ADDR BETA,               \
    GM_ADDR SYNC,                                                       \
    float scale, float lowerBound,                                      \
    uint32_t numReq, uint32_t numTokens                                 \
) {                                                                     \
    /* IBSet/IBWait reach GetTPipePtr()->FetchEventID() internally, so a */ \
    /* TPipe must exist on every core that synchronizes. No InitBuffer   */ \
    /* is called on it, so it reserves no UB and the absolute address    */ \
    /* map below is unaffected. Omitting it faults the waiting cores.    */ \
    AscendC::TPipe pipe;                                                \
    KdaVarlenRecurrentPrefillH4<TRACK, HTOT> op;                        \
    op.Init(Q, K, V, A, B, A_LOG, DT_BIAS,                              \
            STATE, INDICES, STARTS, OUT,                                \
            TRACK_INDICES, TRACK_LENS,                                  \
            QN, KN, GDECAY, BETA, SYNC,                                 \
            scale, lowerBound, numReq, numTokens);                      \
    op.Process();                                                       \
}

// Tracking is prefill-only, so both variants live here.
DEFINE_KERNEL(kda_varlen_recurrent_kernel_h4,       false, 4)
DEFINE_KERNEL(kda_varlen_recurrent_kernel_track_h4, true,  4)

#undef DEFINE_KERNEL

// ── Host-side launcher ─────────────────────────────────────────────────

#ifndef __CCE_KT_TEST__

void LaunchKdaPrefillH4(const KdaLaunchArgs &args, void *stream)
{
    // 40 blocks: 32 consumers (8 value tiles x 4 heads) plus 8 producers, one
    // wave on a 910B's vector cores. See `kda_prefill` in the header.
    constexpr uint32_t blockDim = kda_prefill::kBlockDim;
    auto acl_stream = static_cast<aclrtStream>(stream);

#define LAUNCH(KERNEL)                                                  \
    KERNEL<<<blockDim, nullptr, acl_stream>>>(                          \
        args.q, args.k, args.v, args.a, args.b,                         \
        args.A_log, args.dt_bias,                                       \
        args.state, args.indices, args.starts, args.out,                \
        args.track_indices, args.track_lens,                            \
        args.qn, args.kn, args.gdecay, args.beta, args.sync,            \
        args.scale, args.lower_bound, args.num_req, args.num_tokens)

    if (args.track_state) {
        LAUNCH(kda_varlen_recurrent_kernel_track_h4);
    } else {
        LAUNCH(kda_varlen_recurrent_kernel_h4);
    }

#undef LAUNCH
}

#endif
