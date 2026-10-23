// GLM-5.3 KDA varlen recurrence — prefill kernel, H=8 (TP8).
//
// One packed sequence (numReq==1) and a long serial token loop. See
// `kda_varlen_recurrent.h` for the shared contract and the reference semantics;
// the host file validates everything before this launcher is reached.
//
// ── Why this is H=4 run twice ─────────────────────────────────────────
//
// Same grid as H=4 — 32 consumers plus 8 producers, one scheduler wave on a
// 910B's 40 vector cores — but H=8 has 64 (head, value tile) work items where
// H=4 has 32. So each consumer runs the token loop twice, once per head:
//
//     round 0: heads 0..3      round 1: heads 4..7
//
// with the same (headSlot, value tile) decomposition inside each round. The
// alternative was to interleave two value tiles inside one token loop, which
// is what the earlier skeleton in this file proposed on the grounds that the
// loop is latency-bound. Measured, it is not: the consumer's `aiv_vec_ratio`
// is 0.93, so only ~7% of the vector pipe is idle and a second tile has
// nowhere to hide. Two serialized rounds of `prepared_recurrent` confirm the
// same thing end to end, at exactly 2.00x the H=4 time.
//
// Serial rounds also cost nothing in UB. Round 0 stores its state tile before
// round 1 loads over it, so the one 8 KB buffer serves both and the H=4 UB map
// carries over unchanged — interleaving would have needed a second tile.
//
// What two rounds does leave on the table is the round boundary: a state
// store, a state load, and a cold prefetch with no outstanding loads to cover
// it. So the next round's first token is prefetched *before* the state store
// that ends the current round, which is the one place this file's token loop
// differs from H=4's. Everything inside the loop is identical, deliberately:
// the recurrence is bit-exact against upstream at H=4 and the arithmetic must
// not move.
//
// Producers need no restructuring. At H=4 two producers feed each of 4 heads;
// here the same two feed that head slot in round 0 and again in round 1, so the
// (producer, consumer) pairing and its notification banks are unchanged. A
// producer walks its head's chunks for round 0, then for round 1, in the order
// its consumers consume them, so one ring of `eventID`s still serves both.

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

// Consumer, lifted from `prepared_recurrent.cpp` via the H=4 kernel.
constexpr int32_t EVENT_MTE2_V_S = 0;
constexpr int32_t EVENT_MTE2_V_QKVD_0 = 1;
constexpr int32_t EVENT_MTE2_V_QKVD_1 = 2;
constexpr int32_t EVENT_V_S_BETA = 0;
constexpr int32_t EVENT_S_V = 2;
constexpr int32_t EVENT_V_MTE2 = 3;
constexpr int32_t EVENT_MTE3_V = 4;
constexpr int32_t EVENT_MTE3_V_TRACK = 0;
constexpr int32_t EVENT_V_MTE3 = 5;
// Round boundary only: round 0's state store (MTE3) against round 1's state
// load (MTE2) into the same buffer. Its own pipe pair, so id 0 is free.
constexpr int32_t EVENT_MTE3_MTE2 = 0;

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
// H_TOTAL       heads per state slot; 8 for prefill at TP8.
template<bool TRACK_STATE, uint32_t H_TOTAL>
class KdaVarlenRecurrentPrefillH8 {
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
        GM_ADDR BETA,                                  // fp32 [H, Tpad]
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

        // Heads covered by the 32 consumers in one pass, and the number of
        // passes needed to cover H_TOTAL.
        static constexpr uint32_t HEADS_PER_ROUND = kda_prefill::kHeadsPerRound;
        static constexpr uint32_t ROUNDS = H_TOTAL / HEADS_PER_ROUND;  // 2

        // Consumer lanes fed by one producer, and producers per head slot.
        static constexpr uint32_t LANES_PER_HEAD = NUM_V_TILES;            // 8
        static constexpr uint32_t PRODUCERS_PER_HEAD =
            NUM_PRODUCERS / HEADS_PER_ROUND;                               // 2

        static_assert(NUM_CONSUMERS == NUM_V_TILES * NUM_REQ * HEADS_PER_ROUND,
                      "32 consumers = 8 value tiles x 4 head slots");
        static_assert(H_TOTAL % HEADS_PER_ROUND == 0,
                      "every round must cover the same number of heads");
        static_assert(ROUNDS <= kda_prefill::kMaxRounds,
                      "more rounds than the shared grid constants allow");
        static_assert(PRODUCERS_PER_HEAD >= 1,
                      "each head slot needs at least one producer");
    };

    // UB map, identical to the H=4 kernel's. The consumer half is
    // `prepared_recurrent`'s map at the same addresses, deliberately: the token
    // loop's repeat strides are derived from these offsets. The producer half
    // and the sync scratch go in the 57 KB hole that map leaves between the
    // state tile (8 KB at 0x0) and 0x10100.
    //
    // Two rounds do not widen this. Round 0's state tile is stored to GM before
    // round 1 loads into the same buffer, and every other buffer here is
    // per-token scratch that neither round outlives.
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
        // per-head constants, reloaded once per round: the dt_bias row.
        static constexpr uint32_t P_BIAS_ADDR = P_BT_ADDR + 64;
        // scratch: one row of squares, and the folded half-sums of q and k,
        // which take a full row between them.
        static constexpr uint32_t P_TMP_ADDR = P_BIAS_ADDR + ROW_F32;
        static constexpr uint32_t P_RED_ADDR = P_TMP_ADDR + ROW_F32;
        // One repeat of lower_bound, the gate division's numerator. `Div`
        // takes both operands as tensors, so the scalar has to be broadcast;
        // a src0 repeat stride of 0 then reuses this one repeat for all rows.
        static constexpr uint32_t P_LB_ADDR = P_RED_ADDR + ROW_F32;
        // Scalar/Vector exchange slot. This architecture has no scalar `exp`,
        // so exp(A_log) is computed on the Vector unit and read back here.
        static constexpr uint32_t P_SCR_ADDR = P_LB_ADDR + 256;
        // One row of the broadcast q/k norm. `Div` takes both operands as
        // tensors, and the division is kept rather than folded into a
        // reciprocal multiply so the result matches upstream bit for bit.
        static constexpr uint32_t P_NORM_ADDR = P_SCR_ADDR + 256;
        static constexpr uint32_t P_END = P_NORM_ADDR + ROW_F32;

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

    // One head's token loop. `preloaded` says buf[0] already holds token `bos`
    // (prefetched by the previous round, which also took its chunk-0 `IBWait`);
    // `prefetchHead` is the head whose token `bos` to pull in as this round
    // winds down, or H_TOTAL for the last round.
    __aicore__ inline void RunConsumerRound(
        uint32_t blockIdx, uint32_t iv, uint32_t head, uint32_t req,
        bool preloaded, uint32_t prefetchHead);

    // Issue the loads for one token into buffer `buf`, without waiting on them.
    __aicore__ inline void PrefetchToken(
        uint32_t buf, uint32_t iv, uint32_t head, int32_t token);

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
    //
    // Rounds share the ring. A producer walks round 0's chunks then round 1's,
    // in the order its consumers consume them, so the pair stays FIFO and a
    // slot is always cleared by the matching `IBWait` before it is reused.
    static __aicore__ inline int32_t SyncEventId(uint32_t consumerBlock,
                                                 uint32_t chunk)
    {
        const uint32_t slot = (chunk / config::PRODUCERS_PER_HEAD) %
                              config::RING;
        return static_cast<int32_t>(consumerBlock * config::RING + slot);
    }
};

template<bool TRACK_STATE, uint32_t H_TOTAL>
__aicore__ inline void KdaVarlenRecurrentPrefillH8<TRACK_STATE, H_TOTAL>::Init(
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

template<bool TRACK_STATE, uint32_t H_TOTAL>
__aicore__ inline void
KdaVarlenRecurrentPrefillH8<TRACK_STATE, H_TOTAL>::LoadStateTile(
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
KdaVarlenRecurrentPrefillH8<TRACK_STATE, H_TOTAL>::StoreStateTile(
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
KdaVarlenRecurrentPrefillH8<TRACK_STATE, H_TOTAL>::SaveTrackedState(
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
//
// The H=4 body, wrapped in a loop over rounds: this producer serves its head
// slot for heads `headSlot`, then `HEADS_PER_ROUND + headSlot`. The per-head
// constants (dt_bias, exp(A_log)) are reloaded at the top of each round.
template<bool TRACK_STATE, uint32_t H_TOTAL>
__aicore__ inline void
KdaVarlenRecurrentPrefillH8<TRACK_STATE, H_TOTAL>::RunProducer(
    uint32_t blockIdx)
{
    constexpr uint32_t D = config::D;
    constexpr uint32_t C = config::CHUNK;
    constexpr uint32_t PPH = config::PRODUCERS_PER_HEAD;
    constexpr uint32_t LANES = config::LANES_PER_HEAD;

    const uint32_t prod = blockIdx - config::NUM_CONSUMERS;  // 0..NUM_PRODUCERS-1
    const uint32_t headSlot = prod / PPH;
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
    AscendC::LocalTensor<float> tmpLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_TMP_ADDR, D);
    AscendC::LocalTensor<float> redLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_RED_ADDR, D);
    AscendC::LocalTensor<float> lbLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_LB_ADDR, 64);
    AscendC::LocalTensor<float> scrLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_SCR_ADDR, 64);
    AscendC::LocalTensor<float> normLocal(
        AscendC::TPosition::VECCALC, UbLayout::P_NORM_ADDR, D);

    // The gate's division numerator, broadcast once — round-independent, so it
    // stays outside the round loop. `Div` takes both operands as tensors on
    // this architecture and `Divs` is unsupported, so a scalar numerator has to
    // be materialized. `Reciprocal` would avoid this but is documented as
    // missing the double-ten-thousandth bar for float, and the gate it would
    // feed multiplies the state once per token, so the error would compound
    // across the sequence.
    AscendC::Duplicate(lbLocal, lowerBound_, 64);

    for (uint32_t round = 0; round < config::ROUNDS; ++round) {
        const uint32_t head = round * config::HEADS_PER_ROUND + headSlot;

        // ── Per-head constants, computed once per round ──
        //
        // `decay = exp(A_log[head])` is a scalar, but this architecture has no
        // scalar `exp`: the only `exp` is the Vector instruction. So A_log goes
        // out to UB on the Scalar pipe, through `Exp`, and back — two real
        // hardware events, since `PipeBarrier<PIPE_V>` does not order Vector
        // against Scalar.
        AscendC::DataCopy(biasLocal, dtBiasGm[head * D], D);
        // `Exp` works a whole 8-float datablock at a time, so all eight lanes
        // are written rather than just the one that is read back — UB arrives
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

        // The dt_bias row is read by Vector in the loop below.
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V_PROD);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V_PROD);

        for (uint32_t chunk = phase; chunk < numChunks; chunk += PPH) {
            const uint32_t tok0 = static_cast<uint32_t>(bos) + chunk * C;
            const uint32_t rows =
                (tok0 + C <= static_cast<uint32_t>(eos)) ? C
                                                         : static_cast<uint32_t>(eos) - tok0;

            // Rows are strided by H in GM, so one DataCopy per tensor per chunk
            // with a row gap of (H-1) rows. blockLen/gaps are in 32 B
            // datablocks: a 128-lane bf16 row is 256 B = 8 blocks, and the gap
            // skips H-1 rows.
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

            // b is [1,T,H]: one fp32 per row once converted, strided by H in
            // GM. One element per 32 B block is below the DMA's useful
            // granularity, so Scalar gathers the chunk and hands it to Vector
            // in one go.
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
            // otherwise carry uninitialized UB through Exp. The padding is
            // written to GM and never read, but it must not be inf or NaN.
            AscendC::Muls(brawLocal, brawLocal, -1.0f, C);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(brawLocal, brawLocal, C);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(brawLocal, brawLocal, 1.0f, C);

            // The reciprocal runs on Scalar, one token at a time, rather than
            // as a Vector `Div`. `Div` is documented at 1 ulp here: its 0-ulp
            // algorithms (DivAlgo::PRECISION_0ULP_*) are 950-series only, so
            // this hardware gets DivAlgo::INTRINSIC. Upstream's division is
            // IEEE-exact, and beta scales the state update on every step of the
            // recurrence, so a last-bit difference compounds along the
            // sequence. One division per token is cheap next to the 128-lane
            // work around it.
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
            // repeat covers 64, so the add runs twice with a repeat stride of
            // one row (16 datablocks) and a src1 stride of 0 to hold bias in
            // place.
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
            // gate = lower_bound / (1 + exp(-decay*x)), with the numerator held
            // in place by a src0 repeat stride of 0 so one repeat of `lbLocal`
            // covers every row. Two passes, 64 lanes each, as above.
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

            // ── q, k: L2 normalize row by row; q also takes `scale` ──
            //
            // The norm is a per-row scalar, so each row costs a Vector→Scalar
            // round trip. `sqrt` exists on the Scalar unit, so the root is
            // taken there and does not need a second trip back through Vector.
            for (uint32_t r = 0; r < rows; ++r) {
                auto qRow = qnLocal[r * D];
                auto kRow = knLocal[r * D];

                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(qRow, qbLocal[r * D], AscendC::RoundMode::CAST_NONE, D);
                AscendC::Cast(kRow, kbLocal[r * D], AscendC::RoundMode::CAST_NONE, D);
                AscendC::PipeBarrier<PIPE_V>();
                // sum(x*x) over 128 lanes, in upstream's summation order.
                //
                // `tl.sum` over 128 fp32 lanes is bit-identical to
                // `torch.sum(-1)` and to neither a 2x64 nor a 4x32
                // decomposition. Of thirteen candidate trees measured against
                // it, exactly one matched on every row: add the two 64-lane
                // halves elementwise, then reduce the 64 partials as a binary
                // tree over adjacent lanes. A 2x64 `WholeReduceSum` pair
                // differs on ~28% of rows. The norm divides q and k on every
                // step of a recurrence, so a last-bit difference here compounds
                // along the sequence; the order has to match.
                //
                // q folds into the low half of `redLocal` and k into the high
                // half, so one two-repeat `WholeReduceSum` finishes both and
                // the norms land in lanes 0 and 1. Source and destination must
                // not overlap, hence the write back into `tmpLocal`.
                AscendC::Mul(tmpLocal, qRow, qRow, D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(redLocal, tmpLocal, tmpLocal[D / 2], D / 2);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(tmpLocal, kRow, kRow, D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(redLocal[D / 2], tmpLocal, tmpLocal[D / 2], D / 2);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum<float>(tmpLocal, redLocal, 64, 2, 1, 1, 8);

                // Vector result read by Scalar: a real event, not a PipeBarrier.
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_V_S_PROD);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_V_S_PROD);
                // sum(x*x) + 1e-6, then sqrt, matching upstream's epsilon
                // placement. The root stays on the Scalar unit: the Vector
                // `Sqrt` is documented at 1 ulp and measurably further from
                // `tl.sqrt` here, roughly doubling the surviving output
                // mismatches.
                const float qNorm = sqrt(tmpLocal.GetValue(0) + 1e-6f);
                const float kNorm = sqrt(tmpLocal.GetValue(1) + 1e-6f);
                AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_S_V_PROD);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_S_V_PROD);
                // Upstream is `q = q / norm` then `q *= scale`, and
                // `k = k / norm`. Keep that order and keep both as real
                // divisions: folding them into `scale / qNorm`, or into a
                // reciprocal multiply, reassociates and costs a bf16 ulp on a
                // few output elements.
                AscendC::Duplicate(normLocal, qNorm, D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Div(qRow, qRow, normLocal, D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(qRow, qRow, scale_, D);
                AscendC::Duplicate(normLocal, kNorm, D);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Div(kRow, kRow, normLocal, D);
            }

            // Publish the chunk. The four stores must land before any consumer
            // is told the chunk exists, hence V_MTE3 before and MTE3_S after.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3_PROD);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3_PROD);
            // Vector is done with the bf16 staging; MTE2 may refill it next
            // round.
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
            // upstream layout one chunk of one head is `rows` fp32 strided by
            // H, and DMA writes whole 32 B datablocks, so storing head h's
            // token t would also clobber the other heads at that token —
            // written concurrently by other cores. Here a head owns a
            // contiguous run and `Tpad` is a multiple of CHUNK, so every chunk
            // is its own 64 B line and no two cores ever touch the same
            // datablock. The padding past `rows` is written with stale values
            // and never read.
            AscendC::DataCopy(betaGm[static_cast<uint64_t>(head) * betaStride_ +
                                     chunk * C],
                              btLocal, C);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_MTE3_S_PROD);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_MTE3_S_PROD);

            // Notify this head slot's consumers, one bank each — IBWait
            // consumes the flag, so a single IBSet cannot serve all eight. The
            // consumer lanes are the same in both rounds, since a lane runs
            // round 0's head then round 1's.
            #pragma unroll
            for (uint32_t lane = 0; lane < LANES; ++lane) {
                const uint32_t consumer = headSlot * LANES + lane;
                AscendC::IBSet<true>(syncGm, syncLocal,
                                     static_cast<int32_t>(blockIdx),
                                     SyncEventId(consumer, chunk));
            }

            // MTE3 has to finish reading the fp32 outputs before the next
            // iteration's Vector work overwrites them.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V_PROD);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V_PROD);
        }

        // The last iteration's V_MTE2 has no matching Wait; drain it so the
        // next round — or the core's exit — does not see a flag still set.
        if (numChunks > phase) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2_PROD);
        }
    }
}

template<bool TRACK_STATE, uint32_t H_TOTAL>
__aicore__ inline void
KdaVarlenRecurrentPrefillH8<TRACK_STATE, H_TOTAL>::WaitChunk(
    uint32_t blockIdx, uint32_t head, uint64_t betaBase, uint32_t chunk)
{
    constexpr uint32_t PPH = config::PRODUCERS_PER_HEAD;
    constexpr uint32_t C = config::CHUNK;

    // The producer cores are indexed by head *slot*, not by head: a slot's two
    // producers serve it in both rounds.
    const uint32_t headSlot = head % config::HEADS_PER_ROUND;
    const uint32_t prodBase = config::NUM_CONSUMERS + headSlot * PPH;
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

template<bool TRACK_STATE, uint32_t H_TOTAL>
__aicore__ inline void
KdaVarlenRecurrentPrefillH8<TRACK_STATE, H_TOTAL>::PrefetchToken(
    uint32_t buf, uint32_t iv, uint32_t head, int32_t token)
{
    constexpr uint32_t D = config::D;
    constexpr uint32_t BV = config::BV;

    const uint64_t row = static_cast<uint64_t>(token) * H_TOTAL + head;
    AscendC::DataCopy(qLocal[buf], qnGm[row * D], D);
    AscendC::DataCopy(kLocal[buf], knGm[row * D], D);
    AscendC::DataCopy(decayLocal[buf], decayGm[row * D], D);
    AscendC::DataCopy(vLocal[buf], vGm[row * D + iv * BV], BV);
}

// ── Consumer ──────────────────────────────────────────────────────────
//
// `prepared_recurrent.cpp`'s token loop, via the H=4 kernel. Two structural
// changes from that file, both about the chunk notifications and the round
// boundary; the arithmetic is untouched, because it is bit-exact at H=4.
//
//   * the chunk-boundary IBWait sits immediately before the prefetch of the
//     token it guards, so the double-buffered pipeline keeps running;
//   * the last token's prefetch slot, which would otherwise go unused, pulls
//     in the *next round's* first token. That load then overlaps this round's
//     state store and the next round's state load, which is the whole of the
//     round-boundary bubble.
template<bool TRACK_STATE, uint32_t H_TOTAL>
__aicore__ inline void
KdaVarlenRecurrentPrefillH8<TRACK_STATE, H_TOTAL>::RunConsumerRound(
    uint32_t blockIdx, uint32_t iv, uint32_t head, uint32_t req,
    bool preloaded, uint32_t prefetchHead)
{
    constexpr uint32_t STATE_STRIDE = config::STATE_STRIDE;
    constexpr uint32_t D = config::D;
    constexpr uint32_t BV = config::BV;
    constexpr uint32_t C = config::CHUNK;

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

    // Prefetch the first token into buf[0], once its chunk is published —
    // unless the previous round already did both, in which case its MTE2_V
    // flag is still outstanding and the loop below consumes it.
    if (bos < eos && !preloaded) {
        WaitChunk(blockIdx, head, betaBase, 0);
        PrefetchToken(0, iv, head, bos);
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
            PrefetchToken(nxt, iv, head, token + 1);
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

    // Pull the next round's first token in before this round's state store, so
    // the load runs under the store and the next round's own state load. The
    // token loop drains every flag it sets, so buf[0] is free except against
    // the Vector work of the final token — hence the explicit V_MTE2 pair.
    // The next round starts at token `bos` with `cur == 0`, so buf[0] is the
    // buffer it will read.
    const bool lastRound = prefetchHead >= H_TOTAL;
    if (!lastRound && bos < eos) {
        const uint64_t nextBetaBase =
            static_cast<uint64_t>(prefetchHead) * betaStride_;
        WaitChunk(blockIdx, prefetchHead, nextBetaBase, 0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_V_MTE2);
        PrefetchToken(0, iv, prefetchHead, bos);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_MTE2_V_QKVD_0);
    }

    if (slot > 0) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_V_MTE3);
        StoreStateTile(stateLocal, pstate);
        // The next round loads its own tile into this same buffer, and that
        // load is MTE2 while the store above is MTE3 still reading it — a WAR
        // hazard between two move pipes, so MTE3_MTE2 rather than the MTE3_V
        // the tracking checkpoint uses. Only on the `slot > 0` path, which is
        // exactly the path where the next round's load is a DataCopy; at
        // slot 0 it is a Vector `Duplicate` and there is no store to race.
        if (!lastRound) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_MTE3_MTE2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_MTE3_MTE2);
        }
    }

    // The final token's output store is still in flight: the loop only arms
    // MTE3_V when another token follows, so nothing orders it against the next
    // round's first `Cast` into `outLocal`. Inside a round the first token is
    // the launch's first write and there is no store to race; across a round
    // boundary there is. Drain it here rather than widening the loop's
    // condition, which would cost a barrier on every token to fix one.
    if (!lastRound && bos < eos) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_MTE3_V);
    }
}

template<bool TRACK_STATE, uint32_t H_TOTAL>
__aicore__ inline void
KdaVarlenRecurrentPrefillH8<TRACK_STATE, H_TOTAL>::RunConsumer(
    uint32_t blockIdx)
{
    constexpr uint32_t NUM_V_TILES = config::NUM_V_TILES;
    constexpr uint32_t NUM_REQ = config::NUM_REQ;

    const uint32_t iv       = blockIdx % NUM_V_TILES;
    const uint32_t req      = (blockIdx / NUM_V_TILES) % NUM_REQ;
    const uint32_t headSlot = blockIdx / (NUM_V_TILES * NUM_REQ);

    // One head per round, the slot held fixed: this core runs head `headSlot`,
    // then `HEADS_PER_ROUND + headSlot`. Keeping the slot fixed is what lets
    // the producers and their notification banks stay exactly as they are at
    // H=4 — a lane's producer pair does not change between rounds.
    for (uint32_t round = 0; round < config::ROUNDS; ++round) {
        const uint32_t head = round * config::HEADS_PER_ROUND + headSlot;
        const bool last = round + 1 == config::ROUNDS;
        RunConsumerRound(
            blockIdx, iv, head, req,
            /*preloaded=*/round > 0,
            /*prefetchHead=*/last ? H_TOTAL
                                  : head + config::HEADS_PER_ROUND);
    }
}

template<bool TRACK_STATE, uint32_t H_TOTAL>
__aicore__ inline void
KdaVarlenRecurrentPrefillH8<TRACK_STATE, H_TOTAL>::Process()
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
    KdaVarlenRecurrentPrefillH8<TRACK, HTOT> op;                        \
    op.Init(Q, K, V, A, B, A_LOG, DT_BIAS,                              \
            STATE, INDICES, STARTS, OUT,                                \
            TRACK_INDICES, TRACK_LENS,                                  \
            QN, KN, GDECAY, BETA, SYNC,                                 \
            scale, lowerBound, numReq, numTokens);                      \
    op.Process();                                                       \
}

// Tracking is prefill-only, so both variants live here.
DEFINE_KERNEL(kda_varlen_recurrent_kernel_h8,       false, 8)
DEFINE_KERNEL(kda_varlen_recurrent_kernel_track_h8, true,  8)

#undef DEFINE_KERNEL

// ── Host-side launcher ─────────────────────────────────────────────────

#ifndef __CCE_KT_TEST__

void LaunchKdaPrefillH8(const KdaLaunchArgs &args, void *stream)
{
    // The same 40 blocks as H=4: 32 consumers (8 value tiles x 4 head slots)
    // plus 8 producers, one wave on a 910B's vector cores. Each consumer runs
    // the token loop twice, once per head. See `kda_prefill` in the header.
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
        LAUNCH(kda_varlen_recurrent_kernel_track_h8);
    } else {
        LAUNCH(kda_varlen_recurrent_kernel_h8);
    }

#undef LAUNCH
}

#endif
