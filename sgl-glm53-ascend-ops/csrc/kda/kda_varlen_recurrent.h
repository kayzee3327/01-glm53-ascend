// GLM-5.3 KDA varlen recurrence — seam between the host op and the kernels.
//
// The operator is split across translation units because its five kernels have
// nothing in common but the math: prefill runs one long serial token loop over a
// single packed sequence, while decode runs one token for each of many requests,
// and each decode H wants its own tiling and block mapping.
//
//   kda_varlen_recurrent.cpp               host: validation, dispatch, registration
//   kda_varlen_recurrent_prefill_h4.cpp    H=4  (TP16), plain + track
//   kda_varlen_recurrent_prefill_h8.cpp    H=8  (TP8),  plain + track
//   kda_varlen_recurrent_decode_h16.cpp    H=16 (TP4)
//   kda_varlen_recurrent_decode_h32.cpp    H=32 (TP2)
//   kda_varlen_recurrent_decode_h64.cpp    H=64 (TP1)
//
// The two prefill files are separate because H=4 and H=8 differ in how many
// times a consumer runs the token loop, not in the grid: both launch the same
// 40 blocks (32 consumers + 8 producers, `kda_prefill` below). At H=4 the 32
// (head, value tile) work items map one-to-one onto the consumers; at H=8 there
// are 64, so each consumer runs the loop twice, once per head. The two token
// loops differ enough in bookkeeping — a state tile swap and a prefetch that
// crosses the round boundary — to keep them in separate files.
//
// A `__global__` kernel can only be launched from the TU that defines it, so
// each kernel file exposes one host-callable launcher declared below. By the
// time a launcher is called every precondition it relies on has been checked,
// so launchers do not re-validate.
//
// Division of labour with the caller: sglang's `glm_kda_varlen_recurrent_npu`
// owns the upstream contract — the shape/metadata `ValueError`s, the output
// allocation, the `scale` default, `.contiguous()` — and all of it runs before
// the op is reached, so the op does not repeat it. The op owns what the kernels
// index against and cannot detect themselves (dtypes, feature widths, the state
// layout), plus the kernel choice. Shapes the launcher set does not cover are
// rejected here rather than worked around; the caller keeps its own fallback
// for those.
//
// This header is included by both the host TU (which also includes torch) and
// the device TUs, so it must stay free of torch and Ascend C headers.

#pragma once

#include <cstdint>

// Head feature width, K == V == D. Shared by the device tiling and the host
// validation, so there is one definition of the only supported shape.
constexpr int64_t config_D = 128;

// State layout. `state` is indexed [slot, head, k, v] but laid out with strides
// (H*K*V, K*V, 1, V): K is the contiguous axis and V strides by 128, so in
// memory a slot is V-major. Read the strides, not the index order — taking K
// for the major axis puts every state store in the transpose of its slot.
// This is the only layout that
// reaches the kernel: the backend normalizes both the prefill pool and the
// already-transposed speculative pool to it (ascend_kda_backend.py,
// `speculative_state_view`), and the generic Triton kernel is numerically wrong
// in the transposed one. Slot 0 is the padding row, never read or written.
//
// Reference semantics (per head, per value tile), from the Triton kernel:
//
//     decay = exp(A_log)
//     g     = lower_bound * sigmoid(decay * (a + dt_bias))
//     beta  = sigmoid(b)
//     q     = q / sqrt(sum(q*q) + 1e-6) * scale
//     k     = k / sqrt(sum(k*k) + 1e-6)
//     h    *= exp(g)                      [broadcast over V]
//     v    -= sum(h * k, axis=K)
//     v    *= beta
//     h    += outer(v, k)
//     out   = sum(h * q, axis=K)

// Grid shape of the prefill kernels, shared because the host has to size the
// notification workspace that the device indexes into. Getting these out of
// step is not a compile error, it is a hang or a silent out-of-bounds write, so
// they live here rather than being written twice.
//
// Both prefill kernels are a producer/consumer pair in one grid: consumers run
// the serial recurrence, producers prepare the per-token inputs ahead of them.
// H=4 and H=8 use the same split; they differ only in how many rounds of
// `kHeadsPerRound` heads a consumer runs.
namespace kda_prefill {

// Vector cores on the deployment target, Atlas A2 / 910B. The whole grid is
// sized to land on exactly this number, so it is the budget the split below
// spends rather than an upper bound it happens to respect.
//
// The `IBSet`/`IBWait` contract requires the logical grid to fit the physical
// core count: with more blocks than cores the scheduler runs them in waves, and
// a wave of blocked consumers can exclude the producers they are waiting on,
// which hangs the kernel rather than slowing it down. So this is a correctness
// constant, not a tuning one.
//
// Note for anyone benchmarking: a 910C has 48, so this grid leaves 8 cores idle
// there. Do not raise it to fill them without re-checking the deadlock
// argument above on the machine that will actually run the model.
constexpr uint32_t kNumVectorCores = 40;

// 32 consumers = 4 heads x 8 value tiles at BV=16, the `prepared_recurrent`
// shape, which keeps that kernel's UB map and token loop reusable verbatim.
// The 8 cores left over become producers, 2 per head, splitting chunks
// even/odd. The consumer count is fixed by the tiling, so on a 910B the
// producer count is whatever remains — buying more producers means giving up a
// value tile, not widening the grid.
constexpr uint32_t kNumConsumers = 32;
constexpr uint32_t kNumProducers = kNumVectorCores - kNumConsumers;  // 8
constexpr uint32_t kBlockDim = kNumConsumers + kNumProducers;        // 40
static_assert(kBlockDim <= kNumVectorCores,
              "the prefill grid must fit one scheduler wave or it can "
              "deadlock on the inter-core notifications");
static_assert(kNumProducers > 0 && kNumConsumers % kNumProducers == 0,
              "every producer must own the same number of consumer lanes");

// Heads a single consumer core runs the token loop for, one after another.
// 32 consumers cover 4 heads x 8 value tiles per round, so H=4 is one round and
// H=8 is two.
//
// Two rounds rather than two value tiles interleaved inside one token loop,
// which is what an earlier design here proposed. Measured on a 910B4-1, the
// consumer's `aiv_vec_ratio` is 0.93: the token loop is vector-throughput
// bound, not latency bound, so there is no idle pipe for a second tile to hide
// in. `prepared_recurrent` run as two half-head launches confirms it end to
// end at exactly 2.00x the H=4 time. Interleaving would also cost a second
// 8 KB state tile in UB; serial rounds reuse the one buffer, because round 0
// stores its tile before round 1 loads over it.
//
// The round boundary is not free — it is a state store, a state load and a
// cold prefetch — so the kernel pulls the next round's first prefetch across
// it. That bubble is the only thing two rounds leaves on the table.
constexpr uint32_t kMaxRounds = 2;
constexpr uint32_t kHeadsPerRound = kNumConsumers / (128 / 16);  // 4

// Tokens prepared per notification. Trades sync frequency against how far a
// producer may run ahead; measured at ~61 consumer cycles per `IBWait` against
// ~3000 cycles/token of real work, so this is not a tight constraint.
constexpr uint32_t kChunkTokens = 16;

// Notification banks per (producer, consumer) pair. `IBSet` blocks while the
// flag is still set and `IBWait` clears it, so one bank is a depth-1 semaphore
// and a producer could not get ahead of its consumers at all. A small ring of
// distinct `eventID`s gives run-ahead in bounded memory. Must be a power of two
// (the device indexes with a mask).
constexpr uint32_t kSyncRing = 4;
static_assert((kSyncRing & (kSyncRing - 1)) == 0, "ring size must be 2^n");

// Number of int32 elements the `sync` workspace needs, per the `IBSet` doc's
// formula `coreNum * 32B * eventID_max + blockIdx_max * 32B + 32B`, in units of
// int32 (8 per 32 B block). `eventID` runs over consumer lane x ring slot and
// `blockIdx` over the whole grid. The core count is `kBlockDim` because this
// kernel is AIV-only (in MIX mode the API doubles it).
constexpr uint32_t kSyncEventIds = kNumConsumers * kSyncRing;
constexpr uint32_t kSyncInt32Count =
    kBlockDim * 8 * kSyncEventIds + kBlockDim * 8 + 8;

}  // namespace kda_prefill

// Device pointers and scalars for one launch. Raw `uint8_t*` rather than
// typed pointers because that is what the Ascend C `GM_ADDR` kernel parameters
// take; the comments carry the real element types.
//
// `out` is allocated by the caller and passed in, not allocated by the op. The
// kernels write every element they are responsible for, but a launcher covers
// only its own blocks, so the buffer must arrive zeroed.
struct KdaLaunchArgs {
    uint8_t *q;             // bf16 [1,T,H,D]
    uint8_t *k;             // bf16 [1,T,H,D]
    uint8_t *v;             // bf16 [1,T,H,D]
    uint8_t *a;             // bf16 [1,T,H,K]
    uint8_t *b;             // bf16 [1,T,H]
    uint8_t *A_log;         // fp32 [H]
    uint8_t *dt_bias;       // fp32 [H*D]
    uint8_t *state;         // fp32 [slots,H,K,V], K-contiguous (V-major)
    uint8_t *indices;       // int32 [numReq]
    uint8_t *starts;        // int32 [numReq+1]
    uint8_t *out;           // bf16 [1,T,H,D]
    uint8_t *track_indices; // int32 [numReq], null unless tracking
    uint8_t *track_lens;    // int32 [numReq], null unless tracking

    // Per-step state checkpoints for speculative verify; null when not
    // requested, which is what a decode launcher switches on. fp32, same
    // dtype and same K-contiguous orientation as the state pool, and indexed
    // by the request's position in the batch rather than by its state slot:
    //
    //     mid[req * mid_stride_req + step * mid_stride_step
    //                              + head * mid_stride_head + k]
    //
    // Strides are in elements and come from the caller's tensor, because the
    // buffer it hands over is a transposed view of a 5-D pool and is not
    // contiguous in the last two axes. When this is set the final state is
    // *not* written back: the commit takes the accepted step out of here
    // instead, and the source slot has to keep its pre-verify value.
    uint8_t *mid;           // fp32 [slots,steps,H,K,V]
    uint32_t mid_stride_req;
    uint32_t mid_stride_step;
    uint32_t mid_stride_head;

    // Scratch, below. Allocated by the host and owned by it for the duration of
    // the launch; a kernel that does not need a buffer ignores it rather than
    // asserting on it, so only the launchers documented as using one require it
    // to be non-null. Today that is the H=4 prefill launcher alone.
    //
    // The first four are the per-token quantities the recurrence consumes but
    // does not itself produce — the hoisted gate, beta and q/k L2
    // normalization. Upstream these are four `torch.empty` tensors that
    // `_prepare_inputs` fills in a separate Triton launch; here the producer
    // blocks of the same kernel fill them, so they never leave device memory
    // between the two phases. Laid out exactly as upstream lays them out, so
    // the row indexing is identical: row = token * H + head.
    uint8_t *qn;            // fp32 [1,T,H,D], normalized q, scale folded in
    uint8_t *kn;            // fp32 [1,T,H,D], normalized k
    uint8_t *gdecay;        // fp32 [1,T,H,D], exp(gate)
    uint8_t *beta;          // fp32 [1,T,H],   sigmoid(b)

    // Inter-core notification banks for `IBSet`/`IBWait`, int32 and **zeroed**
    // by the host: the API requires a zero-initialized workspace, and a stale
    // nonzero flag either hangs a producer or lets a consumer read a chunk that
    // was never written. Indexed by the API as
    // `[blockNum * 8 * eventID + blockIdx * 8]`.
    uint8_t *sync;

    float scale;
    float lower_bound;
    uint32_t num_req;
    uint32_t num_tokens;    // T; the producers chunk over it
    bool track_state;       // prefill only; decode compiles the store out
};

// One launcher per kernel file. `stream` is an `aclrtStream`, taken as `void*`
// so this header stays independent of the ACL headers.
//
// `LaunchKdaPrefillH4` is the one launcher that runs producers and consumers in
// the same grid, so it is also the one that requires the scratch and `sync`
// buffers above. See the file comment in `kda_varlen_recurrent_prefill_h4.cpp`
// for the block split and why its grid is capped at the physical core count.
void LaunchKdaPrefillH4(const KdaLaunchArgs &args, void *stream);
void LaunchKdaPrefillH8(const KdaLaunchArgs &args, void *stream);
void LaunchKdaDecodeH16(const KdaLaunchArgs &args, void *stream);
void LaunchKdaDecodeH32(const KdaLaunchArgs &args, void *stream);
void LaunchKdaDecodeH64(const KdaLaunchArgs &args, void *stream);
