// GLM-5.3 KDA varlen recurrence v2 — seam between the host op and the kernels.
//
//   kda_varlen_recurrent_v2.cpp               host: validation, dispatch, registration
//   kda_varlen_recurrent_v2_prefill_h4.cpp    H=4  (TP16), stub
//   kda_varlen_recurrent_v2_prefill_h8.cpp    H=8  (TP8),  stub
//   kda_varlen_recurrent_v2_decode_h16.cpp    H=16 (TP4),  stub
//   kda_varlen_recurrent_v2_decode_h32.cpp    H=32 (TP2),  stub
//   kda_varlen_recurrent_v2_decode_h64.cpp    H=64 (TP1),  stub
//
// Registered as `kda_varlen_recurrent_v2` during development. Cutover to
// `kda_varlen_recurrent` is two changes: rename the TORCH_LIBRARY_FRAGMENT
// string in kda_varlen_recurrent_v2.cpp and drop v1 from CMakeLists. The
// Python wrapper needs no change.
//
// Included by both host and device TUs; must stay free of torch headers.

#pragma once

#include <cstdint>
#include "kernel_operator.h"

constexpr int64_t config_D_v2 = 128;

namespace kda_prefill_v2 {

constexpr uint32_t kNumVectorCores = 40;

constexpr uint32_t kNumConsumers   = 32;
constexpr uint32_t kNumProducers   = kNumVectorCores - kNumConsumers;  // 8
constexpr uint32_t kBlockDim       = kNumConsumers + kNumProducers;    // 40
static_assert(kBlockDim <= kNumVectorCores,
              "the prefill grid must fit one scheduler wave or it can "
              "deadlock on the inter-core notifications");
static_assert(kNumProducers > 0 && kNumConsumers % kNumProducers == 0,
              "every producer must own the same number of consumer lanes");

constexpr uint32_t kMaxRounds      = 2;
constexpr uint32_t kHeadsPerRound  = kNumConsumers / (128 / 16);  // 4

constexpr uint32_t kChunkTokens    = 16;

constexpr uint32_t kSyncRing       = 4;
static_assert((kSyncRing & (kSyncRing - 1)) == 0, "ring size must be 2^n");

constexpr uint32_t kSyncEventIds   = kNumConsumers * kSyncRing;
constexpr uint32_t kSyncInt32Count =
    kBlockDim * 8 * kSyncEventIds + kBlockDim * 8 + 8;

}  // namespace kda_prefill_v2


struct KdaV2LaunchArgs {
    GM_ADDR q;             // bf16 [1,T,H,D]
    GM_ADDR k;             // bf16 [1,T,H,D]
    GM_ADDR v;             // bf16 [1,T,H,D]
    GM_ADDR a;             // bf16 [1,T,H,K]
    GM_ADDR b;             // bf16 [1,T,H]
    GM_ADDR A_log;         // fp32 [H]
    GM_ADDR dt_bias;       // fp32 [H*D]
    GM_ADDR state;         // fp32 [slots,H,K,V], K-contiguous (V-major)
    GM_ADDR indices;       // int32 [numReq]
    GM_ADDR starts;        // int32 [numReq+1]
    GM_ADDR out;           // bf16 [1,T,H,D], allocated by the caller
    GM_ADDR track_indices; // int32 [numReq], null unless tracking
    GM_ADDR track_lens;    // int32 [numReq], null unless tracking

    GM_ADDR mid;           // fp32 [slots,steps,H,K,V]
    uint32_t mid_stride_req;
    uint32_t mid_stride_step;
    uint32_t mid_stride_head;

    GM_ADDR qn;            // fp32 [1,T,H,D], normalized q, scale folded in
    GM_ADDR kn;            // fp32 [1,T,H,D], normalized k
    GM_ADDR gdecay;        // fp32 [1,T,H,D], exp(gate)
    GM_ADDR beta;          // fp32 [H,Tpad],  sigmoid(b)

    GM_ADDR sync;          // IBSet/IBWait notification banks

    float    scale;
    float    lower_bound;
    uint32_t num_req;
    uint32_t num_tokens;   // T; the producers chunk over it
    bool     track_state;  // prefill only; decode compiles the store out
};

void LaunchKdaRecurrentV2(const KdaV2LaunchArgs &args, void *stream);
