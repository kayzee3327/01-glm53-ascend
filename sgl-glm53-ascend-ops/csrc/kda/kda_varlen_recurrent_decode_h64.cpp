// GLM-5.3 KDA varlen recurrence — decode kernel, H=64 (TP1).
//
// Decode is the opposite shape from prefill: one token per request, `T ==
// numReq`, so there is no long serial token loop to hide latency behind. Each
// request takes a single recurrence step, and the whole call is short enough
// that launch overhead is a real share of it — measured on the Triton baseline,
// decode at batch 1 costs ~0.18 ms regardless of H, i.e. it is launch-bound
// rather than work-bound. Only from batch 32 up does H start to matter
// (B1/B32/B128 = 0.18/0.61/1.88 ms).
//
// See `kda_varlen_recurrent.h` for the shared contract and the reference
// semantics; the host file validates everything before this launcher is reached.
//
// H=64 is kept in its own translation unit so its tiling, UB map and block
// mapping can diverge from the other decode widths. At H=64 the heads
// over-subscribe the 48 cores on their own, so this is the width most likely to
// want a coarse BV and the fewest blocks per head.
//
// Tracking is prefill-only, so the prefix-checkpoint store is compiled out here
// and there is a single kernel instantiation.

#include "kernel_operator.h"
#include "kda_varlen_recurrent.h"
#include "utils.h"

#ifndef __CCE_KT_TEST__
#include <acl/acl.h>
#endif

// ── Device-side ────────────────────────────────────────────────────────

// H_TOTAL   heads per state slot; 64 for decode at TP1.
template<uint32_t H_TOTAL>
class KdaVarlenRecurrentDecodeH64 {
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
        float scale,
        float lowerBound,
        uint32_t numReq);

    __aicore__ inline void Process();

private:
    struct config {
        static constexpr uint32_t D = static_cast<uint32_t>(config_D);
        static constexpr uint32_t H = 64;
        // TODO(tiling): BV / NUM_V_TILES and the UB map. One step per request,
        // so prefer a grid that covers (request, head, value tile) in one wave
        // over the 48 vector cores.
    };
};

template<uint32_t H_TOTAL>
__aicore__ inline void KdaVarlenRecurrentDecodeH64<H_TOTAL>::Init(
    GM_ADDR Q, GM_ADDR K, GM_ADDR V, GM_ADDR A,
    GM_ADDR B,
    GM_ADDR A_LOG,
    GM_ADDR DT_BIAS,
    GM_ADDR STATE,
    GM_ADDR INDICES,
    GM_ADDR STARTS,
    GM_ADDR OUT,
    float scale,
    float lowerBound,
    uint32_t numReq)
{
    // TODO(device): bind GlobalTensors and lay out UB.
}

template<uint32_t H_TOTAL>
__aicore__ inline void KdaVarlenRecurrentDecodeH64<H_TOTAL>::Process()
{
    // TODO(device): block -> (request, head, value tile), then one recurrence
    // step. Each block owns a disjoint output/state range. `numReq` is a runtime
    // value here, unlike prefill's numReq==1, so the decomposition must handle
    // a grid that does not divide the work evenly.
}

extern "C" __global__ __vector__
void kda_varlen_recurrent_kernel_decode_h64(
    GM_ADDR Q, GM_ADDR K, GM_ADDR V, GM_ADDR A, GM_ADDR B,
    GM_ADDR A_LOG, GM_ADDR DT_BIAS,
    GM_ADDR STATE, GM_ADDR INDICES, GM_ADDR STARTS, GM_ADDR OUT,
    float scale, float lowerBound, uint32_t numReq
) {
    KdaVarlenRecurrentDecodeH64<64> op;
    op.Init(Q, K, V, A, B, A_LOG, DT_BIAS,
            STATE, INDICES, STARTS, OUT,
            scale, lowerBound, numReq);
    op.Process();
}

// ── Host-side launcher ─────────────────────────────────────────────────

#ifndef __CCE_KT_TEST__

void LaunchKdaDecodeH64(const KdaLaunchArgs &args, void *stream)
{
    // TODO(host): real grid once the tiling is chosen. Decode work scales with
    // numReq, so the block count should depend on it rather than being fixed.
    constexpr uint32_t blockDim = 32;
    auto acl_stream = static_cast<aclrtStream>(stream);

    kda_varlen_recurrent_kernel_decode_h64<<<blockDim, nullptr, acl_stream>>>(
        args.q, args.k, args.v, args.a, args.b,
        args.A_log, args.dt_bias,
        args.state, args.indices, args.starts, args.out,
        args.scale, args.lower_bound, args.num_req);
}

#endif
