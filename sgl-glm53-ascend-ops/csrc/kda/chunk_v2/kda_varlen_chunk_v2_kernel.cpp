// GLM-5.3 chunked KDA v2 — MIX kernel stub (1 AIC + 2 AIV per block).
//
// Kernel implementation goes here. Entry points and launch wrapper below
// are wired up and callable; fill in Init/Process before enabling in prod.

#include "kernel_operator.h"
#include "kda_varlen_chunk_v2.h"

#ifndef __CCE_KT_TEST__
#include <acl/acl.h>
#endif

namespace kda_chunk_v2 {

constexpr uint32_t kFracRow = 16;
constexpr uint32_t kFracCol = 32 / sizeof(float);

constexpr uint32_t kL1AOffset = 0;
constexpr uint32_t kL1BOffset = 128 * 1024;

}  // namespace kda_chunk_v2

template <bool TRACK_STATE>
class KdaVarlenChunkV2 {
public:
    __aicore__ inline void Init(const KdaChunkV2LaunchArgs &args,
                                uint32_t numHeads)
    {
        // TODO: initialize GM pointers and UB layout from args.
        (void)args;
        (void)numHeads;
    }

    __aicore__ inline void Process()
    {
        // TODO: implement the v2 kernel.
    }
};

#define KDA_CHUNK_V2_KERNEL(NAME, TRACK)                                    \
extern "C" __global__ __aicore__                                            \
void NAME(                                                                  \
    GM_ADDR Q, GM_ADDR K, GM_ADDR V, GM_ADDR A, GM_ADDR B,                  \
    GM_ADDR A_LOG, GM_ADDR DT_BIAS,                                         \
    GM_ADDR STATE, GM_ADDR INDICES, GM_ADDR STARTS, GM_ADDR OUT,            \
    GM_ADDR TRACK_INDICES, GM_ADDR TRACK_LENS, GM_ADDR WORKSPACE,           \
    float scale, float lowerBound,                                          \
    uint32_t numReq, uint32_t numHeads                                      \
) {                                                                         \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                      \
    KdaChunkV2LaunchArgs args{                                              \
        Q, K, V, A, B, A_LOG, DT_BIAS, STATE, INDICES, STARTS, OUT,         \
        TRACK_INDICES, TRACK_LENS, WORKSPACE,                               \
        scale, lowerBound, numReq, numHeads, TRACK};                        \
    KdaVarlenChunkV2<TRACK> op;                                             \
    op.Init(args, numHeads);                                                \
    op.Process();                                                           \
}

KDA_CHUNK_V2_KERNEL(kda_chunk_v2_plain, false)
KDA_CHUNK_V2_KERNEL(kda_chunk_v2_track, true)

#ifndef __CCE_KT_TEST__

void LaunchKdaVarlenChunkV2(const KdaChunkV2LaunchArgs &args,
                            uint32_t num_heads, void *stream)
{
    const uint32_t blockDim = args.num_req * num_heads;
    auto acl_stream = static_cast<aclrtStream>(stream);

#define KDA_CHUNK_V2_CALL(KERNEL)                                           \
    KERNEL<<<blockDim, nullptr, acl_stream>>>(                              \
        args.q, args.k, args.v, args.a, args.b,                             \
        args.A_log, args.dt_bias,                                           \
        args.state, args.indices, args.starts, args.out,                    \
        args.track_indices, args.track_lens, args.workspace,                \
        args.scale, args.lower_bound, args.num_req, num_heads)

    if (args.track_state) {
        KDA_CHUNK_V2_CALL(kda_chunk_v2_track);
    } else {
        KDA_CHUNK_V2_CALL(kda_chunk_v2_plain);
    }

#undef KDA_CHUNK_V2_CALL
}

#endif
