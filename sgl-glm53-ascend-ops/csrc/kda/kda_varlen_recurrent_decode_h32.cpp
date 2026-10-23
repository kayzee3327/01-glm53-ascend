// GLM-5.3 KDA varlen recurrence — decode kernel, H=32 (TP2).
//
// The implementation is shared across the three decode widths and lives in
// `kda_varlen_recurrent_decode.h`; that file carries the tiling rationale and
// the UB map. This translation unit only pins H and emits the launcher.
//
// H=32 is the priority shape: DP4TP2 at 256K context puts 2 requests and 32
// heads on a device, 4 tokens per request under EAGLE verify. That is 64 blocks
// against 40 vector cores — 1.6 waves, so the second wave runs at 60%
// occupancy and the kernel is not latency-hidden. The traffic floor for the
// shape is ~17 us (20 MB, four checkpoints per step dominating), against the
// Triton baseline's 38.8 us under graph replay.

#include "kda_varlen_recurrent_decode.h"

#ifndef __CCE_KT_TEST__
#include <acl/acl.h>
#endif

KDA_DEFINE_DECODE_WIDTH(32)

#ifndef __CCE_KT_TEST__

void LaunchKdaDecodeH32(const KdaLaunchArgs &args, void *stream)
{
    KDA_LAUNCH_DECODE(32);
}

#endif
