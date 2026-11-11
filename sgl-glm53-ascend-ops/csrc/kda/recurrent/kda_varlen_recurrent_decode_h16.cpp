// GLM-5.3 KDA varlen recurrence — decode kernel, H=16 (TP4).
//
// The implementation is shared across the three decode widths and lives in
// `kda_varlen_recurrent_decode.h`; that file carries the tiling rationale and
// the UB map. This translation unit only pins H and emits the launcher.
//
// H=16 is the narrowest decode width and the one most starved of blocks: at
// DP2TP4 the profile shows 3 requests at 256K (48 blocks, 1.2 waves) and a
// single request at 512K (16 blocks, 40% of one wave). Nothing in the tiling
// can fix the 512K case — there are only 16 independent work items on the
// device — so it stays the shape where a value split would buy occupancy if
// the per-token duplication it costs ever became worth paying.

#include "kda_varlen_recurrent_decode.h"

#ifndef __CCE_KT_TEST__
#include <acl/acl.h>
#endif

KDA_DEFINE_DECODE_WIDTH(16)

#ifndef __CCE_KT_TEST__

void LaunchKdaDecodeH16(const KdaLaunchArgs &args, void *stream)
{
    KDA_LAUNCH_DECODE(16);
}

#endif
