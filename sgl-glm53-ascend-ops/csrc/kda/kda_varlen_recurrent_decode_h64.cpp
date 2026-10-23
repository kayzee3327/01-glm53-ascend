// GLM-5.3 KDA varlen recurrence — decode kernel, H=64 (TP1).
//
// The implementation is shared across the three decode widths and lives in
// `kda_varlen_recurrent_decode.h`; that file carries the tiling rationale and
// the UB map. This translation unit only pins H and emits the launcher.
//
// H=64 holds every KDA head on one device, so a single request already fills
// 64 blocks against 40 vector cores and concurrency past that just adds waves.
// It is the one decode width where blocks are never scarce, and correspondingly
// the one where the fixed per-block cost — the state load and the per-token
// Vector/Scalar round trips — is least able to hide.

#include "kda_varlen_recurrent_decode.h"

#ifndef __CCE_KT_TEST__
#include <acl/acl.h>
#endif

KDA_DEFINE_DECODE_WIDTH(64)

#ifndef __CCE_KT_TEST__

void LaunchKdaDecodeH64(const KdaLaunchArgs &args, void *stream)
{
    KDA_LAUNCH_DECODE(64);
}

#endif
