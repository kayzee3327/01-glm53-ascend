// GLM-5.3 KDA varlen recurrence v2 — single kernel implementation.
//
// One kernel handles both prefill and decode across all supported head
// widths. The host validates the (prefill, H) combination and routes here;
// internal branching replaces v1's one-kernel-per-width layout.

#include "kernel_operator.h"
#include "kda_varlen_recurrent_v2.h"

#ifndef __CCE_KT_TEST__
#include <acl/acl.h>
#endif

template <bool TRACK_STATE>
class KdaRecurrentV2Kernel {
public:
    __aicore__ inline void Init(const KdaV2LaunchArgs &args)
    {
        // TODO: initialize GM pointers and UB layout from args.
        (void)args;
    }

    __aicore__ inline void Process()
    {
        // TODO: implement the v2 recurrent kernel.
    }
};

#ifndef __CCE_KT_TEST__

void LaunchKdaRecurrentV2(const KdaV2LaunchArgs &args, void *stream)
{
    // TODO: launch KdaRecurrentV2Kernel.
    (void)args;
    (void)stream;
}

#endif
