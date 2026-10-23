"""Route sglang's KDA recurrence through the Ascend C op, for the two
upstream manual tests.

``glm_kda_varlen_recurrent_ascendc`` is imported from the benchmark rather
than re-copied: it is already the 1:1 replacement of
``glm_kda_varlen_recurrent_npu``, keeping upstream's validation and tile
arithmetic verbatim and swapping only the compute dispatch for
``torch.ops.sgl_glm53_ascend.kda_varlen_recurrent``. One copy, so the tests
and the benchmark cannot drift apart.

The op covers single-sequence H=4/8 prefill and H=16/32/64 decode. The two
tests also drive regimes outside that set -- multi-sequence packed prefill and
H=4 decode/verify -- where the op raises on purpose and production keeps its
Triton fallback. So this dispatcher mirrors production: it calls the op, falls
back to Triton on the op's own rejections, and nothing else. Every rejection is
counted and printed, so a run that silently fell back to Triton everywhere
cannot look like a pass.

Note this bypasses ``SGLANG_GLM53_KDA_PREFILL_PREPARE``: the op replaces the
whole implementation-selection block, both arms, so the prepared-prefill
variants of a case become repeats of the generic one rather than a second path.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

BENCH_DIR = Path("/data/wangkaize/projects/01-glm53-ascend/bench/kda_recurrent")
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from bench_kda_varlen_recurrent import (  # noqa: E402
    glm_kda_varlen_recurrent_ascendc,
)

from sglang.srt.hardware_backend.npu.attention.glm53 import (  # noqa: E402
    kda_recurrent_npu,
)

_TRITON = kda_recurrent_npu.glm_kda_varlen_recurrent_npu

# The op's deliberate `TORCH_CHECK`s, as in the benchmark's `unsupported`.
# Anything else is a real failure and propagates.
#
# `must be K-contiguous` belongs here for the same reason as the others: the op
# requires the native [slots,H,K,V] pool with stride(2)==1, the layout the
# Ascend backend normalizes both of its pools to, and the prefix test's
# `transpose=False` case deliberately hands over a plain contiguous pool that
# production never produces.
_REJECTIONS = (
    "not implemented yet",
    "does not support",
    "supports H=",
    "expects one",
    "must be K-contiguous",
)

counts: Counter[str] = Counter()
fallbacks: Counter[str] = Counter()


def glm_kda_varlen_recurrent_dispatch(**kwargs):
    """The op where it applies, Triton where the op rejects the regime."""
    try:
        result = glm_kda_varlen_recurrent_ascendc(**kwargs)
    except RuntimeError as exc:
        message = str(exc)
        if not any(marker in message for marker in _REJECTIONS):
            raise
        fallbacks[message.splitlines()[0].strip()] += 1
        return _TRITON(**kwargs)
    counts["ascendc"] += 1
    return result


def patch() -> None:
    """Point the sglang backend's recurrence at the dispatcher.

    ``AscendKDAAttnBackend._glm_recurrent`` imports the function inside the
    call, so rebinding the module attribute is enough to cover the backend
    cases too.
    """
    kda_recurrent_npu.glm_kda_varlen_recurrent_npu = (
        glm_kda_varlen_recurrent_dispatch
    )


def report() -> dict:
    """Coverage of the run: op calls, and each rejection that fell back."""
    return {
        "ascendc_calls": counts["ascendc"],
        "triton_fallback_calls": sum(fallbacks.values()),
        "triton_fallback_reasons": dict(fallbacks),
    }
