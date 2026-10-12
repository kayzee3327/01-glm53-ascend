from typing import Optional

import torch
from sglang.srt.plugins.hook_registry import HookRegistry, HookType


def _prepared_recurrent(
    qn: torch.Tensor,
    kn: torch.Tensor,
    v: torch.Tensor,
    gdecay: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor,
    state: torch.Tensor,
    indices: torch.Tensor,
    starts: torch.Tensor,
    track_indices: Optional[torch.Tensor] = None,
    track_lens: Optional[torch.Tensor] = None,
    track_state: bool = False,
):
    from .ops.kda import prepared_recurrent

    return prepared_recurrent(
        qn, kn, v, gdecay, beta, out, state, indices, starts,
        track_indices, track_lens, track_state,
    )


def register() -> None:
    HookRegistry.register(
        "SGLANG_TARGET_PATH",
        _prepared_recurrent,
        HookType.REPLACE,
    )