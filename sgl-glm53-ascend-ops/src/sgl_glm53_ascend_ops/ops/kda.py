from typing import Optional

import torch


def prepared_recurrent(
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
    return torch.ops.sgl_glm53_ascend.prepared_recurrent(
        qn, kn, v, gdecay, beta, out, state, indices, starts,
        track_indices, track_lens, track_state,
    )