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
    head_offset: int = 0,
):
    return torch.ops.sgl_glm53_ascend.prepared_recurrent(
        qn, kn, v, gdecay, beta, out, state, indices, starts,
        track_indices, track_lens, track_state, head_offset,
    )


def kda_varlen_recurrent(
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: float,
    scale: float,
    intermediate_state: Optional[torch.Tensor] = None,
    prefill: bool = False,
    track_state_indices: Optional[torch.Tensor] = None,
    track_state_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused KDA varlen recurrence; returns the output, updates state in place.

    Replaces the implementation-selection block inside sglang's
    ``glm_kda_varlen_recurrent_npu`` — both arms, the prepared prefill fast path
    and the generic Triton launch — not the function around it. Argument names
    follow that function, with one departure: ``lower_bound`` and ``scale`` are
    required rather than optional, since the caller resolves both beforehand.

    Because the caller's own validation runs before the selection, this op does
    not repeat it, and so it is not a standalone substitute for
    ``glm_kda_varlen_recurrent_npu``: called with inputs that never passed those
    checks it will index out of bounds rather than raise. The op validates only
    what the kernels index against — dtypes, H == HV, K == V == 128, the state
    layout.

    ``initial_state_source`` must be [slots, H, K, V] with K contiguous
    (stride(2)==1, stride(3)==128), the layout the Ascend backend normalizes
    both of its pools to.

    Supported: prefill at H=4/8 with one packed sequence, decode at H=16/32/64.
    Anything else raises, including ``intermediate_state`` (target verify).

    Not implemented yet: the host function raises after selecting a kernel.
    """
    return torch.ops.sgl_glm53_ascend.kda_varlen_recurrent(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
        scale=scale,
        intermediate_state=intermediate_state,
        prefill=prefill,
        track_state_indices=track_state_indices,
        track_state_lens=track_state_lens,
    )