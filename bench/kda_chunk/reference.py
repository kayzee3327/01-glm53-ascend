"""A literal transcription of the ChunkKdaFwd formulas, in plain torch.

This exists to be *obviously* correct rather than fast. The Triton arm of the
benchmark is itself one of the three kernels under comparison, so it cannot be
its own correctness story; this module is the independent anchor both it and
the CANN kernel are checked against once, at small shapes, before either is
trusted as a reference.

Every expression below is the corresponding line of
``ops-transformer/attention/chunk_kda_fwd/README.md`` written out directly: no
algebraic rearrangement, no factoring of the decay out of the pairwise terms.
That is the whole point. The decay inside a chunk is applied as the per-pair
difference ``2^(gk_i - gk_j)``, which costs an O(C^2 K) temporary but cannot
overflow, where the separable form ``(q 2^gk) (k 2^-gk)^T`` that a real kernel
must use to reach a matmul can and does overflow in fp32 once the chunk-local
cumulative gate passes -128 in log2.

Layout is TND with ``cu_seqlens`` throughout -- the packed-varlen prefill shape
the caller actually runs -- so there is one layout to reason about here.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

LN2 = math.log(2.0)


def activate_gate(
    g: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    *,
    safe_gate: bool,
    lower_bound: float,
    use_gate_in_kernel: bool,
) -> torch.Tensor:
    """The activated natural-log gate ``gamma``, from raw or pre-activated ``g``.

    ``g`` is ``[T, HV, K]``. With ``use_gate_in_kernel=False`` the caller has
    already done this and ``g`` passes through untouched, which is the branch
    the ``gk``-only comparisons use.
    """
    if not use_gate_in_kernel:
        return g

    if A_log is None:
        raise ValueError("use_gate_in_kernel=True requires A_log")

    _, hv, k = g.shape
    x = g
    if dt_bias is not None:
        x = x + dt_bias.reshape(1, hv, k)

    a = torch.exp(A_log).reshape(1, hv, 1)
    if safe_gate:
        # Bounded gate: gamma in (lower_bound, 0), so the per-token decay can
        # never be worse than `lower_bound` regardless of what x does.
        return lower_bound * torch.sigmoid(a * x)
    return -a * torch.nn.functional.softplus(x)


def chunk_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    chunk_size: int,
    cu_seqlens: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    safe_gate: bool = True,
    lower_bound: float = -5.0,
    use_gate_in_kernel: bool = True,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(attn_out, final_state)`` for TND varlen input.

    ``q``/``k`` are ``[T, H, K]``, ``v`` is ``[T, HV, V]``, ``g`` is
    ``[T, HV, K]``, ``beta`` is ``[T, HV]``, and ``cu_seqlens`` is ``[N+1]``.
    ``attn_out`` comes back in ``q``'s dtype as ``[T, HV, V]``; the state stays
    fp32 as ``[N, HV, K, V]``, matching the op.

    ``dtype`` is the accumulation dtype. fp32 mirrors what a kernel can
    actually hold; pass fp64 to separate a real disagreement from accumulated
    rounding when a comparison is marginal.
    """
    t, h, kdim = q.shape
    hv, vdim = v.shape[1], v.shape[2]
    if hv % h:
        raise ValueError(f"HV={hv} must be a multiple of H={h}")

    out_dtype = q.dtype
    # GQA: value head `hv` reads the query/key head it is grouped under.
    group = hv // h
    qf = q.to(dtype).repeat_interleave(group, dim=1)
    kf = k.to(dtype).repeat_interleave(group, dim=1)
    vf = v.to(dtype)
    bf = beta.to(dtype)

    gamma = activate_gate(
        g.to(dtype),
        None if A_log is None else A_log.to(dtype),
        None if dt_bias is None else dt_bias.to(dtype),
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
    )

    seqlens = cu_seqlens.tolist()
    n = len(seqlens) - 1

    attn_out = torch.empty((t, hv, vdim), dtype=dtype, device=q.device)
    final_state = torch.empty((n, hv, kdim, vdim), dtype=torch.float32, device=q.device)
    eye = torch.eye(chunk_size, dtype=dtype, device=q.device)

    for seq in range(n):
        start, end = seqlens[seq], seqlens[seq + 1]
        if initial_state is None:
            state = torch.zeros((hv, kdim, vdim), dtype=dtype, device=q.device)
        else:
            state = initial_state[seq].to(dtype)

        for base in range(start, end, chunk_size):
            stop = min(base + chunk_size, end)
            length = stop - base

            # Chunk-local cumulative gate, base 2. Monotonically decreasing in
            # the token index, because gamma is negative.
            gk = torch.cumsum(gamma[base:stop], dim=0) / LN2  # [L, HV, K]
            gk_last = gk[length - 1]  # [HV, K] -- gk at the last valid token

            qc, kc, vc = qf[base:stop], kf[base:stop], vf[base:stop]
            bc = bf[base:stop]  # [L, HV]

            qg = qc * torch.exp2(gk)
            kg = kc * torch.exp2(gk_last.unsqueeze(0) - gk)

            # The pairwise decay, as a difference and never as a ratio of two
            # separately exponentiated factors. Masked entries are driven to
            # -inf first so that exp2 yields an exact zero instead of an inf
            # that a later multiply by zero would turn into a NaN.
            idx = torch.arange(length, device=q.device)
            causal = idx.unsqueeze(1) >= idx.unsqueeze(0)  # [L, L], j <= i
            strict = idx.unsqueeze(1) > idx.unsqueeze(0)  # [L, L], j < i

            diff = gk.unsqueeze(1) - gk.unsqueeze(0)  # [L, L, HV, K]

            def decayed(lhs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                """``sum_d lhs[i,d] k[j,d] 2^(gk[i,d]-gk[j,d])`` under ``mask``."""
                p = torch.exp2(diff.masked_fill(~mask[:, :, None, None], -torch.inf))
                return (lhs.unsqueeze(1) * kc.unsqueeze(0) * p).sum(-1)  # [L, L, HV]

            # Aqk carries `scale`; the state term below applies it separately.
            aqk = scale * decayed(qc, causal)
            lmat = decayed(kc, strict) * bc.unsqueeze(1)

            # Akk = (I + L)^-1, with I + L unit lower triangular.
            aqk = aqk.permute(2, 0, 1)  # [HV, L, L]
            lmat = lmat.permute(2, 0, 1)
            akk = torch.linalg.solve_triangular(
                eye[:length, :length] + lmat, eye[:length, :length], upper=False
            )

            w = akk @ (bc.unsqueeze(-1) * kc * torch.exp2(gk)).permute(1, 0, 2)
            u = akk @ (bc.unsqueeze(-1) * vc).permute(1, 0, 2)
            v_new = u - w @ state  # [HV, L, V]

            attn_out[base:stop] = (
                scale * (qg.permute(1, 0, 2) @ state) + aqk @ v_new
            ).permute(1, 0, 2)

            state = torch.exp2(gk_last).unsqueeze(-1) * state + (
                kg.permute(1, 2, 0) @ v_new
            )

        final_state[seq] = state.to(torch.float32)

    return attn_out.to(out_dtype), final_state
