"""PyTorch references for the GLM KDA varlen operator, chunked and recurrent.

Imported by ``bench_kda_chunk.py`` for its correctness anchor, so this module
must not import that one at module scope -- the import that used to be here
pulled the benchmark in as a second module object and re-ran its device pin.
``run()`` below does the comparison these two are checked against each other
with, and takes the benchmark's inputs through a deferred import.
"""

from typing import Optional, Tuple

import torch
import torch_npu  # noqa: F401  registers the NPU device backend
from einops import rearrange, einsum

def kda_pre_torch(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    b: torch.Tensor,
    lower_bound,
    scale,
) -> Tuple[torch.Tensor]:
    q, k, a, b = (t.float() for t in (q, k, a, b))
    # EPSILON = 1e-6
    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * scale
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)

    # Log decay: [1, T, HV, K]
    gate = (
        A_log.float().exp()[None, None, :, None]
        * (a + dt_bias.float()[None, None, :, :])
    )
    g = lower_bound * torch.sigmoid(gate)
    # BETA: [1, T, HV]
    beta = torch.sigmoid(b)
    return qn, kn, g, beta

def kda_chunk_torch(
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
    lower_bound: Optional[float],
    chunk_size: int,
    scale: Optional[float] = None,
    prefill: bool = False,
    track_state_indices: Optional[torch.Tensor] = None,
    track_state_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    PyTorch reference for GLM KDA varlen recurrent.

    Inputs:
        q, k:                 [1, T, H, K]
        v:                    [1, T, HV, V]
        a:                    [1, T, HV, K]
        b:                    [1, T, HV]
        A_log:                [HV]
        dt_bias:              [HV, K]
        initial_state_source: [N, HV, K, V]
        initial_state_indices:[B]
        cu_seqlens:           [B + 1]
        chunk_size:           sequence lengths must be a multiple of it

    Returns:
        output:               [1, T, HV, V]

    State updates are written in-place to the provided buffers.
    """
    dtype = v.dtype
    T, H, K = q.shape[1:]
    HV, V = v.shape[2:]
    B = cu_seqlens.numel() - 1
    BT = chunk_size
    if scale is None:
        scale = K ** -0.5

    # assume no gqa
    assert H == HV

    # precompute
    qn, kn, g, beta = kda_pre_torch(A_log, a, dt_bias, q, k, b, lower_bound, scale)

    output = torch.zeros(1, T, HV, V, dtype=torch.float32, device=v.device)
    v = v.float()

    for seq in range(B):
        bos = int(cu_seqlens[seq])
        eos = int(cu_seqlens[seq + 1])
        slot = int(initial_state_indices[seq])
        NT = (eos - bos) // BT
        assert (eos - bos) % BT == 0

        # State layout: [1, HV, K, V]
        if slot > 0:
            S = initial_state_source[slot : slot + 1].float().clone()
        else:
            S = torch.zeros((1, HV, K, V), dtype=torch.float32, device=v.device)

        # Checkpoint after `track_len` consumed tokens, sequence-local like the
        # recurrence. `track_len == 0` is the initial state, which no chunk
        # produces, so it is written here instead.
        track_index = 0
        if track_state_indices is not None:
            track_index = int(track_state_indices[seq])
            if track_index > 0:
                track_len = int(track_state_lens[seq])
                if track_len == 0:
                    initial_state_source[track_index].copy_(S[0])

        # Rearrange into chunks: [B, head, NT, BT, ...]
        q_c, k_c, v_c, g_c, beta_c = [
            rearrange(x[:, bos:eos], 'b (nt bt) h ... -> b h nt bt ...', bt=BT)
            for x in (qn, kn, v, g, beta)
        ]

        # log decay cumsum
        g_c = g_c.cumsum(-2)

        # lower triangular inverse M_t as A
        mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
        A = torch.zeros(*g_c.shape[:-1], BT, dtype=torch.float, device=q.device)
        for i in range(BT):
            k_i = k_c[..., i, :]
            g_i = g_c[..., i:i+1, :]
            A[..., i] = einsum(k_c * (g_c - g_i).exp(), k_i,
                               'b h nt bt k , b h nt k -> b h nt bt')
        A = A * beta_c[..., None]
        A = -A.masked_fill(mask, 0)
        for i in range(1, BT):
            A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
        A = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta_c[..., None, :]

        # auxiliary matrix
        w = A @ (g_c.exp() * k_c)
        u = A @ v_c

        o = torch.zeros_like(v_c)
        mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
        for i in range(0, NT):
            # [B, HV, BT, ...]
            q_i = q_c[:, :, i]      # [B, HV, BT, K]
            k_i = k_c[:, :, i]      # [B, HV, BT, K]
            u_i = u[:, :, i]        # [B, HV, BT, V]
            g_i = g_c[:, :, i]      # [B, HV, BT, K]
            w_i = w[:, :, i]        # [B, HV, BT, K]
            # Intra Chunk Aqk: per value head (q from qk head, g from value head, k from qk head)
            Aqk = torch.zeros(*g_i.shape[:-1], BT, dtype=torch.float, device=q.device)
            for j in range(BT):
                k_j = k_c[:, :, i, j]
                g_j = g_c[:, :, i, j:j+1, :]
                Aqk[..., j] = torch.einsum('... c d, ... d -> ... c', q_i * (g_i - g_j).exp(), k_j)
            Aqk = Aqk.masked_fill(mask, 0)
            # Pseudo-value term
            v_i = u_i - w_i @ S
            # Chunk Output
            o[:, :, i] = (q_i * g_i.exp()) @ S + Aqk @ v_i
            # Prefix state inside this chunk: S_r after its first `r` tokens,
            # reconstructed from the chunk's own start state and pseudo-values.
            # `r == BT` reproduces the chunk-final update below, and `r == 0` is
            # the start state itself, written before the loop.
            if track_index > 0 and i * BT < track_len <= (i + 1) * BT:
                r = track_len - i * BT
                g_r = g_i[:, :, r - 1 : r]
                S_r = S * rearrange(g_r[:, :, 0].exp(), 'b h k -> b h k 1')
                S_r = S_r + rearrange(
                    (g_r - g_i[:, :, :r]).exp() * k_i[:, :, :r], 'b h c k -> b h k c'
                ) @ v_i[:, :, :r]
                initial_state_source[track_index].copy_(S_r[0])
            # Chunk State Update
            S = S * rearrange(g_i[:, :, -1].exp(), 'b h k -> b h k 1')
            S = S + rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i

        output[:, bos:eos] = rearrange(o, 'b h n c d -> b (n c) h d')
        if slot > 0:
            initial_state_source[slot].copy_(S[0])

    return output.to(dtype)

def kda_recur_torch(
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
    lower_bound: Optional[float],
    scale: Optional[float] = None,
    intermediate_state: Optional[torch.Tensor] = None,
    prefill: bool = False,
    track_state_indices: Optional[torch.Tensor] = None,
    track_state_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    PyTorch reference for GLM KDA varlen recurrent.

    Inputs:
        q, k:                 [1, T, H, K]
        v:                    [1, T, HV, V]
        a:                    [1, T, HV, K]
        b:                    [1, T, HV]
        A_log:                [HV]
        dt_bias:              [HV, K]
        initial_state_source: [N, HV, K, V]
        initial_state_indices:[B]
        cu_seqlens:           [B + 1]

    Optional:
        intermediate_state:   [B, max_seq_len, HV, K, V]
        track_state_indices:  [B]
        track_state_lens:     [B]

    Returns:
        output:               [1, T, HV, V]

    State updates are written in-place to the provided buffers.
    """
    T, H, K_DIM = q.shape[1:]
    HV, V_DIM = v.shape[2:]
    B = cu_seqlens.numel() - 1
    if scale is None:
        scale = K_DIM**-0.5

    # assume no gqa
    assert H == HV

    # --------------------------------------------------
    # Precompute Q, K, decay and beta
    # --------------------------------------------------
    qn, kn, g, beta = kda_pre_torch(A_log, a, dt_bias, q, k, b, lower_bound, scale)
    decay = g.exp()

    # --------------------------------------------------
    # Varlen Recurrence
    # --------------------------------------------------
    output = torch.zeros_like(v)

    for seq in range(B):
        bos = int(cu_seqlens[seq])
        eos = int(cu_seqlens[seq + 1])
        slot = int(initial_state_indices[seq])

        # State layout: [1, HV, K, V]
        if slot > 0:
            state = initial_state_source[slot : slot + 1].float().clone()
        else:
            state = torch.zeros((1, HV, K_DIM, V_DIM), dtype=torch.float32, device=v.device)

        if track_state_indices is not None:
            track_index = int(track_state_indices[seq])
            if track_index > 0:
                track_len = int(track_state_lens[seq])
                if track_len == 0:
                    initial_state_source[track_index].copy_(state[0])

        for i in range(bos, eos):
            t = i - bos

            # per channel decay
            # [1, HV, K, V] point-wise [1, HV, K, 1]
            state = torch.einsum("bhkv,bhk->bhkv", state, decay[:,t])

            # compute delta
            prediction = torch.einsum("bhk,bhkv->bhv", kn[:,t], state)
            delta = torch.einsum(
                "bhv,bh->bhv",
                v[:,t].float() - prediction,
                beta[:,t],
            )

            # update state
            state = state + torch.einsum("bhk,bhv->bhkv", kn[:,t], delta)

            # Save checkpoint after consuming `track_len` tokens. The kernel
            # tests `i + 1 == track_len` on its sequence-local counter, so the
            # bound is on `t`, not on the packed index `i`.
            if track_state_indices is not None:
                if track_index > 0 and t + 1 == track_len:
                    initial_state_source[track_index].copy_(state[0])
            # Save per-token intermediate state. Indexed by sequence-local step
            # like the kernel's `i * M1`; `i - 1` would write the packed
            # position, one row short at bos=0 and wrong for every later
            # sequence.
            if intermediate_state is not None and slot > 0:
                intermediate_state[seq, t].copy_(state[0])

            # output with query
            output[:, i] = torch.einsum("bhk,bhkv->bhv", qn[:,t], state)

        # Write back final state when not saving intermediates.
        if intermediate_state is None and slot > 0:
            initial_state_source[slot].copy_(state[0])

    return output.to(v.dtype)


def run():
    """Check the two references against each other on one shape.

    The benchmark is imported here rather than at module scope: it imports this
    module for its anchor, so a top-level import either way round would be
    circular.
    """
    from bench_kda_chunk import ATOL, RTOL, Case, clone_state, make_inputs

    tokens = 2**10 * 4
    heads = 8
    chunk_size = 64
    c = Case(heads, tokens, chunk_size)
    inp = make_inputs(c, chunk=True)

    # Cloned because the final state is written into the pool in place.
    chunk_kwargs = inp.chunk_kwargs()
    chunk_kwargs["initial_state_source"] = clone_state(chunk_kwargs["initial_state_source"])
    recur_kwargs = inp.recurrence_kwargs()
    recur_kwargs["initial_state_source"] = clone_state(recur_kwargs["initial_state_source"])

    # Same recurrence summed in a different order, so compared at the
    # benchmark's tolerances rather than `assert_close`'s fp32 defaults.
    chunk_out = kda_chunk_torch(**chunk_kwargs)
    recur_out = kda_recur_torch(**recur_kwargs)
    torch.testing.assert_close(chunk_out.float(), recur_out.float(), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        chunk_kwargs["initial_state_source"].float(),
        recur_kwargs["initial_state_source"].float(),
        rtol=RTOL,
        atol=ATOL,
    )
    print(f"{c.label}: chunk and recurrent agree, rtol={RTOL} atol={ATOL}")


if __name__ == "__main__":
    run()