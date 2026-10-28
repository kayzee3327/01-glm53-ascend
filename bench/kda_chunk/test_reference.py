"""Checks the chunked reference against a token-at-a-time recurrence.

The chunked formulation is where all the subtlety lives: the WY representation,
the triangular inverse, the chunk-local gate rebasing. A naive scan over tokens
has none of that, so agreement between the two is real evidence the chunked
algebra is right -- and it is the only check here that does not depend on any
kernel or on an NPU.

Run with ``python -m pytest test_reference.py`` from this directory, or just
``python test_reference.py`` for a plain pass/fail.
"""

from __future__ import annotations

import math

import torch

from reference import LN2, activate_gate, chunk_kda_reference


def kda_scan_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    cu_seqlens: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    safe_gate: bool = True,
    lower_bound: float = -5.0,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The delta rule, one token at a time, with no chunking anywhere.

    Per token: decay the state by the gate, then apply a rank-1 delta-rule
    update ``beta * (v - S^T k_decayed)`` against the *decayed* key, and read
    out with the query. This is the definition the chunked form is derived
    from.
    """
    t, h, kdim = q.shape
    hv, vdim = v.shape[1], v.shape[2]
    group = hv // h

    qf = q.to(dtype).repeat_interleave(group, dim=1)
    kf = k.to(dtype).repeat_interleave(group, dim=1)
    vf = v.to(dtype)
    bf = beta.to(dtype)
    gamma = activate_gate(
        g.to(dtype),
        A_log.to(dtype),
        dt_bias.to(dtype),
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=True,
    )

    seqlens = cu_seqlens.tolist()
    out = torch.zeros((t, hv, vdim), dtype=dtype, device=q.device)
    finals = []

    for seq in range(len(seqlens) - 1):
        start, end = seqlens[seq], seqlens[seq + 1]
        state = torch.zeros((hv, kdim, vdim), dtype=dtype, device=q.device)
        for i in range(start, end):
            decay = torch.exp(gamma[i])  # [HV, K], in (0, 1]
            state = decay.unsqueeze(-1) * state
            ki = kf[i].unsqueeze(1)  # [HV, 1, K]
            # delta = beta * (v - S^T k): the error of the current readout.
            pred = (ki @ state).squeeze(1)  # [HV, V]
            delta = bf[i].unsqueeze(-1) * (vf[i] - pred)
            state = state + kf[i].unsqueeze(-1) * delta.unsqueeze(1)
            out[i] = scale * (qf[i].unsqueeze(1) @ state).squeeze(1)
        finals.append(state)

    return out, torch.stack(finals)


def _case(tokens_per_seq, h=2, hv=2, kdim=32, vdim=32, seed=7, dtype=torch.float32):
    torch.manual_seed(seed)
    t = sum(tokens_per_seq)
    dev = "cpu"
    q = (torch.randn(t, h, kdim, device=dev) * 0.05).to(dtype)
    k = (torch.randn(t, h, kdim, device=dev) * 0.05).to(dtype)
    v = (torch.randn(t, hv, vdim, device=dev) * 0.05).to(dtype)
    g = (torch.randn(t, hv, kdim, device=dev) * 0.02 - 0.5).float()
    beta = torch.sigmoid(torch.randn(t, hv, device=dev)).to(dtype)
    A_log = torch.linspace(-0.5, 0.5, hv, dtype=torch.float32, device=dev)
    dt_bias = torch.linspace(-0.1, 0.1, hv * kdim, dtype=torch.float32, device=dev)
    cu = torch.tensor([0, *torch.tensor(tokens_per_seq).cumsum(0).tolist()])
    return q, k, v, g, beta, A_log, dt_bias, cu, kdim


def _compare(tokens_per_seq, chunk_size, **kw):
    q, k, v, g, beta, A_log, dt_bias, cu, kdim = _case(tokens_per_seq, **kw)
    scale = 1.0 / math.sqrt(kdim)
    common = dict(
        scale=scale, cu_seqlens=cu, A_log=A_log, dt_bias=dt_bias,
        safe_gate=True, lower_bound=-5.0,
    )
    got_o, got_s = chunk_kda_reference(
        q, k, v, g, beta, chunk_size=chunk_size, use_gate_in_kernel=True,
        dtype=torch.float64, **common,
    )
    want_o, want_s = kda_scan_reference(q, k, v, g, beta, **common)
    return got_o.double(), got_s.double(), want_o, want_s


def test_chunked_matches_scan_single_sequence():
    for chunk_size in (16, 32, 64):
        got_o, got_s, want_o, want_s = _compare((128,), chunk_size)
        torch.testing.assert_close(got_o, want_o, rtol=1e-6, atol=1e-9)
        torch.testing.assert_close(got_s, want_s, rtol=1e-6, atol=1e-9)


def test_chunked_matches_scan_ragged_tail():
    # Final chunk shorter than chunk_size, which is where gk_last indexing and
    # the masking of invalid rows is easy to get wrong.
    got_o, got_s, want_o, want_s = _compare((70, 33, 1), 32)
    torch.testing.assert_close(got_o, want_o, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(got_s, want_s, rtol=1e-6, atol=1e-9)


def test_chunked_matches_scan_gqa():
    got_o, got_s, want_o, want_s = _compare((96,), 32, h=2, hv=4)
    torch.testing.assert_close(got_o, want_o, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(got_s, want_s, rtol=1e-6, atol=1e-9)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
