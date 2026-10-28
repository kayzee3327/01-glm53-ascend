"""Checks the Triton chunked KDA against the verified torch reference.

Small shapes on purpose: these are correctness cases, and `reference.py` is
O(C^2 K) per chunk and slow. The bench script covers the deployed shapes.
"""

from __future__ import annotations

import math

import torch
import torch_npu  # noqa: F401  registers the NPU backend

from reference import chunk_kda_reference
from triton_kda_chunk import chunk_kda_triton

DEVICE = "npu:0"

# The Triton arm accumulates in fp32 and the reference in fp64, over different
# summation orders, so these are rounding bars rather than exactness bars.
RTOL, ATOL = 2e-2, 2e-4


def _case(tokens_per_seq, h=2, hv=2, kdim=128, vdim=128, seed=11,
          dtype=torch.bfloat16):
    torch.manual_seed(seed)
    t = sum(tokens_per_seq)
    q = (torch.randn(t, h, kdim) * 0.05).to(dtype).npu()
    k = (torch.randn(t, h, kdim) * 0.05).to(dtype).npu()
    v = (torch.randn(t, hv, vdim) * 0.05).to(dtype).npu()
    g = (torch.randn(t, hv, kdim) * 0.02 - 0.5).float().npu()
    beta = torch.sigmoid(torch.randn(t, hv)).to(dtype).npu()
    A_log = torch.linspace(-0.5, 0.5, hv, dtype=torch.float32).npu()
    dt_bias = torch.linspace(-0.1, 0.1, hv * kdim, dtype=torch.float32).npu()
    cu = torch.tensor(
        [0, *torch.tensor(tokens_per_seq).cumsum(0).tolist()], dtype=torch.int64
    ).npu()
    return dict(
        q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias,
        cu_seqlens=cu, scale=1.0 / math.sqrt(kdim),
    )


def _on_cpu(case):
    """The reference runs on CPU: the NPU has no fp64, and fp64 is the point."""
    return {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in case.items()
    }


def _check(tokens_per_seq, chunk_size, label, **kw):
    case = _case(tokens_per_seq, **kw)
    got_o, got_s = chunk_kda_triton(chunk_size=chunk_size, **case)
    want_o, want_s = chunk_kda_reference(
        chunk_size=chunk_size, dtype=torch.float64, **_on_cpu(case)
    )
    torch.testing.assert_close(
        got_o.float().cpu(), want_o.float(), rtol=RTOL, atol=ATOL,
        msg=lambda s: f"{label}: attn_out\n{s}",
    )
    torch.testing.assert_close(
        got_s.float().cpu(), want_s.float(), rtol=RTOL, atol=ATOL,
        msg=lambda s: f"{label}: final_state\n{s}",
    )
    print(f"{label}: PASS")


def main():
    _check((256,), 64, "single sequence, C=64")
    _check((256,), 128, "single sequence, C=128")
    # Ragged: a short tail chunk and a length-1 sequence.
    _check((130, 65, 1), 64, "ragged varlen, C=64")
    # GQA: four value heads over two query heads.
    _check((192,), 64, "GQA HV=4 H=2", h=2, hv=4)
    # Unbounded gate (softplus branch).
    case = _case((128,))
    got_o, _ = chunk_kda_triton(chunk_size=64, safe_gate=False, **case)
    want_o, _ = chunk_kda_reference(
        chunk_size=64, safe_gate=False, dtype=torch.float64, **_on_cpu(case)
    )
    torch.testing.assert_close(
        got_o.float().cpu(), want_o.float(), rtol=RTOL, atol=ATOL
    )
    print("unbounded gate (softplus): PASS")


if __name__ == "__main__":
    torch.npu.set_device(DEVICE)
    main()
