"""Correctness and speedup for the GLM-5.3 KDA varlen recurrence op.

``glm_kda_varlen_recurrent_ascendc`` below is a copy of sglang's
``glm_kda_varlen_recurrent_npu`` with only the compute dispatch replaced by
``torch.ops.sgl_glm53_ascend.kda_varlen_recurrent``. Everything above that
point is kept verbatim so the two arms differ in the kernel and nothing else.

Regimes are the deployed parallel strategies, ``H = 64 / attn_tp_size``:
prefill at H=4/8 (TP16/TP8), decode at H=16/32/64 (TP4/TP2/TP1). A regime whose
kernel is not written yet is skipped, so implementing one width at a time shows
up as that row starting to report.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch
import torch_npu  # noqa: F401  registers the NPU device backend
import triton

# Match serving/test initialization before the first Triton compilation.
# Importing this after the first NPU JIT changes autotuner decoration.
import sglang.srt.layers.quantization  # noqa: F401
import sgl_glm53_ascend_ops  # noqa: F401  loads the compiled library
from sglang.srt.hardware_backend.npu.attention.glm53.kda_recurrent_npu import (
    glm_kda_varlen_recurrent_npu,
)

from benchmark import (
    BenchmarkConfig,
    NpuEventMeasurer,
    Reference,
    benchmark,
    make_comparator,
    print_results_table,
)

DEVICE = "npu:0"
D = 128
SCALE = D**-0.5
# The kernel rejects a non-negative bound.
LOWER_BOUND = -5.0

# Outputs here are small — |out| runs ~1e-4 with a ~1e-2 tail, because q and k
# are L2-normalized before the scale. So `atol` has to stay well under that or
# it passes everything on its own and the comparison means nothing; the real
# bar is `rtol`. The prepared prefill path is bit-exact against the generic
# kernel, so a correct Ascend C recurrence should be too: tighten both to 0
# once the kernel is expected to match exactly.
RTOL = 2e-3
ATOL = 1e-6

# Built via `make_comparator` so a failure reports its mismatch count against
# these tolerances rather than against the framework's defaults.
comparator = make_comparator(rtol=RTOL, atol=ATOL)

PREFILL_HEADS = (4, 8)  # TP16 / TP8
PREFILL_TOKENS = (1024, 2048, 4096, 8192, 16384)
DECODE_HEADS = (16, 32, 64)  # TP4 / TP2 / TP1
DECODE_BATCHES = (1, 32, 128)

CONFIG = BenchmarkConfig(warmup=10, rounds=20, batch_size=10)


def glm_kda_varlen_recurrent_ascendc(
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
    """``glm_kda_varlen_recurrent_npu`` with the Ascend C op doing the compute.

    Validation, tile arithmetic and output allocation are copied from upstream
    unchanged; only the kernel launch below differs. Keep them in sync.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or a.ndim != 4:
        raise ValueError("GLM KDA varlen tensors must have shape [1,T,H,D]")
    if q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1 or a.shape[0] != 1:
        raise ValueError("GLM KDA prefill expects one packed token batch")
    if (
        q.shape[:2] != k.shape[:2]
        or q.shape[1] != v.shape[1]
        or q.shape[1] != a.shape[1]
    ):
        raise ValueError("GLM KDA q/k/v/a token dimensions must match")
    if b.shape != v.shape[:-1]:
        raise ValueError("GLM KDA beta must have shape [1,T,HV]")

    num_q_heads, key_dim = k.shape[2:]
    num_value_heads, value_dim = v.shape[2:]
    if a.shape[2:] != (num_value_heads, key_dim):
        raise ValueError("GLM KDA gate must have shape [1,T,HV,K]")
    if num_value_heads % num_q_heads or num_value_heads % 2:
        raise ValueError("GLM KDA requires even HV divisible by H")
    if initial_state_indices.numel() != cu_seqlens.numel() - 1:
        raise ValueError("GLM KDA needs one state index per varlen sequence")
    if lower_bound is None or lower_bound >= 0:
        raise ValueError("GLM KDA requires a negative lower_bound")

    if (track_state_indices is None) != (track_state_lens is None):
        raise ValueError("GLM KDA prefix tracking needs both slots and lengths")
    if track_state_indices is not None:
        if not prefill or intermediate_state is not None:
            raise ValueError("GLM KDA prefix checkpoints are prefill-only")
        if (
            initial_state_source.ndim != 4
            or initial_state_source.shape[1:]
            != (num_value_heads, key_dim, value_dim)
            or initial_state_source.dtype != torch.float32
            or initial_state_source.device != q.device
            or initial_state_indices.device != q.device
        ):
            raise ValueError(
                "GLM KDA prefix checkpoints need the native FP32 state pool"
            )
        for t in (track_state_indices, track_state_lens):
            if (
                t.ndim != 1
                or t.numel() != initial_state_indices.numel()
                or t.dtype not in (torch.int32, torch.int64)
                or t.device != initial_state_indices.device
                or not t.is_contiguous()
            ):
                raise ValueError("Invalid GLM KDA prefix checkpoint metadata")

    block_k = triton.next_power_of_2(key_dim)
    if triton.cdiv(key_dim, block_k) != 1:
        raise NotImplementedError(
            "GLM KDA key dimensions spanning tiles are unsupported"
        )
    if scale is None:
        scale = key_dim**-0.5

    if intermediate_state is not None:
        if (
            intermediate_state.ndim != 5
            or intermediate_state.shape[0] < initial_state_indices.numel()
            or intermediate_state.shape[2:] != (num_value_heads, key_dim, value_dim)
            or intermediate_state.dtype != initial_state_source.dtype
        ):
            raise ValueError("Invalid GLM KDA per-token intermediate state layout")
    q, k, v, a, b = (t.contiguous() for t in (q, k, v, a, b))

    # Upstream allocates the output here and launches Triton. The op allocates
    # and returns its own, so this is the whole replacement.
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


@dataclass
class Case:
    """One recurrence invocation. ``state`` is written in place."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    state: torch.Tensor
    indices: torch.Tensor
    starts: torch.Tensor
    prefill: bool


def native_state(slots: int, heads: int) -> torch.Tensor:
    """The only state layout that reaches the kernel in production.

    Allocated contiguous as [slots,H,V,K] and viewed as [slots,H,K,V], which
    gives stride(2)=1, stride(3)=128. The backend normalizes both pools to it.
    """
    pool = torch.randn(slots, heads, D, D, device=DEVICE) * 0.1
    return pool.transpose(-1, -2)


def clone_state(state: torch.Tensor) -> torch.Tensor:
    """Copy a state pool preserving its [V,K] backing.

    A plain ``.clone()`` makes the view contiguous and silently selects a
    different Triton specialization.
    """
    return state.transpose(-1, -2).clone().transpose(-1, -2)


def make_case(heads: int, lengths: tuple[int, ...], prefill: bool) -> Case:
    tokens = sum(lengths)
    batch = len(lengths)
    tensors = {
        name: (torch.randn(1, tokens, heads, D, device=DEVICE) * 0.2).bfloat16()
        for name in ("q", "k", "v", "a")
    }
    return Case(
        **tensors,
        b=torch.randn(1, tokens, heads, device=DEVICE).bfloat16(),
        A_log=torch.randn(heads, device=DEVICE) * 0.1,
        dt_bias=torch.randn(heads * D, device=DEVICE) * 0.1,
        # Slot zero is the padding row and is never written.
        state=native_state(batch + 1, heads),
        indices=torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE),
        starts=torch.tensor(
            [sum(lengths[:i]) for i in range(batch + 1)],
            dtype=torch.int32,
            device=DEVICE,
        ),
        prefill=prefill,
    )


def call_kwargs(case: Case) -> dict:
    return dict(
        q=case.q,
        k=case.k,
        v=case.v,
        a=case.a,
        b=case.b,
        A_log=case.A_log,
        dt_bias=case.dt_bias,
        initial_state_source=case.state,
        initial_state_indices=case.indices,
        cu_seqlens=case.starts,
        lower_bound=LOWER_BOUND,
        scale=SCALE,
        prefill=case.prefill,
    )


def triton_launch(case: Case) -> torch.Tensor:
    return glm_kda_varlen_recurrent_npu(**call_kwargs(case))


def ascendc_launch(case: Case) -> torch.Tensor:
    return glm_kda_varlen_recurrent_ascendc(**call_kwargs(case))


def arm(launch, case: Case):
    """Bind a launch to its own case, capturing the output for validation."""
    box = {}

    def run():
        box["out"] = launch(case)

    def output():
        return box["out"], case.state

    return run, output


def fresh(case: Case) -> Case:
    """An arm-private copy, so every arm starts from identical values."""
    return replace(case, state=clone_state(case.state))


def bench_one(case: Case):
    baseline_run, baseline_output = arm(triton_launch, fresh(case))
    target_run, target_output = arm(ascendc_launch, fresh(case))
    return benchmark(
        target=target_run,
        references=[Reference("triton", baseline_run, baseline_output)],
        comparator=comparator,
        measurer=NpuEventMeasurer(),
        config=CONFIG,
        output_fn=target_output,
    )


def regimes():
    for heads in PREFILL_HEADS:
        for tokens in PREFILL_TOKENS:
            yield f"prefill H{heads} T{tokens // 1024}k", heads, (tokens,), True
    for heads in DECODE_HEADS:
        for batch in DECODE_BATCHES:
            # Decode is one token per sequence, so T == batch.
            yield f"decode  H{heads} B{batch}", heads, (1,) * batch, False


def unimplemented(case: Case) -> bool:
    """True if this regime's kernel is still a stub, so the row is skipped."""
    try:
        ascendc_launch(fresh(case))
        torch.npu.synchronize()
        return False
    except RuntimeError as exc:
        if "not implemented yet" in str(exc):
            return True
        raise


def main():
    print(f"tolerance: rtol={RTOL} atol={ATOL}")
    print()

    rows = []
    skipped = []
    for label, heads, lengths, prefill in regimes():
        torch.manual_seed(1979 + heads + sum(lengths))
        case = make_case(heads, lengths, prefill)
        if unimplemented(case):
            skipped.append(label)
            continue
        print(f">>> {label} ...")
        rows.append((label, bench_one(case)))
    print()

    if rows:
        print_results_table(rows)
    else:
        print("no kernels implemented yet")

    if skipped:
        print()
        print(f"skipped, kernel not implemented: {', '.join(skipped)}")


if __name__ == "__main__":
    torch.npu.set_device(DEVICE)
    main()
