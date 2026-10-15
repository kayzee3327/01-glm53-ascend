"""Benchmark the Ascend C prepared_recurrent kernel against the Triton recurrence.

Toggle the module-level ``H`` below to pick a production head count:

* ``H = 4`` (TP16): the Ascend C 32-block launch (8 value tiles x 4 heads)
  against one Triton launch on grid (8, 1, 4).
* ``H = 8`` (TP8): four implementations compared in a single run.

The H=8 implementations, all covering heads 0..7 with BV=16:

``triton_loop_2x32``
    Two Triton launches on grid (8, 1, 4), HEAD_OFFSET 0 then 4. Mirrors the
    ``for head_offset in range(0, heads, 4)`` loop in sglang's
    ``run_prepared_prefill``. Baseline for the speedup column.
``ascendc_loop_2x32``
    Two sequential 32-block Ascend C launches, HEAD_OFFSET 0 then 4, through
    the v1 op. What the kernel does today, and the only Ascend C arm that
    measures against an unmodified build.
``ascendc_h8opt``
    The H8-optimized kernel. Not implemented yet; skipped until it exists.

The launch grid is a ``constexpr`` at the ``<<<>>>`` site, so unlike Triton it
cannot be chosen from Python. ``ascendc_h8opt`` therefore needs the library
rebuilt and the op registered with a trailing ``h8_variant`` selector (see
``H8_VARIANT_OPT``); it reports as skipped until then.
"""

import statistics
from dataclasses import dataclass, replace
from typing import Callable

from benchmark import (
    benchmark,
    BenchmarkConfig,
    print_results_table,
    NpuEventMeasurer,
    Reference,
)

import triton
import triton.language as tl
import torch
import torch_npu

device = "npu"

# Head count under test: 4 (TP16) or 8 (TP8).
H = 8

D = 128
BV = 16
# Every Ascend C kernel instantiation covers exactly 4 heads per launch.
HEADS_PER_LAUNCH = 4

# ``prepared_recurrent_v2(..., track_state, h8_variant)`` launch selectors.
# Values 1 (two 32-block launches) and 2 (one 64-block launch) are reserved for
# the host-side variants and are not benched here.
H8_VARIANT_AUTO = 0
H8_VARIANT_OPT = 3


@triton.jit
def _prepared_recurrent(
    QN,
    KN,
    V,
    GDECAY,
    BETA,
    OUT,
    STATE,
    INDICES,
    STARTS,
    TRACK_INDICES,
    TRACK_LENS,
    TRACK_STATE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    HEAD_OFFSET: tl.constexpr = 0,
):
    iv, req = tl.program_id(0), tl.program_id(1)
    head = HEAD_OFFSET + tl.program_id(2)
    bos = tl.load(STARTS + req).to(tl.int64)
    eos = tl.load(STARTS + req + 1).to(tl.int64)
    ks = tl.arange(0, D)
    vs = iv * BV + tl.arange(0, BV)
    slot = tl.load(INDICES + req)
    pstate = STATE + slot * S0 + head * S1 + ks[None, :] * SK + vs[:, None] * SV
    state = tl.zeros((BV, D), tl.float32)
    if slot > 0:
        state = tl.load(pstate).to(tl.float32)
    if TRACK_STATE:
        track_slot = tl.load(TRACK_INDICES + req)
        track_len = tl.load(TRACK_LENS + req)
        ptrack = (
            STATE + track_slot * S0 + head * S1 + ks[None, :] * SK + vs[:, None] * SV
        )
        if track_slot > 0 and track_len == 0:
            tl.store(ptrack, state)
    for i in range(eos - bos):
        row = (bos + i) * H + head
        q = tl.load(QN + row * D + ks)
        k = tl.load(KN + row * D + ks)
        v = tl.load(V + row * D + vs).to(tl.float32)
        decay = tl.load(GDECAY + row * D + ks)
        beta = tl.load(BETA + row)
        state *= decay[None, :]
        v -= tl.sum(state * k[None, :], axis=1)
        v *= beta
        state += k[None, :] * v[:, None]
        if TRACK_STATE:
            if track_slot > 0 and i + 1 == track_len:
                tl.store(ptrack, state)
        out = tl.sum(state * q[None, :], axis=1)
        tl.store(OUT + row * D + vs, out.to(OUT.dtype.element_ty))
    if slot > 0:
        tl.store(pstate, state)


from sgl_glm53_ascend_ops.ops import prepared_recurrent as _prepared_recurrent_torch


@dataclass
class Case:
    """One set of recurrence tensors. ``out``/``state`` are written in place."""

    qn: torch.Tensor
    kn: torch.Tensor
    v: torch.Tensor
    gdecay: torch.Tensor
    beta: torch.Tensor
    out: torch.Tensor
    state: torch.Tensor
    indices: torch.Tensor
    starts: torch.Tensor
    track_indices: torch.Tensor
    track_lens: torch.Tensor
    track_state: bool


def _triton_launch(c: Case):
    """One (8, 1, 4) grid per 4-head group, as sglang launches it."""
    for head_offset in range(0, H, HEADS_PER_LAUNCH):
        _prepared_recurrent[(triton.cdiv(D, BV), 1, HEADS_PER_LAUNCH)](
            c.qn,
            c.kn,
            c.v,
            c.gdecay,
            c.beta,
            c.out,
            c.state,
            c.indices,
            c.starts,
            c.track_indices,
            c.track_lens,
            c.track_state,
            H,
            D,
            BV,
            *c.state.stride(),
            HEAD_OFFSET=head_offset,
            num_warps=1,
            num_stages=3,
            multibuffer=True,
        )


def _ascendc_v1_launch(c: Case, head_offsets):
    """One 32-block launch per entry in ``head_offsets`` via the v1 op."""
    for head_offset in head_offsets:
        _prepared_recurrent_torch(
            qn=c.qn,
            kn=c.kn,
            v=c.v,
            gdecay=c.gdecay,
            beta=c.beta,
            out=c.out,
            state=c.state,
            indices=c.indices,
            starts=c.starts,
            track_indices=c.track_indices if c.track_state else None,
            track_lens=c.track_lens if c.track_state else None,
            track_state=c.track_state,
            head_offset=head_offset,
        )


def _ascendc_v2_launch(c: Case, h8_variant: int):
    """Single call into the consolidated v2 op, which owns the launch grid."""
    _v2_op()(
        c.qn,
        c.kn,
        c.v,
        c.gdecay,
        c.beta,
        c.out,
        c.state,
        c.indices,
        c.starts,
        c.track_indices if c.track_state else None,
        c.track_lens if c.track_state else None,
        c.track_state,
        h8_variant,
    )


def _v2_op():
    try:
        return torch.ops.sgl_glm53_ascend.prepared_recurrent_v2
    except AttributeError:
        return None


def _has_grid_selector() -> bool:
    """True once v2 is registered with the trailing ``h8_variant`` selector."""
    op = _v2_op()
    if op is None:
        return False
    try:
        schema = next(iter(op._schemas.values()))
    except (AttributeError, StopIteration):
        return False
    return any(arg.name == "h8_variant" for arg in schema.arguments)


def _loop_2x32(c: Case):
    """One 32-block launch per 4-head group, as the kernel ships today."""
    _ascendc_v1_launch(c, tuple(range(0, H, HEADS_PER_LAUNCH)))


@dataclass
class Impl:
    name: str
    launch: Callable[[Case], None]
    available: Callable[[], bool] = lambda: True
    reason: str = ""
    # The baseline every other implementation is validated and scaled against.
    baseline: bool = False


IMPLS_H4 = [
    Impl(name="triton", launch=_triton_launch, baseline=True),
    Impl(name="ascendc", launch=lambda c: _ascendc_v1_launch(c, (0,))),
]

IMPLS_H8 = [
    Impl(name="triton_loop_2x32", launch=_triton_launch, baseline=True),
    Impl(name="ascendc_loop_2x32", launch=_loop_2x32),
    Impl(
        name="ascendc_h8opt",
        launch=lambda c: _ascendc_v2_launch(c, H8_VARIANT_OPT),
        available=_has_grid_selector,
        reason="not implemented yet",
    ),
]


def impls():
    return IMPLS_H4 if H == 4 else IMPLS_H8


TOKEN_COUNTS_BY_H = {
    4: [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
    # H8 runs every implementation per point. Keep to the 128..16384 range
    # sglang supports.
    8: [1024, 2048, 4096, 8192, 16384],
}


def make_case(tokens: int, track_state: bool) -> Case:
    N = 3 if track_state else 2
    B = 1

    QN = torch.randn([B, tokens, H, D], device=device, dtype=torch.float32) * 0.01
    KN = torch.randn([B, tokens, H, D], device=device, dtype=torch.float32) * 0.01
    V = (
        torch.randn([B, tokens, H, D], device=device, dtype=torch.float32) * 0.01
    ).to(torch.bfloat16)
    GDECAY = torch.sigmoid(
        torch.randn([B, tokens, H, D], device=device, dtype=torch.float32)
    )
    BETA = torch.randn([B, tokens, H], device=device, dtype=torch.float32) * 0.01
    OUT = torch.empty([B, tokens, H, D], device=device, dtype=torch.bfloat16)
    STATE = torch.randn([N, H, D, D], device=device, dtype=torch.float32) * 0.01
    INDICES = torch.tensor([1], device=device, dtype=torch.int32)
    STARTS = torch.tensor([0, tokens], device=device, dtype=torch.int32)

    if track_state:
        TRACK_INDICES = torch.tensor([2], device=device, dtype=torch.int32)
        TRACK_LENS = torch.tensor([tokens // 2], device=device, dtype=torch.int32)
    else:
        TRACK_INDICES = torch.zeros([1], device=device, dtype=torch.int32)
        TRACK_LENS = torch.zeros([1], device=device, dtype=torch.int32)

    return Case(
        qn=QN,
        kn=KN,
        v=V,
        gdecay=GDECAY,
        beta=BETA,
        out=OUT,
        state=STATE,
        indices=INDICES,
        starts=STARTS,
        track_indices=TRACK_INDICES,
        track_lens=TRACK_LENS,
        track_state=track_state,
    )


def bench_one(tokens: int, track_state: bool = False):
    """Measure every available implementation on one shared set of inputs.

    Returns ``(name -> BenchmarkResult)`` with one entry per non-baseline
    implementation, each already compared against the shared baseline.
    """
    base = make_case(tokens, track_state)
    available = [impl for impl in impls() if impl.available()]
    baseline = next(impl for impl in available if impl.baseline)

    # Each implementation mutates its own out/state, so they all start from
    # identical values and stay in lockstep through the validation calls.
    def fresh():
        return replace(
            base, out=base.out.clone().detach(), state=base.state.clone().detach()
        )

    baseline_case = fresh()
    baseline_ref = Reference(
        baseline.name,
        lambda: baseline.launch(baseline_case),
        lambda: (baseline_case.out.clone(), baseline_case.state.clone()),
    )

    results = {}
    for impl in available:
        if impl.baseline:
            continue
        case = fresh()
        print(f">>> benchmarking H={H} tokens={tokens} {impl.name} ...")
        try:
            results[impl.name] = benchmark(
                # Bind impl/case per iteration; a bare closure would late-bind
                # to the last loop value.
                lambda impl=impl, case=case: impl.launch(case),
                references=[baseline_ref],
                measurer=NpuEventMeasurer(),
                config=BenchmarkConfig(10, 20, 20),
                output_fn=lambda case=case: (case.out.clone(), case.state.clone()),
            )
        except Exception as exc:  # keep the sweep going past a broken variant
            print(f"    failed: {type(exc).__name__}: {exc}")
    return results


def skipped_impls():
    return [
        (impl.name, impl.reason or "unavailable")
        for impl in impls()
        if not impl.available()
    ]


def _print_comparison(rows, baseline_name: str):
    """One row per token count, one column group per implementation."""
    names = []
    for _, results in rows:
        for name in results:
            if name not in names:
                names.append(name)

    headers = ["Tokens", f"{baseline_name} (ms)"]
    for name in names:
        headers += [f"{name} (ms)", "vs base"]

    table = []
    for label, results in rows:
        # The baseline is re-measured alongside each implementation, so every
        # ratio uses its own paired baseline run. The column reports the median
        # of those runs.
        base_runs = [
            res.references[0].measurement.median_ms
            for res in results.values()
            if res.references and res.references[0].measurement
        ]
        base_ms = statistics.median(base_runs) if base_runs else None
        row = [label, f"{base_ms:.4f}" if base_ms else "-"]
        for name in names:
            res = results.get(name)
            m = res.target.measurement if res else None
            if m is None:
                # No measurement means validation failed or the call raised.
                row += ["FAIL" if res else "-", "-"]
                continue
            paired = (
                res.references[0].measurement.median_ms
                if res.references and res.references[0].measurement
                else None
            )
            row.append(f"{m.median_ms:.4f}")
            row.append(f"{paired / m.median_ms:.2f}x" if paired else "-")
        table.append(row)

    widths = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells):
        return "  ".join(
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
            for i, cell in enumerate(cells)
        )

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in table:
        print(fmt(row))
    print()
    print(f"vs base : median({baseline_name}) / median(impl), >1 means impl is faster")


def run_sweep(track_state: bool):
    rows = []
    for tokens in TOKEN_COUNTS_BY_H[H]:
        label = f"{tokens // 1024}k" if tokens >= 1024 else str(tokens)
        results = bench_one(tokens, track_state=track_state)
        if results:
            rows.append((label, results))
    return rows


def report(title: str, rows):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    if not rows:
        print("no implementation produced measurements")
    elif H == 4:
        # Preserve the original single-reference table for H4.
        print_results_table([(label, r["ascendc"]) for label, r in rows])
    else:
        baseline = next(impl for impl in impls() if impl.baseline)
        _print_comparison(rows, baseline.name)

    skipped = skipped_impls()
    if skipped:
        print()
        print("Skipped:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")


def main():
    print(f"H = {H}, D = {D}, BV = {BV}")

    report(
        f"prepared_recurrent H={H} (no tracking)",
        run_sweep(track_state=False),
    )
    report(
        f"prepared_recurrent H={H} (track_state)",
        run_sweep(track_state=True),
    )


if __name__ == "__main__":
    main()
