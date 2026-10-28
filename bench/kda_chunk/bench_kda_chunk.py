"""Correctness and latency for the chunked KDA prefill forward.

Three kernels compute the same thing, and this script compares them on the
shapes GLM-5.3 prefill actually runs:

``cann``     the official ``ChunkKdaFwd`` from CANN 9.2.0's ops-transformer.
             The installed toolkit here is 9.0.0, which does not ship this
             operator at all, so it is built from source as a custom op
             package -- see ``build_cann_op.py``, which owns the build and is
             also runnable on its own.
``triton``   ``triton_kda_chunk.py``, written for this comparison.
``fla``      ``chunk_kda_fwd`` from flash-linear-attention-npu (Tianjin
             University), installed as a wheel plus a run package. Supports
             ``chunk_size=64`` only, so the C128 rows report ``-``.
``ascendc``  the kernel in the sglang GLM-5.3 plugin. Not implemented yet; the
             arm is wired up and reports ``-`` until the op appears.

The target is whichever arm you are working on -- ``ascendc``, so speedups read
as "ours vs theirs" once it exists. With no Ascend C op present the script falls
back to measuring ``triton`` as the target so the other comparisons still run.

There are no command-line options: every parameter is a module constant below,
so a run is reproducible from the file alone and two runs are always comparable.
To measure something else, edit the constants.

Correctness is not decided between the three arms. Each is checked against
``reference.py``, a direct transcription of the operator's documented formulas
that is itself verified against a token-at-a-time recurrence in
``test_reference.py``. An arm that disagrees with the reference is reported as
FAIL rather than quietly becoming the baseline.

Shapes come from the deployed parallel strategies, ``H = 64 / attn_tp_size``:
prefill at H=4 (TP16) and H=8 (TP8), over the prefill token counts the
recurrent bench also uses, so the two scripts' rows line up.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# Operator development on this machine is restricted to physical card 4, and the
# pin has to happen before torch_npu initializes -- `DEVICE` below is "npu:0",
# which is an index into the *visible* set, so this makes it card 4. Set here
# rather than left to the caller's shell because forgetting it silently
# benchmarks on someone else's card.
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "4")

# --- vendor selection, before anything touches CANN ----------------------
#
# `cann` and `fla` cannot both run in one process. Both vendor packages
# register the *same* op type, `ChunkKdaFwd`, and CANN's launcher registry is
# keyed by that name, so the second registration loses: `cann` segfaults, or
# `fla` raises "ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaFwd failed", depending on
# which resolved first. Path ordering does not help -- there is one slot for
# that name and two packages claiming it.
#
# So the arm is chosen per process by `KDA_CHUNK_VENDOR`, and `main` re-runs
# this script once per vendor so a single command still measures both. The
# selection has to happen here, at import time: the dlsym handler list is built
# on the first aclnn call and cached in a function-local static for the life of
# the process, and `fla_npu`'s own .pth prepends its vendor at interpreter
# start, before any user code runs.
VENDOR_ENV = "KDA_CHUNK_VENDOR"
VENDORS = ("cann", "fla")

_CANN_VENDOR_DIRS = (
    "/usr/local/Ascend/cann-9.0.0/opp/vendors/glm53chunk_transformer",
    "/usr/local/Ascend/cann-9.0.0/opp/vendors/glm53recurrent_transformer",
)
_FLA_VENDOR_DIR = Path(
    "/data/wangkaize/.vsglang/lib/python3.11/site-packages/fla_npu"
    "/opp/vendors/fla_npu_transformer"
)


def _select_vendor(name: str) -> None:
    """Point the aclnn dlsym search at exactly one of the two packages.

    Setting `ASCEND_CUSTOM_OPP_PATH` explicitly rather than appending, because
    the point is to make the *other* package unreachable -- `cann_op.available`
    scans `opp/vendors` only when this variable is unset, and would otherwise
    find and expose both.
    """
    if name == "cann":
        # fla_npu's .pth would re-add its vendor at every interpreter start.
        os.environ["FLA_NPU_DISABLE_PTH"] = "1"
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(_CANN_VENDOR_DIRS)
    else:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(
            [str(_FLA_VENDOR_DIR), str(_FLA_VENDOR_DIR / "op_api" / "lib")]
        )


VENDOR = os.environ.get(VENDOR_ENV, VENDORS[0])
if VENDOR not in VENDORS:
    raise SystemExit(f"{VENDOR_ENV} must be one of {VENDORS}, got {VENDOR!r}")
_select_vendor(VENDOR)

import torch  # noqa: E402  must follow the device pin and vendor selection
import torch_npu  # noqa: F401,E402  registers the NPU device backend

from benchmark import (  # noqa: E402
    BenchmarkConfig,
    NpuEventMeasurer,
    Reference,
    benchmark,
    make_comparator,
)

from reference import chunk_kda_reference  # noqa: E402
from triton_kda_chunk import chunk_kda_triton  # noqa: E402

# Index into the visible set, which the pin above narrows to physical card 4.
DEVICE = "npu:0"
D = 128  # head dim, K = V = 128 for GLM-5.3
SCALE = D**-0.5
# The deployed gate: bounded, activated inside the kernel.
LOWER_BOUND = -5.0
SAFE_GATE = True
USE_GATE_IN_KERNEL = True

# bf16 inputs accumulated in fp32, against a reference that accumulates in fp64.
#
# Which of the two bounds actually binds, measured at H4 T1024 C64 rather than
# assumed: `|attn_out|` runs ~4e-5 mean (median 2.2e-5, max 1.4e-3), so 97.8% of
# its elements sit under `atol` and every element is below the `atol/rtol`
# crossover at 1e-2. For `attn_out` it is therefore `atol` that decides, not
# `rtol`. `final_state` is the larger tensor (mean 8.4e-4, max 1.9e-2) and only
# 29% of it falls under `atol`, so that is where `rtol` does the work.
#
# The pair is still not vacuous -- zeroing or halving `attn_out`, or zeroing
# `final_state`, each produce FAIL -- but a regime that only perturbed the small
# bulk of `attn_out` would need `atol` tightened to be caught.
RTOL = 2e-2
ATOL = 2e-4
comparator = make_comparator(rtol=RTOL, atol=ATOL)

PREFILL_HEADS = (4, 8)  # TP16 / TP8
PREFILL_TOKENS = (1024, 2048, 4096, 8192, 16384)
# The operator supports 64 and 128.
CHUNK_SIZES = (64, 128)

# The arm speedups are quoted against, or None to pick it automatically:
# `ascendc` when that kernel exists, else `triton` so the CANN comparison still
# has a baseline.
TARGET: Optional[str] = None

CONFIG = BenchmarkConfig(warmup=5, rounds=20, batch_size=5)


@dataclass(frozen=True)
class Case:
    """One benchmark shape, as a packed single-sequence prefill."""

    heads: int
    tokens: int
    chunk_size: int

    @property
    def label(self) -> str:
        return f"prefill H{self.heads:<2d} T{self.tokens:<6d} C{self.chunk_size}"


@dataclass
class Inputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    cu_seqlens: torch.Tensor

    def as_kwargs(self) -> dict:
        return dict(
            q=self.q, k=self.k, v=self.v, g=self.g, beta=self.beta,
            A_log=self.A_log, dt_bias=self.dt_bias, cu_seqlens=self.cu_seqlens,
            scale=SCALE, safe_gate=SAFE_GATE, lower_bound=LOWER_BOUND,
            use_gate_in_kernel=USE_GATE_IN_KERNEL,
        )


def make_inputs(case: Case, seed: int = 20261026) -> Inputs:
    """TND varlen inputs for one packed sequence, on device.

    Scaled the way the pytest suite in ops-transformer scales them: small q/k/v
    and a gate centred near -0.5, which is what the deployed activation sees.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    t, h = case.tokens, case.heads

    def rand(*shape, scale=0.05):
        return (torch.randn(*shape, generator=gen) * scale).to(torch.bfloat16).npu()

    return Inputs(
        q=rand(t, h, D),
        k=rand(t, h, D),
        v=rand(t, h, D),
        g=(torch.randn(t, h, D, generator=gen) * 0.02 - 0.5).float().npu(),
        beta=torch.sigmoid(torch.randn(t, h, generator=gen)).to(torch.bfloat16).npu(),
        A_log=torch.linspace(-0.5, 0.5, h, dtype=torch.float32).npu(),
        dt_bias=torch.linspace(-0.1, 0.1, h * D, dtype=torch.float32).npu(),
        cu_seqlens=torch.tensor([0, t], dtype=torch.int64).npu(),
    )


# --- the three arms -------------------------------------------------------
#
# Each returns (attn_out, final_state): attn_out as [T, HV, V] in the input
# dtype, final_state as fp32 [N, HV, K, V]. Anything an arm computes beyond
# those two is ignored, so an arm that also emits backward intermediates is not
# penalised for it in the comparison -- but it does pay for them in latency,
# which is noted in the table legend.


def run_triton(inputs: Inputs, case: Case):
    return chunk_kda_triton(chunk_size=case.chunk_size, **inputs.as_kwargs())


def run_cann(inputs: Inputs, case: Case):
    """The official kernel, via the built custom op package."""
    from cann_op import chunk_kda_fwd_cann

    return chunk_kda_fwd_cann(chunk_size=case.chunk_size, **inputs.as_kwargs())


def run_fla(inputs: Inputs, case: Case):
    """flash-linear-attention-npu's ``chunk_kda_fwd``.

    Reached through the installed wheel, so there is no build step here and no
    loader like ``cann_op.py`` -- see ``README.md`` for the install.

    Two details of the call, both the wrapper's rather than the kernel's:

    * ``cu_seqlens`` is passed as a host list. The wrapper guards on
      ``if cu_seqlens and not chunk_indices``, which calls ``bool()`` on a
      tensor and raises "Boolean value of Tensor ... is ambiguous"; a list
      takes the same path the op wants and derives the canonical
      ``chunk_indices`` from it. Upstream's own
      ``npu_chunk_gated_delta_rule_fwd_prepare`` converts tensors for exactly
      this reason, so this is a missed call site, not an interface choice.
    * The return is a 12-tuple of forward and backward intermediates; this
      bench reads only the first two, matching the other arms. The optional
      backward exports are left unrequested, but ``gk``/``Aqk``/``Akk`` are
      computed regardless, the same way the CANN arm pays for its mandatory
      outputs.

    ``beta`` is already sigmoid-activated by ``make_inputs`` and q/k are not
    L2-normalized, so the in-kernel switches for both stay off -- which is also
    what keeps this on the op's plain path: turning either on would require
    ``chunk_size=64`` *and* bf16 *and* K=V=128 together.
    """
    # Imported here rather than at module scope: under `KDA_CHUNK_VENDOR=cann`
    # this package's operators are deliberately off the search path, and
    # importing it anyway warns at length about an OPP it cannot reach.
    from fla_npu.ops.ascendc import chunk_kda_fwd as fla_chunk_kda_fwd

    kwargs = inputs.as_kwargs()
    cu_seqlens = [int(x) for x in kwargs["cu_seqlens"].cpu().tolist()]
    attn_out, final_state, *_ = fla_chunk_kda_fwd(
        kwargs["q"],
        kwargs["k"],
        kwargs["v"],
        kwargs["g"],
        kwargs["beta"],
        kwargs["scale"],
        case.chunk_size,
        layout="TND",
        cu_seqlens=cu_seqlens,
        A_log=kwargs["A_log"],
        dt_bias=kwargs["dt_bias"],
        use_gate_in_kernel=kwargs["use_gate_in_kernel"],
        safe_gate=kwargs["safe_gate"],
        lower_bound=kwargs["lower_bound"],
        output_final_state=True,
        state_v_first=False,
    )
    return attn_out, final_state


def run_ascendc(inputs: Inputs, case: Case):
    """The sglang GLM-5.3 plugin kernel.

    Not implemented. When the op lands, dispatch to it here and return
    (attn_out, final_state); the rest of the script needs no change.
    """
    raise NotImplementedError(
        "Ascend C chunk KDA kernel is not implemented yet"
    )


def run_reference(inputs: Inputs, case: Case):
    """The fp64 anchor. Runs on CPU, because the NPU has no fp64.

    fp64 describes the *accumulation*, not the result that comes back:
    ``reference.py`` rounds ``attn_out`` to the input dtype before returning, so
    the anchor arrives in bf16 and the rounding alone costs up to ~3.9e-3
    relative (measured at H4 T1024), about a fifth of the ``rtol`` budget.
    ``final_state`` is unaffected -- it stays fp32.

    Kept rather than fixed, because ``torch.testing.assert_close`` requires
    matching dtypes, so returning fp64 here would mean every arm's output had to
    be upcast at the comparison site. The budget it eats is accounted for in the
    tolerances above.
    """
    kwargs = {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.as_kwargs().items()
    }
    return chunk_kda_reference(
        chunk_size=case.chunk_size, dtype=torch.float64, **kwargs
    )


ARMS: dict[str, Callable[[Inputs, Case], tuple]] = {
    "ascendc": run_ascendc,
    "triton": run_triton,
    "cann": run_cann,
    "fla": run_fla,
}

# What to say when an arm cannot run, instead of a traceback. The CANN arm's
# own message already explains how to build it.
UNAVAILABLE_MARKERS = (
    "not implemented",
    "no custom op vendor packages",
    "no libcust_opapi.so",
    # A vendor package is installed but it is the sibling bench's operator, not
    # this one.
    "export no aclnnChunkKdaFwd",
)


def wrong_vendor(name: str) -> Optional[str]:
    """Why arm ``name`` is not reachable in *this* process, or None.

    Checked before the arm is probed, because probing a deselected arm is the
    failure mode this exists to prevent: the loser of the `ChunkKdaFwd`
    registration does not raise, it segfaults.
    """
    if name in VENDORS and name != VENDOR:
        return f"not selected: {VENDOR_ENV}={VENDOR}"
    return None


def unsupported_shape(name: str, case: Case) -> Optional[str]:
    """Why arm ``name`` does not cover ``case``, or None if it does.

    Separate from ``arm_status``, which asks whether an arm exists at all and
    answers once per run from one probe shape. Support can also depend on the
    shape, and an arm that is installed and working for most rows must not be
    dropped from all of them -- nor silently crash on the rows it does not
    cover, which is what a single-shape probe would allow.
    """
    if name == "fla" and case.chunk_size != 64:
        # `aclnn_chunk_kda_fwd.cpp` rejects anything else in parameter
        # validation ("chunkSize only supports 64."), and its api.md documents
        # that as the supported range rather than a gap. So this is scope, not
        # a bug to report per row.
        return "supports chunk_size=64 only"
    return None


def arm_status(name: str, case: Case) -> Optional[str]:
    """``None`` if the arm runs, else a one-line reason it does not.

    Probing by running it once: an arm that is not built, or not written, is a
    skipped column rather than a failed benchmark.
    """
    try:
        ARMS[name](make_inputs(case), case)
        torch.npu.synchronize()
        return None
    except NotImplementedError as exc:
        return str(exc).splitlines()[0]
    except (RuntimeError, ImportError, OSError) as exc:
        message = str(exc)
        for marker in UNAVAILABLE_MARKERS:
            if marker in message:
                return message.splitlines()[0]
        raise


def bench_one(case: Case, target_name: str, reference_names: list[str]):
    """One row: every live arm validated against fp64, then all of them timed.

    Validation is done here rather than through the framework's
    target-vs-reference comparison, because correctness is not a question about
    the target: each arm is checked against ``reference.py`` independently, so a
    wrong *reference* arm is reported as wrong instead of silently becoming the
    standard the target is held to.

    That leaves the framework doing what it is good at -- warmup, repeats and
    paired statistics -- and keeps the measurement meaningful even when the
    target is the only live arm.
    """
    inputs = make_inputs(case)
    expected = tuple(t.float() for t in run_reference(inputs, case))

    # An arm that does not cover this shape is left out of both the validation
    # and the measurement, so its cells read `-` on this row and stay populated
    # on the rows it does cover.
    live = [
        name for name in reference_names if unsupported_shape(name, case) is None
    ]

    validity = {}
    for name in [target_name, *live]:
        got = tuple(t.float().cpu() for t in ARMS[name](inputs, case))
        try:
            comparator(got, expected)
            validity[name] = True
        except AssertionError:
            validity[name] = False

    result = benchmark(
        target=functools.partial(ARMS[target_name], inputs, case),
        references=[
            Reference(name=name, fn=functools.partial(ARMS[name], inputs, case))
            for name in live
        ],
        # Validation already happened above, against fp64. Passing a comparator
        # that accepts anything keeps the framework from re-deciding
        # correctness by comparing two arms to each other.
        comparator=lambda actual, expected_: None,
        measurer=NpuEventMeasurer(),
        config=CONFIG,
    )
    return result, validity


def print_table(rows, target_name: str, reference_names: list[str]):
    """One row per shape, one latency/speedup pair per reference arm.

    The framework's ``print_results_table`` renders only the first reference,
    which would hide an arm here, so the table is assembled directly.
    """
    headers = [
        "Benchmark", f"{target_name} (ms)", "MAD", f"{target_name} ok",
    ]
    for name in reference_names:
        headers += [f"{name} (ms)", f"vs {name}", f"{name} ok"]

    table = []
    for label, (result, validity) in rows:
        target = result.target.measurement
        row = [
            label,
            f"{target.median_ms:.4f}" if target else "-",
            f"{target.mad_ms:.4f}" if target else "-",
            "PASS" if validity.get(target_name) else "FAIL",
        ]
        by_name = {ref.name: ref for ref in result.references}
        for name in reference_names:
            ref = by_name.get(name)
            measurement = ref.measurement if ref else None
            if measurement is None:
                row += ["-", "-", "-"]
                continue
            row.append(f"{measurement.median_ms:.4f}")
            row.append(
                f"{measurement.median_ms / target.median_ms:.2f}x"
                if target and target.median_ms > 0 else "-"
            )
            row.append("PASS" if validity.get(name) else "FAIL")
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

    legend = [
        (f"{target_name} (ms)", "median latency of the target arm"),
        ("MAD", "median absolute deviation of that latency"),
        ("<arm> ok", "agrees with the fp64 reference at the tolerance above"),
        ("vs <arm>", "median(arm) / median(target), >1 means target is faster"),
    ]
    arms = [
        ("cann", "CANN 9.2.0 ChunkKdaFwd, built from source as a custom op\n"
                 "package; also writes the mandatory Aqk/Akk intermediates,\n"
                 "which stays in its measured time"),
        ("triton", "triton_kda_chunk.py, five kernels over the same data"),
        ("fla", "flash-linear-attention-npu chunk_kda_fwd, from the installed\n"
                "wheel; also writes gk/Aqk/Akk, which stays in its measured\n"
                "time. '-' on the C128 rows: it supports chunk_size=64 only"),
        ("ascendc", "the sglang GLM-5.3 plugin kernel, not yet implemented"),
    ]
    # Describe only the arms this table has columns for. The deselected vendor
    # is explained once, in the skip line above the table, so repeating it here
    # would read as a column that failed to print.
    arms = [
        (name, text) for name, text in arms
        if name == target_name or name in reference_names
    ]
    pad = max(len(name) for name, _ in [*legend, *arms])

    print()
    for name, text in legend:
        print(f"{name.ljust(pad)} : {text}")
    print()
    for name, text in arms:
        head, *rest = text.split("\n")
        print(f"{name.ljust(pad)} : {head}")
        for line in rest:
            print(f"{' ' * pad}   {line}")


def run_every_vendor() -> int:
    """Re-run this script once per vendor, in a fresh process each time.

    What makes a subprocess necessary rather than fussy: the losing
    `ChunkKdaFwd` registration cannot be undone within a process, and the
    handler list the dlsym search uses is cached in a function-local static on
    the first aclnn call. So "both arms in one run" has to mean two processes,
    and this is the thing that makes `python bench_kda_chunk.py` still do it.

    Each child prints its own full table. `triton` and the target arm run in
    both, which is what keeps the two tables comparable: the vendor-specific
    arm changes, the baseline does not.
    """
    import subprocess
    import sys

    failed = 0
    for i, vendor in enumerate(VENDORS):
        if i:
            print()
        print("=" * 72)
        print(f"=== {VENDOR_ENV}={vendor}")
        print("=" * 72)
        print(flush=True)
        completed = subprocess.run(
            [sys.executable, os.path.abspath(__file__)],
            # Setting the variable is also what tells the child it is the
            # measurement rather than another driver.
            env={**os.environ, VENDOR_ENV: vendor},
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if completed.returncode != 0:
            failed += 1
            print(f"\n{VENDOR_ENV}={vendor} exited {completed.returncode}")
    return failed


def main() -> None:
    print(f"tolerance: rtol={RTOL} atol={ATOL}  (each arm vs the fp64 reference)")
    print(f"vendor: {VENDOR}  ({VENDOR_ENV}; the other chunk arm is off the path)")

    # A small shape decides which arms are live, so an unavailable arm costs
    # one cheap launch rather than one per row. Each arm is probed on a shape
    # it actually supports, so "is this arm installed?" is never answered by a
    # shape it declines -- which would report a working arm as missing, and
    # with a message about chunk sizes rather than about the install.
    def probe_case(name: str) -> Optional[Case]:
        for chunk_size in CHUNK_SIZES:
            case = Case(heads=PREFILL_HEADS[0], tokens=256, chunk_size=chunk_size)
            if unsupported_shape(name, case) is None:
                return case
        return None

    status = {}
    for name in ARMS:
        reason = wrong_vendor(name)
        if reason is not None:
            status[name] = reason
            continue
        probe = probe_case(name)
        if probe is None:
            status[name] = "covers none of the configured chunk sizes"
            continue
        status[name] = arm_status(name, probe)

    live = [name for name, reason in status.items() if reason is None]
    skipped = {name: reason for name, reason in status.items() if reason is not None}

    target_name = TARGET
    if target_name is None:
        target_name = "ascendc" if "ascendc" in live else "triton"
    if target_name not in live:
        print()
        print(f"target arm '{target_name}' is unavailable: {skipped[target_name]}")
        if target_name == "ascendc":
            print("measuring 'triton' as the target instead.")
            target_name = "triton"
        else:
            raise SystemExit(1)

    reference_names = [name for name in ("cann", "triton", "fla", "ascendc")
                       if name in live and name != target_name]

    print(f"target: {target_name}")
    print(f"references: {', '.join(reference_names) or 'none'}")
    if skipped:
        print()
        for name, reason in skipped.items():
            print(f"skipped {name}: {reason}")
    print()

    if not reference_names:
        # Still worth running: the target's own correctness and latency are
        # measured against the fp64 reference, which is the part that does not
        # depend on another arm existing.
        print("no other arm is available; measuring the target on its own.")
        print("build the CANN op with `python build_cann_op.py` to compare.")
        print()

    rows = []
    for chunk_size in CHUNK_SIZES:
        for heads in PREFILL_HEADS:
            for tokens in PREFILL_TOKENS:
                case = Case(heads=heads, tokens=tokens, chunk_size=chunk_size)
                torch.manual_seed(1979 + heads + tokens + chunk_size)
                print(f">>> {case.label} ...")
                rows.append(
                    (case.label, bench_one(case, target_name, reference_names))
                )
    print()
    print_table(rows, target_name, reference_names)


if __name__ == "__main__":
    # Invoked bare: drive one child per vendor, so a plain run still covers
    # both chunk arms. With the variable already set -- by the driver below or
    # by hand -- this process is the measurement and runs a single vendor.
    if VENDOR_ENV not in os.environ:
        raise SystemExit(run_every_vendor())

    torch.npu.set_device(DEVICE)
    main()
