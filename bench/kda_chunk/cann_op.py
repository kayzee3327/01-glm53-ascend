"""Loads the built ``ChunkKdaFwd`` custom op for the bench's CANN arm.

Why a loader rather than a plain import: the installed toolkit is CANN 9.0.0,
which does not ship this operator, so its aclnn entry point comes from a custom
vendor package compiled from the 9.2.0 sources -- see ``build_cann_op.py``.

The repo's ``ACLNN_CMD`` macro resolves the symbol with ``dlsym`` at first call,
searching the custom vendor packages' ``libcust_opapi.so`` *before* the built-in
``libopapi.so``. That is what makes the cross-version build work without
touching the installed toolkit. The search covers ``ASCEND_CUSTOM_OPP_PATH`` and
``LD_LIBRARY_PATH``, neither of which CANN's ``set_env.sh`` points at the
vendors directory, so this module sets ``ASCEND_CUSTOM_OPP_PATH`` itself when it
finds an installed package. Doing it here rather than in a shell script to be
sourced means the bench runs correctly straight after a build, with no extra
step to forget.

Everything above ``load`` is the same code as the recurrent bench's
``cann_op.py``, deliberately: the two benches are separately runnable, so each
carries its own copy rather than importing across directories. Fix a bug in one
and fix it in the other.

Where they differ is the binding itself. ``chunk_kda_fwd`` ships no
``torch_extension/`` wrapper upstream -- unlike ``recurrent_kda``, which has one
and only needs the wheel installed to reach it -- so there is no ready-made
Python entry point to import. This module supplies the missing one, reusing
ops-transformer's own ``OpBuilder`` machinery so the binding is built and loaded
the same way every other operator in that repo is: a small C++ bridge,
JIT-compiled by ``torch.utils.cpp_extension.load``, calling ``aclnnChunkKdaFwd``
through the repo's ``ACLNN_CMD`` macro.

The operator computes several backward intermediates that this bench does not
compare; the ones that are optional are not requested, but ``Aqk`` and ``Akk``
are mandatory outputs and are allocated and written regardless. That cost is
real and stays in the measurement -- it is part of what the official kernel
does.

A missing package is reported, not raised: the bench drops the arm and prints
why, rather than failing the rows that do not need it.
"""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("OPS_TRANSFORMER_DIR", "/data/wangkaize/ops-transformer"))

# No vendor name here: the package is found by looking for the operator's own
# aclnn symbol, not by knowing which vendor dir holds it. A second copy of the
# name next to `build_cann_op.py`'s would only be a thing to let drift.

_HINT = "run `python build_cann_op.py` first"


def _vendor_dirs() -> list[Path]:
    """Candidate custom op vendor packages, most specific first.

    ``ASCEND_CUSTOM_OPP_PATH`` wins if already set, so an out-of-tree install
    can be pointed at explicitly; otherwise the toolkit's own ``opp/vendors``
    is scanned, which is where ``build_cann_op.py`` installs.
    """
    roots: list[Path] = [
        Path(entry)
        for entry in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
        if entry.strip()
    ]

    opp = os.environ.get("ASCEND_OPP_PATH")
    if not opp:
        home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
            "ASCEND_TOOLKIT_HOME"
        )
        opp = str(Path(home) / "opp") if home else None
    if opp:
        vendors = Path(opp) / "vendors"
        if vendors.is_dir():
            roots.extend(sorted(p for p in vendors.iterdir() if p.is_dir()))

    return roots


def _exports(lib: Path, symbol: str) -> bool:
    """Whether ``lib`` defines ``symbol``."""
    try:
        out = subprocess.run(
            ["nm", "-D", "--defined-only", str(lib)],
            capture_output=True, text=True, check=False,
        ).stdout
    except OSError:
        # No nm: assume it is the right library and let dlsym report the truth.
        return True
    return any(
        line.split()[-1] == symbol for line in out.splitlines() if symbol in line
    )


def available(symbol: str) -> tuple[bool, str]:
    """Whether a vendor package exporting ``symbol`` is installed, and why not.

    Checks the aclnn symbol by name rather than settling for the first vendor
    directory that holds a ``libcust_opapi.so``. The sibling bench installs one
    of those for a *different* operator, and it sorts first, so a
    first-match-wins scan picks the wrong package -- the op then looks available
    and fails at the first launch with "not in libopapi.so".
    """
    dirs = _vendor_dirs()
    if not dirs:
        return False, f"no custom op vendor packages installed; {_HINT}"
    libs = [lib for vendor in dirs for lib in vendor.rglob("libcust_opapi.so")]
    if not libs:
        names = [d.name for d in dirs]
        return False, f"vendor packages {names} hold no libcust_opapi.so; {_HINT}"
    if not any(_exports(lib, symbol) for lib in libs):
        names = [d.name for d in dirs]
        return False, f"vendor packages {names} export no {symbol}; {_HINT}"
    _expose(libs)
    return True, ""


def _expose(libs: list[Path]) -> None:
    """Put every vendor package on the paths ``ACLNN_CMD``'s dlsym search reads.

    All of them, not just the one exporting this op: ``GetCustOpApiHandlers``
    opens every entry on the two variables and caches the resulting handler list
    in a function-local ``static``, built on the first aclnn call in the process.
    So whatever is on the path at that moment is all this process will ever
    search -- naming one vendor directory here would leave a second custom op
    unreachable for the rest of the run.

    Must happen before that first call, not before process start: the macro
    resolves lazily, so setting these from Python is early enough.
    """
    values: list[tuple[str, str]] = []
    for lib in libs:
        # libcust_opapi.so sits at <vendor>/op_api/lib/; ASCEND_CUSTOM_OPP_PATH
        # wants the vendor root and appends op_api/lib itself, LD_LIBRARY_PATH
        # wants the directory as-is.
        values.append(("ASCEND_CUSTOM_OPP_PATH", str(lib.parent.parent.parent)))
        values.append(("LD_LIBRARY_PATH", str(lib.parent)))

    for var, value in values:
        current = os.environ.get(var, "")
        entries = current.split(os.pathsep) if current else []
        if value not in entries:
            os.environ[var] = os.pathsep.join([value, *entries]) if entries else value


def aclnn_symbol(name: str) -> str:
    """The aclnn entry point for op ``name``: ``chunk_kda_fwd`` -> ``aclnnChunkKdaFwd``."""
    return "aclnn" + "".join(part.capitalize() for part in name.split("_"))


_BRIDGE = r"""
// Torch bridge for aclnnChunkKdaFwd. Mirrors the shape of ops-transformer's
// own torch_extension bridges; chunk_kda_fwd has none upstream.
#include <string>
#include <tuple>
#include <vector>

#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {

// Only the outputs this benchmark reads are requested. The optional ones
// (gk, w, u, qg, kg, v_new, h) are passed as undefined tensors, which the L2
// interface treats as "do not export". Aqk and Akk are not optional.
std::tuple<at::Tensor, at::Tensor> chunk_kda_fwd(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &g, const at::Tensor &beta,
    const c10::optional<at::Tensor> &aLog,
    const c10::optional<at::Tensor> &dtBias,
    const c10::optional<at::Tensor> &initialState,
    const c10::optional<at::Tensor> &cuSeqlens,
    std::string layout, double scale, int64_t chunkSize,
    bool safeGate, double lowerBound, bool useGateInKernel, bool stateVFirst)
{
    TORCH_CHECK(q.dim() == 3, "chunk_kda_fwd here expects TND inputs");
    const int64_t T = q.size(0);
    const int64_t K = q.size(2);
    const int64_t HV = v.size(1);
    const int64_t V = v.size(2);
    const int64_t C = chunkSize;

    // One logical sequence per cu_seqlens interval.
    TORCH_CHECK(cuSeqlens.has_value(), "chunk_kda_fwd here requires cu_seqlens");
    const at::Tensor cu = cuSeqlens.value().to(at::kCPU).to(at::kLong);
    const int64_t N = cu.numel() - 1;
    const int64_t *cuData = cu.data_ptr<int64_t>();

    // cu_seqlens crosses as an aclIntArray, not a tensor, so it must be a
    // host-side buffer that outlives the call below.
    const std::vector<int64_t> cuVec(cuData, cuData + cu.numel());
    const at::IntArrayRef cuRef(cuVec);

    auto optsIn = q.options();
    auto optsF32 = q.options().dtype(at::kFloat);

    at::Tensor attnOut = at::empty({T, HV, V}, optsIn);
    at::Tensor finalState = stateVFirst
        ? at::empty({N, HV, V, K}, optsF32)
        : at::empty({N, HV, K, V}, optsF32);
    // Mandatory outputs, head-major.
    at::Tensor aqk = at::empty({HV, T, C}, optsIn);
    at::Tensor akk = at::empty({HV, T, C}, optsIn);

    // An undefined tensor converts to a null aclTensor*, which is how the L2
    // interface is told to skip an optional output. An empty optional does the
    // same for chunk_indices, which the op derives from cu_seqlens when absent.
    const at::Tensor none;
    const c10::optional<at::IntArrayRef> noChunkIndices = c10::nullopt;

    ACLNN_CMD(aclnnChunkKdaFwd, q, k, v, g, beta, aLog, dtBias, initialState,
              cuRef, noChunkIndices, layout, scale, chunkSize,
              safeGate, lowerBound, useGateInKernel, stateVFirst,
              attnOut, finalState, /*gkOut=*/none, aqk, akk,
              /*wOut=*/none, /*uOut=*/none, /*qgOut=*/none, /*kgOut=*/none,
              /*vNewOut=*/none, /*hOut=*/none);

    return std::make_tuple(attnOut, finalState);
}

} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("chunk_kda_fwd", &op_api::chunk_kda_fwd, "aclnnChunkKdaFwd (TND)");
}
"""


@functools.lru_cache(maxsize=1)
def _load_module(name: str):
    """JIT-compile the bridge against the repo's headers and the vendor libs.

    Uses ops-transformer's ``OpBuilder`` only for its include and link paths --
    getting those right by hand is the fiddly part, and the repo already
    encodes the ordering constraints (notably that the CANN headers must
    precede torch_npu's older vendored copies).
    """
    import logging
    import sys

    import torch_npu  # noqa: F401
    from torch.utils.cpp_extension import load

    # Importing the source-tree package for `OpBuilder` alone also runs its
    # `ops/__init__.py`, which tries to alias every operator in the repo to its
    # legacy module name and warns once per failure -- 54 lines of
    # "Failed to register legacy module" ahead of the first benchmark row. They
    # are expected rather than a problem: a single-op checkout contains none of
    # those modules, and this bench wants none of them. Raised to ERROR, not
    # silenced, so a real failure still prints.
    #
    # The sibling bench never sees this: its wheel is vendor-renamed, so the
    # guard `if __name__ == "cann_ops_transformer.ops"` in that file is false
    # and the loop does not run at all.
    logging.getLogger("cann_ops_transformer.ops").setLevel(logging.ERROR)

    sys.path.insert(0, str(REPO / "torch_extension"))
    from cann_ops_transformer.op_builder import OpBuilder

    class _Bridge(OpBuilder):
        def __init__(self):
            super().__init__(name, category=None)

        def sources(self):
            return []

        def schema(self):
            return None

        def register_meta(self):
            return None

    bridge = _Bridge()
    bridge.ensure_initialized()

    build_dir = HERE / ".build_cann_op"
    build_dir.mkdir(exist_ok=True)
    source = build_dir / f"{name}_bridge.cpp"
    if not source.is_file() or source.read_text() != _BRIDGE:
        source.write_text(_BRIDGE)

    return load(
        name=f"{name}_bench_bridge",
        sources=[str(source)],
        extra_include_paths=bridge.include_paths(),
        extra_cflags=bridge.cxx_args(),
        extra_ldflags=bridge.extra_ldflags(),
        build_directory=str(build_dir),
        verbose=False,
    )


@functools.lru_cache(maxsize=None)
def load(name: str) -> tuple[Optional[Any], Optional[str]]:
    """Return (callable, None) for op ``name``, or (None, reason) if absent.

    ``reason`` is a one-line explanation suitable for printing next to a
    skipped benchmark arm. Same signature as the recurrent bench's ``load``,
    though the callable on the other side of it is this module's own bridge
    rather than a wheel's entry point.

    Cached, and that is not just tidiness: ``available`` shells out to ``nm``
    once per vendor library, so an uncached call from inside the benchmark's
    timed loop adds ~68ms of subprocess to every measurement -- which reads as
    a flat, shape-independent latency for the whole arm.
    """
    ok, why = available(aclnn_symbol(name))
    if not ok:
        return None, why
    try:
        module = _load_module(name)
    except Exception as exc:  # the bridge compile, which has its own failures
        return None, f"building the {name} torch bridge failed: {exc}"
    return getattr(module, name), None


def chunk_kda_fwd_cann(
    q,
    k,
    v,
    g,
    beta,
    *,
    scale: float,
    chunk_size: int,
    cu_seqlens,
    A_log=None,
    dt_bias=None,
    initial_state=None,
    safe_gate: bool = True,
    lower_bound: float = -5.0,
    use_gate_in_kernel: bool = True,
    state_v_first: bool = False,
):
    """``(attn_out, final_state)`` from the official kernel, for TND input.

    Signature matches ``chunk_kda_triton`` so the bench arms are
    interchangeable. Raises if the op is unavailable -- the bench's arm probe
    turns that into a skipped column.
    """
    op, why = load("chunk_kda_fwd")
    if op is None:
        raise RuntimeError(why)
    return op(
        q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens,
        "TND", float(scale), int(chunk_size),
        bool(safe_gate), float(lower_bound), bool(use_gate_in_kernel),
        bool(state_v_first),
    )
