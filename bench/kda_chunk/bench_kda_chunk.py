"""Correctness and latency for the chunked KDA prefill forward.

Target: the Ascend C kernel in the sglang GLM-5.3 plugin, written here from
scratch rather than wrapped around someone else's.

``ascendc``  the kernel under development, through
             ``sgl_glm53_ascend_ops.ops.kda_varlen_chunk``. Takes the caller's
             raw ``[1, T, H, D]`` inputs and fuses the preparation the other
             two need on the host; chunk size 64 only, since it is a
             compile-time constant of the kernel.
``cann``     CANN 9.2.0's ``ChunkKdaFwd``, via ``ref_ops.cann``.
``fla``      flash-linear-attention-npu's, via ``ref_ops.fla_npu``.
``torch``    the correctness anchor, left blank to be written. Not a measured
             arm: it validates, it is never timed, and it is free to be as slow
             and as literal as it needs to be.

The two outside arms come from the ``ref_ops`` package, which exists so they can
load in one process: both upstreams otherwise claim the op type ``ChunkKdaFwd``,
CANN's launcher registry is keyed by that string, and the loser used to segfault.
``ref_ops`` resolves that by shipping *renamed* copies of the CANN-side ops, so
no vendor selection or re-exec is needed here -- importing it is the whole
mechanism. See ``ref_ops/AGENTS.md``.

Inputs follow ``bench_kda_varlen_recurrent.py``'s ``make_case`` exactly: same
shapes, same dtypes, same distributions, same native state-pool layout. The two
benchmarks measure the two halves of the same deployed prefill, so a shape that
disagreed between them would make the comparison meaningless.

Every arm is called with those inputs *raw* and unmodified: rank-4
``[1, T, H, D]`` as ``BSND``, un-normalized q/k, raw gate, raw beta logits, and
the state pool in its own K-contiguous orientation. ``ascendc`` and ``fla`` fuse
all of the preparation; ``cann``'s operator has no L2-norm or beta-sigmoid
attribute to fuse it with, so that one arm does those two steps on the host
inside its timed region. ``FUSION`` records the difference and the table prints
it -- the alternative, preparing on the host for everyone, charged the two
reference arms for work the target does on-chip and understated them by up to
24%.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Callable, Optional

# Default npu:4
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "4")

import torch  # noqa: E402  must follow the device pin
import torch_npu  # noqa: F401,E402  registers the NPU device backend

from benchmark import (  # noqa: E402
    BenchmarkConfig,
    BenchmarkResult,
    ImplementationResult,
    NpuEventMeasurer,
    Reference,
    ReferenceResult,
    benchmark,
    make_comparator,
)

# Imported at module scope, and deliberately not inside the arms: `ref_ops`
# builds the aclnn dlsym search path, and `GetCustOpApiHandlers` caches that
# path in a function-local static on the *first* operator launch in the process.
# Whatever is on it at that moment is all the process will ever search, so an
# import deferred until after the target arm has run would leave both reference
# vendors unreachable for the rest of the run.
#
# A failure here is a skipped column, not a dead script -- the target does not
# need this package -- so the reason is kept and reported by `arm_status`.
try:
    from ref_ops import cann as ref_cann, fla_npu as ref_fla  # noqa: E402
except Exception as exc:  # pragma: no cover - depends on the install
    ref_cann = ref_fla = None
    REF_OPS_ERROR = f"ref_ops is not importable: {exc}"
else:
    REF_OPS_ERROR = None

# The target. Imported here for the same reason and with the same contract as
# `ref_ops` above: importing the package is what loads the compiled library, a
# missing build is a skipped column rather than a dead script, and the order
# matters -- this import must not be the first thing to touch the aclnn path.
try:
    from sgl_glm53_ascend_ops.ops import kda_varlen_chunk  # noqa: E402
except Exception as exc:  # pragma: no cover - depends on the build
    kda_varlen_chunk = None
    ASCENDC_ERROR = f"sgl_glm53_ascend_ops is not importable: {exc}"
else:
    ASCENDC_ERROR = None

# Index into the visible set, which the pin above narrows to physical card 4.
DEVICE = "npu:0"
D = 128  # head dim, K = V = 128 for GLM-5.3
SCALE = D**-0.5
# The deployed gate: bounded, activated inside the kernel.
LOWER_BOUND = -5.0
SAFE_GATE = True
USE_GATE_IN_KERNEL = True

# bf16 inputs accumulated in fp32, against a reference expected to accumulate
# in fp64.
#
# Which of the two bounds actually binds was measured on these inputs at H4
# T1024 C64: `|attn_out|` runs ~5e-4 mean with a ~1.6e-2 max, so most of it sits
# under `atol` and `atol` is what decides there. `final_state` is larger (mean
# ~5e-3) and is where `rtol` does the work.
#
# The size of the disagreement these have to admit is measured, not guessed:
# `cann` and `fla` on identical inputs differ by ~1 bf16 ulp, and `cann` against
# the deployed Triton recurrence in `bench_kda_varlen_recurrent.py` differs by
# 6.1e-5 on `attn_out` and 4.4e-4 on `final_state` -- a different summation
# order over the same recurrence, not a different answer. `atol` has to clear
# that and still be well under the 5e-4 signal, which is what sets it here.
#
# The pair is not vacuous -- zeroing or halving `attn_out`, or zeroing
# `final_state`, each produced FAIL -- but a regime that only perturbed the
# small bulk of `attn_out` would need `atol` tightened to be caught.
RTOL = 2e-2
ATOL = 2e-4
comparator = make_comparator(rtol=RTOL, atol=ATOL)

PREFILL_HEADS = (8,)  # TP16 / TP8
PREFILL_TOKENS = (1024, 2048, 4096,)
# The operator supports 64 and 128.
CHUNK_SIZES = (64,)

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
    """One invocation's tensors, in the recurrence benchmark's layout.

    ``q``/``k``/``v``/``a`` are ``[1, T, H, D]`` and ``b`` is ``[1, T, H]``,
    which is the caller's packed-prefill shape and the shape
    ``bench_kda_varlen_recurrent.py`` measures. Every arm takes them at this
    rank, as ``BSND``. ``a`` is the *raw* gate and ``b`` the *raw* beta: the
    activation and the sigmoid happen downstream, in the kernel wherever the
    operator can do it, never here.

    ``dt_bias`` is flat ``[HV*K]``: the form both reference arms require, and
    the form the Triton recurrence indexes as ``i_hv * K + o_k``.

    The last three are the reference's optional prefill outputs and are ``None``
    in the measured regime: the chunk operators have no equivalent, and
    ``intermediate`` being absent is what makes the recurrence write its final
    state to the pool at all. ``make_inputs`` builds them on request, for
    checking a chunked kernel's state at the chunk boundaries.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    state: torch.Tensor
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    intermediate: Optional[torch.Tensor] = None
    track_indices: Optional[torch.Tensor] = None
    track_lens: Optional[torch.Tensor] = None
    chunk_size: Optional[int] = None

    def to(self, device) -> "Inputs":
        """The same tensors on ``device``, for the fp64 anchor on the host.

        The pool moves through its transpose because a plain ``.to`` returns it
        contiguous, dropping the ``[V, K]`` backing ``native_state`` builds.
        """
        def move(name):
            t = getattr(self, name)
            if t is None:
                return None
            # Same trailing layout one axis deeper for `intermediate`.
            if name in ("state", "intermediate"):
                return t.transpose(-1, -2).to(device).transpose(-1, -2)
            return t.to(device)

        return Inputs(**{n: move(n) for n in self.__dataclass_fields__})

    def recurrence_kwargs(self) -> dict:
        """These tensors as ``kda_python.kda_recur_torch``'s keyword arguments.

        ``state``/``indices`` take their deployed names, ``dt_bias`` becomes the
        ``[HV, K]`` view, and the scalars come from the module constants.

        The three optionals pass through as built. All ``None`` is the measured
        regime, and is also the only setting under which the final state reaches
        the pool -- the recurrence takes that branch on ``intermediate is
        None``, and upstream rejects tracking unless ``prefill`` is set.

        The pool is written in place, so clone it first if you need it intact.
        """
        heads = self.q.shape[2]
        return dict(
            A_log=self.A_log,
            a=self.a,
            dt_bias=self.dt_bias.view(heads, D),
            q=self.q,
            k=self.k,
            v=self.v,
            b=self.b,
            initial_state_source=self.state,
            initial_state_indices=self.indices,
            cu_seqlens=self.cu_seqlens,
            lower_bound=LOWER_BOUND,
            scale=SCALE,
            intermediate_state=self.intermediate,
            prefill=True,
            track_state_indices=self.track_indices,
            track_state_lens=self.track_lens,
        )

    def chunk_kwargs(self) -> dict:
        heads = self.q.shape[2]
        return dict(
            A_log=self.A_log,
            a=self.a,
            dt_bias=self.dt_bias.view(heads, D),
            q=self.q,
            k=self.k,
            v=self.v,
            b=self.b,
            initial_state_source=self.state,
            initial_state_indices=self.indices,
            cu_seqlens=self.cu_seqlens,
            lower_bound=LOWER_BOUND,
            scale=SCALE,
            prefill=True,
            track_state_indices=self.track_indices,
            track_state_lens=self.track_lens,
            chunk_size=self.chunk_size
        )

def native_state(slots: int, heads: int) -> torch.Tensor:
    """The only state layout that reaches the kernel in production.

    Allocated contiguous as [slots,H,V,K] and viewed as [slots,H,K,V], which
    gives stride(2)=1, stride(3)=128. Copied from the recurrence benchmark,
    where the stride is load-bearing: a plain contiguous pool silently selects a
    different Triton specialization there. The chunk operators take their
    initial state as a normal contiguous tensor and do not care, but the pool is
    built the same way so both benchmarks start from the same bytes.
    """
    pool = torch.randn(slots, heads, D, D, device=DEVICE) * 0.1
    return pool.transpose(-1, -2)


def native_intermediate(slots: int, steps: int, heads: int) -> torch.Tensor:
    """The per-token checkpoint buffer as the backend hands it over.

    [slots, steps, HV, K, V] with stride(3)=1 -- the same K-contiguous
    orientation as the state pool, one axis deeper. Noise rather than zeros so a
    step the kernel fails to write shows as a mismatch instead of matching an
    untouched zero.
    """
    pool = torch.randn(slots, steps, heads, D, D, device=DEVICE) * 0.1
    return pool.transpose(-1, -2)


def clone_state(state: torch.Tensor) -> torch.Tensor:
    """A copy of the pool that keeps its ``[V, K]`` backing.

    A plain ``.clone()`` makes the view contiguous, which in the recurrence
    benchmark silently selects a different Triton specialization. Needed here
    for a different reason: the recurrence reference writes the final state into
    the pool in place, and every arm on a row reads the one pool ``make_inputs``
    produced, so an arm that mutates it has to be given a copy.
    """
    return state.transpose(-1, -2).clone().transpose(-1, -2)


def make_inputs(
    case: Case,
    seed: int = 20261026,
    checkpoints: bool = False,
    track_len: Optional[int] = None,
    chunk: bool = False
) -> Inputs:
    """Inputs for one packed single-sequence prefill, on device.

    Shapes, dtypes and distributions are ``make_case`` from
    ``bench_kda_varlen_recurrent.py``, down to the 0.2 and 0.1 scale factors and
    the padding row at slot zero. That file measures the recurrence this kernel
    replaces, so the two have to agree on their inputs or the comparison between
    them says nothing. The only thing added here is ``chunk_size``, which lives
    on ``Case``.

    Both optional arguments default off, which is the measured regime -- they
    exist for debugging a chunked kernel against the recurrence at intermediate
    points, where "the final states agree" is not the question:

    ``checkpoints``  every per-token state, as [1, T, HV, K, V]. Suppresses the
                     final write to the pool, so ``final_state`` has to be read
                     out of the last step instead.
    ``track_len``    one state, after this many tokens, into pool slot 2. A
                     chunk boundary is the useful value. Mutually exclusive
                     with ``checkpoints`` upstream.
    ``chunk``        whether to use chunk kda
    """
    torch.manual_seed(seed)
    t, h = case.tokens, case.heads
    if checkpoints and track_len is not None:
        raise ValueError("prefix tracking and per-token checkpoints are exclusive")

    tensors = {
        name: (torch.randn(1, t, h, D, device=DEVICE) * 0.2).bfloat16()
        for name in ("q", "k", "v", "a")
    }
    tracking = track_len is not None
    return Inputs(
        **tensors,
        b=torch.randn(1, t, h, device=DEVICE).bfloat16(),
        A_log=torch.randn(h, device=DEVICE) * 0.1,
        dt_bias=torch.randn(h * D, device=DEVICE) * 0.1,
        # Slot zero is the padding row and is never written. One sequence, so
        # one live slot -- plus a second for the tracked checkpoint, which has
        # to be distinct from every live source slot.
        state=native_state(3 if tracking else 2, h),
        indices=torch.ones(1, dtype=torch.int32, device=DEVICE),
        cu_seqlens=torch.tensor([0, t], dtype=torch.int32, device=DEVICE),
        intermediate=native_intermediate(1, t, h) if checkpoints else None,
        track_indices=(
            torch.full((1,), 2, dtype=torch.int32, device=DEVICE) if tracking else None
        ),
        track_lens=(
            torch.full((1,), track_len, dtype=torch.int32, device=DEVICE)
            if tracking
            else None
        ),
        chunk_size=case.chunk_size if chunk else None
    )


# --- the arms, and the anchor they are checked against --------------------
#
# Every arm returns (attn_out, final_state): attn_out as [T, HV, V] in the input
# dtype, final_state as fp32 [N, HV, K, V].
#
# Every arm also *carries* its state in the one layout the Ascend C op accepts
# -- the pool's own, K fastest, which the reference ops spell `state_v_first`.
# That is the orientation `native_state` allocates and the orientation the
# deployed backend normalizes both of its pools to, so it is the only one in
# which all four arms are handed the bytes the kernel reads. Before this, the
# reference arms were gathered into a contiguous [N, HV, K, V] instead, which
# fed them a transposed copy of the pool rather than the pool.
#
# This is a layout correctness point, not a speed one: the repack measured
# 0.025 ms per call at H8 against medians of 1-7 ms, and removing it moved no
# arm outside its MAD. It is worth having because the pool orientation is
# exactly the kind of thing the two reference kernels could have specialized
# on -- `state_v_first` is a tiling attribute, not a host-side transpose -- and
# an arm reading a layout the deployment never produces is measuring the wrong
# kernel path for free.


def native_live_state(state: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """``state``'s live slots as the contiguous ``[N, HV, V, K]`` it already is.

    ``native_state`` allocates ``[slots, HV, V, K]`` and views it
    ``[slots, HV, K, V]``, so ``K`` is the fastest axis. That orientation is
    what the Ascend C op requires and checks for, and it is what the reference
    ops call ``state_v_first``: a contiguous ``[N, HV, V, K]``. Gathering
    through the V-first view is therefore a plain slice copy of the bytes the
    kernel reads, where gathering a contiguous ``[N, HV, K, V]`` -- which is
    what indexing the pool's own view produces -- hands the operator a
    transposed copy of the pool instead of the pool.

    Only the two reference arms need this. ``ascendc`` and ``run_torch`` take
    the pool itself and write their state back into it, so they read their
    result out as a strided view and convert nothing.
    """
    return state.transpose(-1, -2)[indices.long()].contiguous()


def k_first(final_state: torch.Tensor) -> torch.Tensor:
    """A V-first state back in the ``[N, HV, K, V]`` arm contract.

    A transpose, so a view rather than a copy: the bytes are already the
    pool's, and the arms only differ in which axis they call last.
    """
    return final_state.transpose(-1, -2)


def host_l2norm(q: torch.Tensor, k: torch.Tensor):
    """``q``/``k`` L2-normalized in fp32, rank preserved.

    ``kda_pre_torch``'s formula, kept to its association: the epsilon is inside
    the square root, and the reduction is a plain ``sum`` over the last axis.

    Only ``cann`` calls this. Its operator has no in-kernel normalization under
    any flag -- seven attributes, none of them an L2 norm or an epsilon -- so
    the host is the only place it can happen for that arm. ``ascendc`` and
    ``fla`` both fuse it; the asymmetry is reported rather than evened out, in
    ``FUSION``.
    """
    qf, kf = q.float(), k.float()
    qf = qf / torch.sqrt((qf * qf).sum(-1, keepdim=True) + 1e-6)
    kf = kf / torch.sqrt((kf * kf).sum(-1, keepdim=True) + 1e-6)
    return qf.bfloat16(), kf.bfloat16()


def run_ascendc(inputs: Inputs, case: Case):
    """The sglang GLM-5.3 plugin kernel, ``kda_varlen_chunk``.

    No host preparation: this op takes the caller's packed ``[1, T, H, D]``
    tensors and the raw gate and beta directly, and does the L2 norms, the
    sigmoids and the bounded gate in its AIV prologue. ``fla`` is called the
    same way and fuses the same three; ``cann`` fuses only the gate, because
    its operator has no attribute for the other two. Whatever an arm cannot
    fuse stays in its timed region, which is what makes fusing it count.

    Two conversions remain, both of them layout rather than math:

    * ``dt_bias`` back to ``[HV, K]``. The op accepts either, the flat form is
      what ``Inputs`` carries for the reference arms.
    * The state pool is cloned. The op updates it in place, where the chunk
      references return a new tensor, so the arm has to be given a copy or
      successive timed calls would each start from the previous call's final
      state. ``clone_state`` keeps the K-contiguous backing the op requires.

    Returned as ``(attn_out, final_state)`` like the other arms: the output
    squeezed to ``[T, HV, V]``, and the live slots as a strided fp32
    ``[N, HV, K, V]`` view of the pool. A view, not a copy -- this op writes
    the state where the pool already keeps it, so there is nothing to convert
    on the way out. The reference arms reach the same layout by taking their
    contiguous ``[N, HV, V, K]`` state back through ``k_first``.
    """
    if kda_varlen_chunk is None:
        raise NotImplementedError(ASCENDC_ERROR)
    state = clone_state(inputs.state)
    heads = inputs.q.shape[2]
    out = kda_varlen_chunk(
        A_log=inputs.A_log,
        a=inputs.a,
        dt_bias=inputs.dt_bias.view(heads, D),
        q=inputs.q,
        k=inputs.k,
        v=inputs.v,
        b=inputs.b,
        initial_state_source=state,
        initial_state_indices=inputs.indices,
        cu_seqlens=inputs.cu_seqlens,
        lower_bound=LOWER_BOUND,
        scale=SCALE,
        prefill=True,
        track_state_indices=inputs.track_indices,
        track_state_lens=inputs.track_lens,
    )
    return out.squeeze(0), state[inputs.indices.long()].float()


def run_torch(inputs: Inputs, case: Case):
    """The correctness anchor: ``kda_python.kda_chunk_torch``, on the host.

    A literal transcription of the chunked formulas -- it builds the full
    ``[BT, BT]`` ``A`` column by column and inverts it by forward substitution,
    with O(C^2 K) temporaries -- which is affordable because this is never
    timed. It takes the raw gate and raw beta and does the activation, the
    sigmoid and the L2 norms itself, in ``kda_pre_torch``, so it shares no
    preparation code with the measured arms -- which is what makes it an
    independent check on them rather than a restatement of one.

    Runs on the NPU, in fp32. Not a compromise on either count:

    * fp32 is what the reference is, by construction rather than by choice of
      device -- ``kda_chunk_torch`` calls ``.float()`` on q/k/a/b and allocates
      its accumulator ``torch.float32``, so handing it fp64 would downcast at
      the first line. Comparing fp32 against fp32 with a different summation
      order is the question these tolerances were set for.
    * On device it stays on the same inputs as the arms, bit for bit. A host
      copy would be the same values, but the anchor is also the only thing
      standing between a layout bug and a PASS, so there is no reason to move
      it off the pool the kernel reads.

    ``v`` is passed as fp32 purely for the return dtype: ``kda_chunk_torch``
    casts its output back to ``v.dtype`` on the way out, and the bf16 round
    trip would cost up to ~3.9e-3 relative for nothing. The upcast is exact and
    the function upcasts everything internally anyway, so the arithmetic is
    unchanged.

    The state pool is cloned for the same reason as in ``run_ascendc``: this
    reference writes the final state into it in place, and every arm on a row
    reads the one pool ``make_inputs`` produced.
    """
    from kda_python import kda_chunk_torch

    kwargs = inputs.chunk_kwargs()
    # `chunk_kwargs` reads `chunk_size` off `Inputs`, which `make_inputs`
    # populates only under `chunk=True`. The benchmark builds its inputs
    # without it, so it comes from the case instead -- one source either way.
    kwargs["chunk_size"] = case.chunk_size
    kwargs["initial_state_source"] = clone_state(inputs.state)
    kwargs["v"] = inputs.v.float()

    out = kda_chunk_torch(**kwargs)
    return out.squeeze(0), kwargs["initial_state_source"][inputs.indices.long()]


def run_cann(inputs: Inputs, case: Case):
    """CANN 9.2.0's ``ChunkKdaFwd``, through ``ref_ops``' renamed copy.

    Takes the caller's rank-4 ``[1, T, H, D]`` tensors as ``BSND``, the same
    ones ``ascendc`` gets, and fuses the one thing this operator can fuse: the
    bounded gate, from raw ``a`` with ``A_log`` and ``dt_bias``.

    The q/k L2 norm and the beta sigmoid both stay on the host, because this
    operator cannot do either. Its attribute set is seven entries and none is a
    normalization, an epsilon, or a sigmoid; ``beta`` is consumed as a plain
    multiplicative coefficient on every path, so raw logits here would silently
    compute the wrong thing rather than be activated. That is a real capability
    gap against the other two arms, so it is reported rather than evened out --
    see ``FUSION``, which the table prints.

    ``BSND`` rather than ``TND`` is worth 9% at T1024, fading to ~0 by T4096.
    The L2 normalizes every layout to head-major before launching, and that
    costs one materialized transpose per tensor it moves: ``TND`` moves five,
    ``BSND`` moves only ``beta``. Bit-identical either way, measured.

    ``state_v_first`` reads and writes the state in the pool's own
    K-contiguous orientation, which is the layout the Ascend C op requires --
    see ``native_live_state``. Also measured bit-identical to the K-first
    spelling, so it buys the shared layout and nothing else.
    """
    if ref_cann is None:
        raise NotImplementedError(REF_OPS_ERROR)
    q, k = host_l2norm(inputs.q, inputs.k)
    attn_out, final_state = ref_cann.chunk_kda(
        q, k, inputs.v, inputs.a, torch.sigmoid(inputs.b.float()),
        layout="BSND",
        scale=SCALE, chunk_size=case.chunk_size, cu_seqlens=inputs.cu_seqlens,
        A_log=inputs.A_log, dt_bias=inputs.dt_bias,
        initial_state=native_live_state(inputs.state, inputs.indices),
        safe_gate=SAFE_GATE, lower_bound=LOWER_BOUND,
        use_gate_in_kernel=USE_GATE_IN_KERNEL, state_v_first=True,
    )
    return attn_out.squeeze(0), k_first(final_state)


def run_fla(inputs: Inputs, case: Case):
    """flash-linear-attention-npu's ``chunk_kda_fwd``.

    Trimmed to ``(attn_out, final_state)``: upstream also returns the
    ``gk``/``Aqk``/``Akk`` intermediates, which are mandatory outputs and are
    computed either way, so dropping the references to them here changes what
    is read, not what is paid for.

    This arm takes exactly what ``ascendc`` takes -- the caller's rank-4
    ``[1, T, H, D]`` tensors as ``BSND``, raw q/k, raw gate, raw beta logits --
    and fuses all of it: both L2 norms under ``use_qk_l2norm_in_kernel``, the
    beta sigmoid under ``use_beta_sigmoid_in_kernel``, and the bounded gate
    under ``use_gate_in_kernel``. So no host preparation at all, which is the
    comparison this benchmark is for.

    The file used to claim fla's in-kernel path "refuses every chunk size but
    64" as the reason for normalizing on the host. The constraint is real but
    it is the opposite of a blocker here: C64 is the only size this benchmark
    runs, and the only one ``ascendc`` supports. Measured against the
    host-prepared spelling the fused one agrees to 6.1e-5 and runs 2.12 ms
    against 2.77 ms at H8 T1024 -- so the host form was charging this arm ~24%
    for work the target fuses.

    ``state_v_first`` for the same reason as in ``run_cann``, and with the same
    measured result -- bit-identical to the K-first spelling, so the flag buys
    the shared layout and nothing else.
    """
    if ref_fla is None:
        raise NotImplementedError(REF_OPS_ERROR)
    out = ref_fla.chunk_kda(
        inputs.q, inputs.k, inputs.v, inputs.a, inputs.b,
        SCALE, case.chunk_size,
        layout="BSND", cu_seqlens=inputs.cu_seqlens, output_final_state=True,
        initial_state=native_live_state(inputs.state, inputs.indices),
        safe_gate=SAFE_GATE, lower_bound=LOWER_BOUND,
        use_gate_in_kernel=USE_GATE_IN_KERNEL,
        use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True,
        A_log=inputs.A_log, dt_bias=inputs.dt_bias,
        state_v_first=True,
    )
    return out[0].squeeze(0), k_first(out[1])


ARMS: dict[str, Callable[[Inputs, Case], tuple]] = {
    "ascendc": run_ascendc,
    "cann": run_cann,
    "fla": run_fla,
}

# What each arm does inside its own kernel, out of the work the deployed prefill
# has to do per call. Printed under the table, because it is the one thing that
# stops the latency column from being read as a like-for-like race.
#
# Every arm is handed the same raw rank-4 [1,T,H,D] tensors and the same raw
# gate, beta and state pool. Two of the three fuse everything from there.
# `cann`'s operator cannot: its seven attributes include the bounded gate but
# no L2 norm, no epsilon and no beta sigmoid, and `beta` is a plain coefficient
# on every kernel path -- so that arm alone pays `host_l2norm` and a host
# sigmoid, inside its timed region. There is no flag, no sibling op and no
# second entry point that closes the gap: `recurrent_kda` has both flags but is
# the decode kernel, and `kda_input_proj` sigmoids beta but is the projection
# stage and subsumes the GEMMs.
#
# So the `cann` column is its honest end-to-end cost on these inputs, not its
# kernel in isolation. The handicap was measured at ~0.1 ms of 0.87 ms at
# T1024, shrinking with T.
FUSION: dict[str, str] = {
    "ascendc": "L2 norms + beta sigmoid + bounded gate, all in kernel",
    "cann": "bounded gate in kernel; L2 norms + beta sigmoid on host (no flag exists)",
    "fla": "L2 norms + beta sigmoid + bounded gate, all in kernel",
}

# The arm speedups are quoted against. The reference arms are the bar, not the
# standard: correctness is decided by `run_torch` for all three alike.
TARGET = "ascendc"

# What to say when an arm cannot run, instead of a traceback.
UNAVAILABLE_MARKERS = (
    "not implemented",
    # `ref_ops`' own message when a vendor package was never built. Matched on
    # the module prefix rather than on the build instruction that follows it,
    # which has already changed once and took the whole run down with it when
    # it did -- a missing optional arm has to stay a skipped column.
    "ref_ops.cann:",
    "ref_ops.fla_npu:",
    "is not installed",
)


def unsupported_shape(name: str, case: Case) -> Optional[str]:
    """Why arm ``name`` does not cover ``case``, or None if it does.

    Separate from ``arm_status``, which asks whether an arm exists at all and
    answers once per run from one probe shape. Support can also depend on the
    shape, and an arm that is installed and working for most rows must not be
    dropped from all of them -- nor silently crash on the rows it does not
    cover, which is what a single-shape probe would allow.

    ``fla`` is the case in point: its L2-normalizing three-stage path is the
    only one that accepts these inputs and it refuses every chunk size but 64,
    with ``chunkSize only supports 64`` from the workspace call. So it covers
    the C64 rows and reports ``-`` on the C128 ones, rather than being skipped
    outright for a limit that applies to half the table. ``cann`` runs both.

    ``ascendc`` has the same restriction for a different reason: 64 is a
    compile-time constant of the kernel, not an argument -- the UB map, the
    workspace stride and the gate-rebase block count all derive from it, so the
    op takes no ``chunk_size`` at all. Reporting it as unsupported rather than
    silently measuring a C64 kernel in the C128 rows is the point; a row that
    compared C64 against the references' C128 would read as a speedup.
    """
    if name == "fla" and case.chunk_size != 64:
        return "fla's KDA chunk path supports chunk_size=64 only"
    if name == "ascendc" and case.chunk_size != 64:
        return "the Ascend C kernel's chunk size is fixed at 64"
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
    """One row: every live arm validated against the anchor, then all timed.

    Validation is done here rather than through the framework's
    target-vs-reference comparison, because correctness is not a question about
    the target: each arm is checked against ``run_torch`` independently, so a
    wrong *reference* arm is reported as wrong instead of silently becoming the
    standard the target is held to. That matters more with these two arms than
    it did with none -- they agree with each other to a bf16 ulp, which would
    make a shared misreading of the inputs look like consensus.

    That leaves the framework doing what it is good at -- warmup, repeats and
    paired statistics -- and keeps the measurement meaningful even when the
    target is the only live arm.

    The target can also be the arm that does not cover the row -- ``ascendc``
    is C64-only while the table has C128 rows. Such a row still measures every
    reference that covers it, with the first of them standing in as the
    denominator, so the C128 rows keep reporting the bar rather than going
    blank. The target's own cells read ``-`` there.
    """
    inputs = make_inputs(case)
    try:
        # `.cpu()` as well as `.float()`: the arms are compared on the host
        # below, and `assert_close` treats device as an attribute to match, so
        # an anchor left on the NPU fails every comparison on the device alone
        # -- including against a bit-identical tensor. The anchor is computed
        # on the NPU; only the comparison happens here.
        expected = tuple(t.float().cpu() for t in run_torch(inputs, case))
    except NotImplementedError:
        # No anchor yet: measure, and report validity as unknown rather than
        # as PASS. Timing a kernel before it can be checked is useful; calling
        # it correct because nothing contradicted it is not.
        expected = None

    # An arm that does not cover this shape is left out of both the validation
    # and the measurement, so its cells read `-` on this row and stay populated
    # on the rows it does cover.
    live = [
        name for name in reference_names if unsupported_shape(name, case) is None
    ]
    target_covers = unsupported_shape(target_name, case) is None

    validity: dict[str, Optional[bool]] = {}
    for name in ([target_name] if target_covers else []) + live:
        if expected is None:
            validity[name] = None
            continue
        got = tuple(t.float().cpu() for t in ARMS[name](inputs, case))
        try:
            comparator(got, expected)
            validity[name] = True
        except AssertionError:
            validity[name] = False

    if not target_covers:
        # Nothing to time the references against, so the first of them becomes
        # the denominator for this row only. Assembled directly rather than
        # through `benchmark`, which requires a target.
        if not live:
            empty = BenchmarkResult(
                target=ImplementationResult(name="target"), references=[],
            )
            return empty, validity
        stand_in, *rest = live
        result = benchmark(
            target=functools.partial(ARMS[stand_in], inputs, case),
            references=[
                Reference(name=name, fn=functools.partial(ARMS[name], inputs, case))
                for name in rest
            ],
            comparator=lambda actual, expected_: None,
            measurer=NpuEventMeasurer(),
            config=CONFIG,
        )
        # Re-label: the stand-in's numbers belong in its own column, and the
        # target column is empty on this row.
        result.references.insert(
            0,
            ReferenceResult(name=stand_in, measurement=result.target.measurement),
        )
        result.target = ImplementationResult(name="target")
        return result, validity

    result = benchmark(
        target=functools.partial(ARMS[target_name], inputs, case),
        references=[
            Reference(name=name, fn=functools.partial(ARMS[name], inputs, case))
            for name in live
        ],
        # Validation already happened above, against the anchor. Passing a
        # comparator that accepts anything keeps the framework from re-deciding
        # correctness by comparing the arms to each other -- which here would
        # compare two kernels that are not the anchor and pass.
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

    def ok(validity: dict, name: str) -> str:
        # `None` is "no anchor to check against", which is neither PASS nor
        # FAIL and must not read as either.
        verdict = validity.get(name)
        return "-" if verdict is None else ("PASS" if verdict else "FAIL")

    table = []
    for label, (result, validity) in rows:
        target = result.target.measurement
        row = [
            label,
            f"{target.median_ms:.4f}" if target else "-",
            f"{target.mad_ms:.4f}" if target else "-",
            ok(validity, target_name),
        ]
        by_name = {ref.name: ref for ref in result.references}
        # The speedup denominator. Normally the target; on a row the target
        # does not cover, `bench_one` promoted the first reference instead, so
        # the column reads against that one and its own `vs` cell is 1.00x.
        denominator = target
        if denominator is None:
            for name in reference_names:
                ref = by_name.get(name)
                if ref is not None and ref.measurement is not None:
                    denominator = ref.measurement
                    break
        for name in reference_names:
            ref = by_name.get(name)
            measurement = ref.measurement if ref else None
            if measurement is None:
                row += ["-", "-", "-"]
                continue
            row.append(f"{measurement.median_ms:.4f}")
            row.append(
                f"{measurement.median_ms / denominator.median_ms:.2f}x"
                if denominator and denominator.median_ms > 0 else "-"
            )
            row.append(ok(validity, name))
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
        ("<arm> ok", "agrees with `run_torch` at the tolerance above;\n"
                     "'-' means there is no anchor to check against yet"),
        ("vs <arm>", "median(arm) / median(target), >1 means target is faster;\n"
                     "on a row the target does not cover, quoted against the\n"
                     "first reference that does, whose own cell reads 1.00x"),
    ]
    arms = [
        ("ascendc", "the sglang GLM-5.3 plugin kernel, written here;\n"
                    "chunk size fixed at 64, so '-' on the C128 rows"),
        ("cann", "CANN 9.2.0 ChunkKdaFwd, via ref_ops.cann; BSND"),
        ("fla", "flash-linear-attention-npu, via ref_ops.fla_npu; BSND;\n"
                "chunk_size=64 only, so '-' on the C128 rows"),
    ]
    # Describe only the arms this table has columns for.
    arms = [
        (name, text) for name, text in arms
        if name == target_name or name in reference_names
    ]
    pad = max(len(name) for name, _ in [*legend, *arms])

    print()
    for name, text in legend:
        head, *rest = text.split("\n")
        print(f"{name.ljust(pad)} : {head}")
        for line in rest:
            print(f"{' ' * pad}   {line}")
    print()
    for name, text in arms:
        head, *rest = text.split("\n")
        print(f"{name.ljust(pad)} : {head}")
        for line in rest:
            print(f"{' ' * pad}   {line}")

    # Every arm is handed the same raw inputs; they differ in how much of the
    # preparation each can fuse. `cann` cannot fuse the L2 norms or the beta
    # sigmoid -- no attribute exists -- so its column includes host work the
    # other two do on-chip. Printed with the table so the latency column is not
    # read as like-for-like.
    print()
    print("what each arm fuses, from identical raw inputs:")
    for name, text in arms:
        print(f"{name.ljust(pad)} : {FUSION[name]}")


def main() -> None:
    print(f"tolerance: rtol={RTOL} atol={ATOL}  (each arm vs run_torch)")

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
        probe = probe_case(name)
        if probe is None:
            status[name] = "covers none of the configured chunk sizes"
            continue
        status[name] = arm_status(name, probe)

    live = [name for name, reason in status.items() if reason is None]
    skipped = {name: reason for name, reason in status.items() if reason is not None}

    if TARGET in live:
        target_name = TARGET
    elif live:
        # The target is what this benchmark is for, but with reference arms
        # present its absence no longer leaves nothing to measure: the two
        # outside kernels on these shapes are the bar the target is aiming at,
        # and that number is worth having before the target exists. The first
        # live arm stands in so the speedup column has a denominator; it is
        # labelled as the target in the table, so what is being quoted against
        # what stays readable.
        target_name = live[0]
        print()
        print(f"target arm '{TARGET}' is unavailable: {skipped[TARGET]}")
        print(f"measuring the reference arms instead, against '{target_name}'.")
    else:
        print()
        for name, reason in skipped.items():
            print(f"{name} is unavailable: {reason}")
        print()
        print("no arm can run; nothing to measure.")
        raise SystemExit(1)

    reference_names = [name for name in ARMS if name in live and name != target_name]

    print(f"target: {target_name}")
    print(f"references: {', '.join(reference_names) or 'none'}")
    if skipped:
        print()
        for name, reason in skipped.items():
            print(f"skipped {name}: {reason}")
    print()

    if not reference_names:
        # Still worth running: the target's own correctness and latency are
        # measured against the anchor, which is the part that does not depend
        # on another arm existing.
        print("no other arm is available; measuring the target on its own.")
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
    torch.npu.set_device(DEVICE)
    main()
