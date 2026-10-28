"""Correctness and speedup for the GLM-5.3 KDA varlen recurrence op.

``glm_kda_varlen_recurrent_ascendc`` below is a copy of sglang's
``glm_kda_varlen_recurrent_npu`` with only the compute dispatch replaced by
``torch.ops.sgl_glm53_ascend.kda_varlen_recurrent``. Everything above that
point is kept verbatim so the two arms differ in the kernel and nothing else.

Two references, because the caller has two prefill paths and the slower one is
not what it runs:

``triton``    the generic Triton recurrence — one program per value tile per
              head, BV=64. The caller's fallback, and all it has outside
              prefill.
``prepared``  ``SGLANG_GLM53_KDA_PREFILL_PREPARE=1`` plus
              ``SGLANG_GLM53_KDA_PREPARED_ASCENDC=1``: the token-parallel FP32
              input preparation followed by the Ascend C ``prepared_recurrent``
              kernel. This is the deployed prefill path, roughly 3x the generic
              one, so it is the comparison that decides whether the new kernel
              is worth shipping. It covers prefill H4/H8 with one packed
              sequence and no checkpoints; every other row reports ``-``.
``fla``       ``fla_npu.ops.ascendc.recurrent_kda``, Tianjin University's fused
              KDA recurrence from flash-linear-attention-npu. An outside
              implementation of the same function, so it answers a question the
              other two cannot: whether a gap is this op or this hardware. It
              computes the gate, the L2 norm and the beta sigmoid in-kernel from
              the model's own tensors, like ours and unlike ``prepared``. Well
              behind on prefill and at parity on decode, which localizes its
              deficit to block mapping rather than to the recurrence -- see
              ``README.md``. Checkpoints are out of scope, so ``verify`` rows
              report ``-``.

Regimes are the deployed parallel strategies, ``H = 64 / attn_tp_size``:
prefill at H=4/8 (TP16/TP8), decode at H=16/32/64 (TP4/TP2/TP1). A regime whose
kernel is not written yet is skipped, so implementing one width at a time shows
up as that row starting to report.

Decode covers both of the caller's regimes, because they exercise different
code and only the second one is hot:

``decode``  one token per request, no checkpoints — ``forward_decode``.
``verify``  ``draft_token_num`` tokens per request with the state checkpointed
            after every step — EAGLE target verification, and the shape the
            profile is dominated by. Its rows are the profiled deployment
            shapes rather than a sweep, at one DP replica's batch; see
            ``VERIFY_SHAPES``.

Both validate the state pool alongside the output, and ``verify`` validates the
checkpoint buffer too: it is the only output of that path the caller actually
reads, since a checkpointing launch deliberately leaves the state slot alone.
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
from sglang.srt.hardware_backend.npu.attention.glm53.kda_prefill_npu import (
    _bv16_b1_layout_ok,
    _prepare_inputs,
)
from sglang.srt.hardware_backend.npu.attention.glm53.kda_recurrent_npu import (
    glm_kda_varlen_recurrent_npu,
)

from fla_npu.ops.ascendc import recurrent_kda as fla_recurrent_kda

from benchmark import (
    BenchmarkConfig,
    NpuEventMeasurer,
    Reference,
    benchmark,
    make_comparator,
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

# The `fla` arm is the one comparison that is not our own code on both sides,
# so it is the one that cannot be held to the tolerance above. Measured, it
# lands 1-2 bf16 ulps out on ~20 of 2M output elements at ~1e-4 magnitude, with
# the state pool matching inside `RTOL` everywhere — a different summation
# order in a recurrence whose output is small, not a different recurrence.
# bf16's own `assert_close` default is the honest bar for that, and keeping it
# on this arm alone is what preserves the tight one for `triton`/`prepared`,
# where bit-exactness is a real property worth failing on.
FLA_RTOL = 1.6e-2
FLA_ATOL = 1e-5
fla_comparator = make_comparator(rtol=FLA_RTOL, atol=FLA_ATOL)

PREFILL_HEADS = (4, 8)  # TP16 / TP8
PREFILL_TOKENS = (1024, 2048, 4096, 8192, 16384)
DECODE_HEADS = (16, 32, 64)  # TP4 / TP2 / TP1
DECODE_BATCHES = (1, 32, 128)

# `draft_token_num` in the deployed EAGLE config: the recurrence advances four
# steps per request per verify call and checkpoints after each.
DRAFT_TOKENS = 4

# The verify shapes actually measured on device, as (label, heads, requests).
# These are the profiled deployment maxima, not a sweep: what varies across
# rows is the block count the kernel gets to work with (requests * heads),
# against 40 vector cores.
#
# `req` is per DP replica, which is what one launch sees -- not the
# deployment-wide concurrency `C` the serving configs are quoted in. The two
# are easy to mistake for a disagreement, so both are recorded here:
# `req = C / DP`, and `H = 64 / TP` since GLM-5.3-Flash has 64 heads. Every
# config below is 8 cards (DP * TP).
#
#   label             C   DP  TP  heads  req   blocks   waves
#   256K DP4TP2        8   4   2     32     2      64     1.6   <- priority shape
#   256K DP2TP4        6   2   4     16     3      48     1.2
#   256K DP8TP1        8   8   1     64     1      64     1.6
#   512K DP4TP2        4   4   2     32     1      32     0.8
#   512K DP2TP4        2   2   4     16     1      16     0.4
#
# 512K DP8TP1 has no row because that configuration is unavailable in
# deployment, not because it was left unmeasured.
VERIFY_SHAPES = (
    ("256K DP4TP2", 32, 2),
    ("256K DP2TP4", 16, 3),
    ("256K DP8TP1", 64, 1),
    ("512K DP4TP2", 32, 1),
    ("512K DP2TP4", 16, 1),
)

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


def glm_kda_varlen_recurrent_prepared(
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
) -> torch.Tensor:
    """The deployed prefill path: FP32 preparation, then the Ascend C kernel.

    This is ``run_prepared_prefill`` with both environment switches on, inlined
    rather than called, for one reason: upstream reaches the Ascend C kernel
    through ``prepared_recurrent_v2``, a sidecar op built by sglang's
    ``ascendc_prepared/build_compatible.py`` from this repo's
    ``prepared_recurrent.cpp``. That op is not in this build. What the sidecar
    adds over ``prepared_recurrent`` is a native transposed-state fast path and
    int64 metadata, neither of which changes the recurrence, plus the H8
    two-launch host loop — which is exactly the Python loop below. So calling
    v1 twice per H8 measures the same kernels in the same order, and the
    state-layout difference is absorbed by handing it the contiguous ``[K,V]``
    orientation v1 wants.

    Preparation is counted in, because the caller pays for it: it exists only to
    feed this kernel, whereas the fused op reads the model's own tensors.
    """
    heads = q.shape[2]
    q, k, v, a, b = (t.contiguous() for t in (q, k, v, a, b))
    normalized_q = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    normalized_k = torch.empty_like(normalized_q)
    gate_decay = torch.empty_like(normalized_q)
    beta = torch.empty(b.shape, dtype=torch.float32, device=b.device)
    output = torch.zeros_like(v)
    rows = q.shape[1] * heads
    _prepare_inputs[(min(rows, 32768),)](
        q,
        k,
        a,
        b,
        A_log,
        dt_bias,
        normalized_q,
        normalized_k,
        gate_decay,
        beta,
        scale,
        lower_bound,
        heads,
        D,
        rows,
        num_warps=1,
        num_stages=3,
        multibuffer=False,
    )
    # One 32-block launch per 4-head group, the grouping the sidecar's host
    # function performs internally for H8.
    for head_offset in range(0, heads, 4):
        torch.ops.sgl_glm53_ascend.prepared_recurrent(
            normalized_q,
            normalized_k,
            v,
            gate_decay,
            beta,
            output,
            initial_state_source,
            initial_state_indices,
            cu_seqlens,
            None,
            None,
            False,
            head_offset,
        )
    return output


def glm_kda_varlen_recurrent_fla(
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
    one_token_per_seq: bool,
) -> torch.Tensor:
    """flash-linear-attention-npu's fused KDA recurrence on the caller's tensors.

    The three gate switches below are what make this the same function as the
    other arms rather than a near miss: ``safe_gate`` selects
    ``lower_bound * sigmoid(exp(A_log) * (a + dt_bias))``, which is the gate
    sglang's Triton kernel computes inline, and the L2 norm and beta sigmoid are
    likewise the ones that kernel applies per token. So this arm reads the
    model's own tensors, like ours, and unlike ``prepared``'s FP32 preparation.

    Each regime is handed to the op in the spelling it accepts natively, which
    is what keeps this a kernel measurement rather than a measurement of
    conversions the caller would never perform:

    ``one_token_per_seq``  decode and verify. The op's BSND is ``[B,T,H,D]``
        and splits by batch row when given no ``cu_seqlens``, so one token per
        request is ``[B,1,H,D]`` — the same bytes as our ``[1,B,H,D]``, reached
        by ``view`` with no copy. Slots take the documented speculative
        ``[seq_num,max_step]`` form, one row per request and one step.
    otherwise  packed prefill, where the tokens of a sequence are contiguous
        and there is no batch axis to carry them. That is TND, and there slots
        are genuinely per token: the 1D form is read at the packed token
        offset, so the per-request indices have to be expanded across each
        request's length.

    ``initial_state`` is the pool's V-first base: ``native_state`` allocates
    ``[slots,H,V,K]`` contiguous and hands out the ``[slots,H,K,V]`` transpose,
    so transposing back recovers the contiguous V-first orientation this op
    wants while still aliasing the caller's pool, keeping the in-place final
    state landing where the other arms write it.
    """
    state = initial_state_source.transpose(-1, -2)
    if not state.is_contiguous():
        raise ValueError("fla arm expects the V-first state pool base")

    if one_token_per_seq:
        requests = q.shape[1]
        q, k, v, a = (t.view(requests, 1, *t.shape[2:]) for t in (q, k, v, a))
        b = b.view(requests, 1, b.shape[2])
        layout = "BSND"
        # [seq_num, max_step] with max_step == 1.
        slots = initial_state_indices.reshape(-1, 1).to(torch.int32)
        offsets = None
    else:
        q, k, v, a, b = (t.squeeze(0).contiguous() for t in (q, k, v, a, b))
        layout = "TND"
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
        slots = initial_state_indices.repeat_interleave(lengths).to(torch.int32)
        offsets = cu_seqlens

    out, _ = fla_recurrent_kda(
        q,
        k,
        v,
        a,
        b,
        state,
        cu_seqlens=offsets,
        ssm_state_indices=slots,
        A_log=A_log,
        dt_bias=dt_bias,
        layout=layout,
        scale=scale,
        output_final_state=False,
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
        state_v_first=True,
    )
    if one_token_per_seq:
        # Back to the [1,T,H,V] the other arms return, again without a copy:
        # [B,1,H,V] -> [1,B,H,V], where B is the token count.
        return out.view(1, out.shape[0], *out.shape[2:])
    return out.unsqueeze(0)


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
    # Per-step checkpoints, or None outside the verify regime.
    intermediate: Optional[torch.Tensor] = None
    # Whether every request carries exactly one token, which is what lets the
    # `fla` arm use the op's native BSND spelling. Recorded at construction
    # from the host-side lengths: deriving it from `starts` inside the launch
    # would put a device-to-host sync in the timed region, which on a ~0.12ms
    # decode call costs more than the kernel.
    one_token_per_seq: bool = False


def native_state(slots: int, heads: int) -> torch.Tensor:
    """The only state layout that reaches the kernel in production.

    Allocated contiguous as [slots,H,V,K] and viewed as [slots,H,K,V], which
    gives stride(2)=1, stride(3)=128. The backend normalizes both pools to it.
    """
    pool = torch.randn(slots, heads, D, D, device=DEVICE) * 0.1
    return pool.transpose(-1, -2)


def native_intermediate(slots: int, steps: int, heads: int) -> torch.Tensor:
    """The checkpoint buffer as the backend hands it over.

    Upstream this is a slice of ``intermediate_ssm``, allocated
    [layers, slots, steps, HV, V, K] and passed ``.transpose(-1, -2)``, so the
    kernel sees [slots, steps, HV, K, V] with stride(3)=1 — the same
    K-contiguous orientation as the state pool.

    Filled with noise rather than zeros so a step the kernel fails to write
    shows up as a mismatch instead of matching an untouched zero.
    """
    pool = torch.randn(slots, steps, heads, D, D, device=DEVICE) * 0.1
    return pool.transpose(-1, -2)


def clone_state(state: torch.Tensor) -> torch.Tensor:
    """Copy a state pool preserving its [V,K] backing.

    A plain ``.clone()`` makes the view contiguous and silently selects a
    different Triton specialization. Also used for the checkpoint buffer, which
    is the same trailing layout one axis deeper.
    """
    return state.transpose(-1, -2).clone().transpose(-1, -2)


def prepared_state(state: torch.Tensor) -> torch.Tensor:
    """The same values in the contiguous [K,V] layout the v1 op requires.

    v1 stages the state tile through a transpose that assumes stride(2)==128;
    the deployed sidecar adds a branch for the native stride(2)==1 pool but
    computes the identical recurrence either way. Converting here keeps that
    difference out of the measurement — the copy is outside the timed call, and
    the comparison reads the result back through the same conversion.
    """
    return state.contiguous()


def make_case(
    heads: int,
    lengths: tuple[int, ...],
    prefill: bool,
    checkpoints: bool = False,
) -> Case:
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
        # One row per request, indexed by batch position rather than by state
        # slot — `build_verify_intermediate_state_indices` is a plain arange,
        # so request i owns row i. Steps are the longest sequence here, which
        # under verify is every sequence.
        intermediate=(
            native_intermediate(batch, max(lengths), heads) if checkpoints else None
        ),
        one_token_per_seq=all(length == 1 for length in lengths),
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
        intermediate_state=case.intermediate,
        prefill=case.prefill,
    )


def triton_launch(case: Case) -> torch.Tensor:
    return glm_kda_varlen_recurrent_npu(**call_kwargs(case))


def ascendc_launch(case: Case) -> torch.Tensor:
    return glm_kda_varlen_recurrent_ascendc(**call_kwargs(case))


def prepared_launch(case: Case) -> torch.Tensor:
    kwargs = call_kwargs(case)
    del kwargs["intermediate_state"], kwargs["prefill"]
    return glm_kda_varlen_recurrent_prepared(**kwargs)


def fla_launch(case: Case) -> torch.Tensor:
    kwargs = call_kwargs(case)
    del kwargs["intermediate_state"], kwargs["prefill"]
    return glm_kda_varlen_recurrent_fla(
        **kwargs,
        one_token_per_seq=case.one_token_per_seq,
    )


def fla_supported(case: Case) -> bool:
    """Whether the op covers this regime.

    Only checkpoints are out of scope: the op writes one state per token into
    the pool and has no per-step checkpoint tensor, so there is nothing for a
    ``verify`` row to validate against. Prefill and decode are both in scope —
    unlike CANN's ``RecurrentKda``, which capped a sequence at 8 tokens (see
    DISCARDED-cann-recurrent-kda.md).
    """
    return case.intermediate is None


def prepared_supported(case: Case) -> bool:
    """Whether the caller would take the prepared path for this regime.

    The dispatch conditions are the ones in ``glm_kda_varlen_recurrent_npu``
    around its ``run_prepared_prefill`` call, plus upstream's own layout guard
    on the tensors the kernel indexes. Decode, verify and packed prefill all
    fail here, which is correct: outside prefill the caller has no second path.
    """
    if not case.prefill or case.intermediate is not None:
        return False
    if case.q.shape[2] not in (4, 8) or case.indices.numel() != 1:
        return False
    if not 128 <= case.q.shape[1] <= 16384:
        return False
    return _bv16_b1_layout_ok(
        case.q,
        case.k,
        case.v,
        case.a,
        case.b,
        torch.zeros_like(case.v),
        prepared_state(case.state),
        case.indices,
        case.starts,
        None,
        None,
    )


def arm(launch, case: Case):
    """Bind a launch to its own case, capturing the output for validation.

    Everything the call is allowed to write is returned, not just the output
    tensor: the state pool, and under verify the checkpoint buffer. Those are
    in-place arguments, so a kernel that computes the right output while
    corrupting a neighbouring slot has to fail here.
    """
    box = {}

    def run():
        box["out"] = launch(case)

    def output():
        if case.intermediate is None:
            return box["out"], case.state
        return box["out"], case.state, case.intermediate

    return run, output


def fresh(case: Case, contiguous_state: bool = False) -> Case:
    """An arm-private copy, so every arm starts from identical values."""
    state = clone_state(case.state)
    return replace(
        case,
        state=prepared_state(state) if contiguous_state else state,
        intermediate=(
            None if case.intermediate is None else clone_state(case.intermediate)
        ),
    )


def bench_one(case: Case):
    """One row: the op against the generic recurrence, the deployed path and fla.

    An arm is dropped on regimes it does not cover, so its columns read ``-``
    there rather than comparing against something that is not a candidate for
    that shape: ``prepared`` where the caller would not route to it, ``fla``
    where checkpoints are required.
    """
    references = []

    triton_run, triton_output = arm(triton_launch, fresh(case))
    references.append(Reference("triton", triton_run, triton_output))

    if prepared_supported(case):
        # Its state pool carries the same values in the [K,V]-contiguous
        # layout, which compares fine: `assert_close` defaults to
        # `check_stride=False`.
        prepared_run, prepared_output = arm(
            prepared_launch, fresh(case, contiguous_state=True)
        )
        references.append(Reference("prepared", prepared_run, prepared_output))

    if fla_supported(case):
        # Carries its own tolerance: see `fla_comparator`. It is the only arm
        # here that is not our own code, so holding it to the bit-exact bar the
        # others meet would fail the row over bf16 rounding and take the whole
        # row's timings down with it.
        fla_run, fla_output = arm(fla_launch, fresh(case))
        references.append(
            Reference("fla", fla_run, fla_output, comparator=fla_comparator)
        )

    target_run, target_output = arm(ascendc_launch, fresh(case))
    return benchmark(
        target=target_run,
        references=references,
        comparator=comparator,
        measurer=NpuEventMeasurer(),
        config=CONFIG,
        output_fn=target_output,
    )


def regimes():
    """(label, heads, per-request lengths, prefill, checkpoints) per row."""
    for heads in PREFILL_HEADS:
        for tokens in PREFILL_TOKENS:
            yield f"prefill H{heads} T{tokens // 1024}k", heads, (tokens,), True, False
    for heads in DECODE_HEADS:
        for batch in DECODE_BATCHES:
            # Plain decode is one token per sequence, so T == batch.
            yield f"decode  H{heads} B{batch}", heads, (1,) * batch, False, False
    for label, heads, requests in VERIFY_SHAPES:
        # Target verify: `draft_token_num` tokens per request, checkpointed.
        yield (
            f"verify  {label} R{requests}",
            heads,
            (DRAFT_TOKENS,) * requests,
            False,
            True,
        )


def unsupported(case: Case) -> str | None:
    """The op's own rejection message for this regime, or None if it runs.

    A regime the launcher set does not cover is a skipped row, not a failure:
    the caller keeps its Triton fallback for exactly those. Only the op's
    deliberate `TORCH_CHECK`s are treated this way — anything else propagates.
    """
    try:
        ascendc_launch(fresh(case))
        torch.npu.synchronize()
        return None
    except RuntimeError as exc:
        message = str(exc)
        for marker in ("not implemented yet", "does not support", "supports H="):
            if marker in message:
                return message.splitlines()[0]
        raise


def print_table(rows):
    """One row per regime, one latency/speedup column pair per reference.

    The framework's own ``print_results_table`` renders ``references[0]`` only,
    so a second reference would be measured and validated but never shown.
    """
    names = []
    for _, result in rows:
        for ref in result.references:
            if ref.name not in names:
                names.append(ref.name)

    headers = ["Benchmark", "Valid", "Ours (ms)", "MAD"]
    for name in names:
        headers += [f"{name} (ms)", f"vs {name}"]

    table = []
    for label, result in rows:
        valid = all(
            ref.validation.passed
            for ref in result.references
            if ref.validation is not None
        )
        target = result.target.measurement
        row = [
            label,
            "PASS" if valid else "FAIL",
            f"{target.median_ms:.4f}" if target else "-",
            f"{target.mad_ms:.4f}" if target else "-",
        ]
        by_name = {ref.name: ref for ref in result.references}
        for name in names:
            ref = by_name.get(name)
            measurement = ref.measurement if ref else None
            if measurement is None:
                # Absent on a regime the reference does not cover, and on every
                # regime once validation has failed.
                row += ["-", "-"]
                continue
            row.append(f"{measurement.median_ms:.4f}")
            row.append(
                f"{measurement.median_ms / target.median_ms:.2f}x"
                if target and target.median_ms > 0
                else "-"
            )
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

    failures = [
        (label, ref)
        for label, result in rows
        for ref in result.references
        if ref.validation is not None and not ref.validation.passed
    ]
    if failures:
        print()
        print("validation errors:")
        for label, ref in failures:
            first = (
                ref.validation.error.splitlines()[0]
                if ref.validation.error
                else "unknown"
            )
            print(f"  {label} vs {ref.name}: {first}")
            if ref.validation.diff_summary:
                print(ref.validation.diff_summary)

    print()
    print("Ours (ms)       : median latency of kda_varlen_recurrent")
    print("MAD             : median absolute deviation of that latency")
    print("triton          : the generic Triton recurrence, the caller's fallback")
    if "prepared" in names:
        print(
            "prepared        : FP32 input preparation + the Ascend C "
            "prepared_recurrent\n"
            "                  kernel, the deployed prefill path; '-' where the\n"
            "                  caller would not route to it"
        )
    if "fla" in names:
        print(
            "fla             : flash-linear-attention-npu's fused recurrent_kda;\n"
            "                  '-' on verify, which needs per-step checkpoints.\n"
            f"                  Validated at rtol={FLA_RTOL:g} atol={FLA_ATOL:g}, "
            "bf16's own default,\n"
            "                  because it is the one arm that is not our code"
        )
    print("vs <ref>        : median(ref) / median(ours), >1 means ours is faster")


def main():
    print(f"tolerance: rtol={RTOL} atol={ATOL}")
    print(f"           fla arm: rtol={FLA_RTOL:g} atol={FLA_ATOL:g}")
    print()

    rows = []
    skipped = []
    for label, heads, lengths, prefill, checkpoints in regimes():
        torch.manual_seed(1979 + heads + sum(lengths) + len(lengths))
        case = make_case(heads, lengths, prefill, checkpoints)
        reason = unsupported(case)
        if reason is not None:
            skipped.append((label, reason))
            continue
        print(f">>> {label} ...")
        rows.append((label, bench_one(case)))
    print()

    if rows:
        print_table(rows)
    else:
        print("no kernels implemented yet")

    if skipped:
        print()
        print("skipped, unsupported by the op:")
        for label, reason in skipped:
            print(f"  {label}: {reason}")


if __name__ == "__main__":
    torch.npu.set_device(DEVICE)
    main()
