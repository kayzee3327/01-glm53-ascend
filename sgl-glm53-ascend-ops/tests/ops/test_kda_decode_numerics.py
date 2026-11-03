"""Numerical correctness of the decode kernels against the Triton baseline.

The decode side holds the same equality bar as the prefill side, and for the
same reason: see `tests/ops/test_kda_prefill_numerics.py` for the two last-bit
hazards the recurrence compounds along a sequence.

What this file adds is sequence lengths past one step group. The token loop is
strip-mined by `kStepGroup` (4), so a sequence of five or more tokens refills
the staged bf16 q/k/a/v buffers while the previous group's recurrence may still
be reading them -- `v` is read last, inside the step. A missing V_MTE2
handshake there silently shifts v rows by four tokens, which equality on the
output catches and a 2e-3 tolerance does not.

The deployed shapes are one group or less (verify is `draft_token_num`=4 tokens,
decode is 1), so the multi-group cases here are in contract but not on the
serving path.
"""

import pytest
import torch
import torch_npu  # noqa: F401

import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.attention.glm53.kda_recurrent_npu import (
    glm_kda_varlen_recurrent_npu,
)

from sgl_glm53_ascend_ops.ops import kda_varlen_recurrent

# TP4 / TP2 / TP1 on the 64-head model. H=32 has its own kernel file; H=16 and
# H=64 share a template, so both paths need covering.
HEADS = (16, 32, 64)
D = 128
SLOTS = 16
SCALE = D**-0.5
LOWER_BOUND = -5.0
DEVICE = "npu"

# Per-request token counts. kStepGroup is 4, so these cover a single group, an
# exact group boundary, a one-token tail group, several whole groups, and a
# ragged batch where requests land on different group counts.
CASES = [
    (1,),
    (4,),
    (5,),
    (8,),
    (9,),
    (16,),
    (1,) * 8,
    (4,) * 4,
    (1, 2, 3, 4, 5, 6, 7),
    (7, 1, 4, 9),
]


def _case(heads, lengths):
    tokens = sum(lengths)
    batch = len(lengths)
    torch.manual_seed(31337 + tokens * 97 + batch)
    tensors = {
        name: (torch.randn(1, tokens, heads, D, device=DEVICE) * 0.2).bfloat16()
        for name in ("q", "k", "v", "a")
    }
    # Allocated [slots,H,V,K] and viewed [slots,H,K,V] so stride(2)==1: the one
    # state layout that reaches the kernel in production. A nonzero pool is the
    # point -- a zero slot is its own transpose and would hide a layout bug.
    state = (torch.randn(SLOTS, heads, D, D, device=DEVICE) * 0.1).transpose(-1, -2)
    return dict(
        **tensors,
        b=torch.randn(1, tokens, heads, device=DEVICE).bfloat16(),
        A_log=torch.randn(heads, device=DEVICE) * 0.1,
        dt_bias=torch.randn(heads * D, device=DEVICE) * 0.1,
        state=state,
        # Slot 0 is the padding row and is never written, so slots start at 1.
        indices=torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE),
        cu_seqlens=torch.tensor(
            [sum(lengths[:i]) for i in range(batch + 1)],
            dtype=torch.int32,
            device=DEVICE,
        ),
        lengths=lengths,
    )


def _clone_state(state):
    """Copy a state pool preserving its [V,K] backing.

    A plain ``.clone()`` makes the view contiguous, which both changes what the
    kernel reads and silently selects a different Triton specialization.
    """
    return state.transpose(-1, -2).clone().transpose(-1, -2)


def _run(launch, case, checkpoints=False):
    """Run one arm on private copies, returning output, state and checkpoints."""
    state = _clone_state(case["state"])
    mid = None
    if checkpoints:
        batch = len(case["lengths"])
        steps = max(case["lengths"])
        heads = case["q"].shape[2]
        # Noise rather than zeros so a step the kernel fails to write shows up
        # as a mismatch instead of matching an untouched zero. Same
        # K-contiguous orientation as the state pool, one axis deeper.
        #
        # The seed is pinned here, not in `_case`, so both arms start from the
        # same noise: a ragged batch leaves the steps past a short request
        # unwritten, and comparing those would otherwise diff two draws.
        torch.manual_seed(5150)
        mid = (
            torch.randn(batch, steps, heads, D, D, device=DEVICE) * 0.1
        ).transpose(-1, -2)
    out = launch(
        q=case["q"],
        k=case["k"],
        v=case["v"],
        a=case["a"],
        b=case["b"],
        A_log=case["A_log"],
        dt_bias=case["dt_bias"],
        initial_state_source=state,
        initial_state_indices=case["indices"],
        cu_seqlens=case["cu_seqlens"],
        lower_bound=LOWER_BOUND,
        scale=SCALE,
        intermediate_state=mid,
        prefill=False,
    )
    torch.npu.synchronize()
    return out.float(), state.float(), None if mid is None else mid.float()


def _assert_equal(label, got, ref, what):
    assert torch.equal(got, ref), (
        f"{label}: {int((got != ref).sum())}/{got.numel()} {what} differ, "
        f"max {(got - ref).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("lengths", CASES)
def test_decode_matches_triton_bit_exactly(heads, lengths):
    case = _case(heads, lengths)
    out, state, _ = _run(kda_varlen_recurrent, case)
    ref_out, ref_state, _ = _run(glm_kda_varlen_recurrent_npu, case)

    label = f"H={heads} lengths={lengths}"
    _assert_equal(label, out, ref_out, "outputs")
    _assert_equal(label, state, ref_state, "state elements")


@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("lengths", [(4,), (4,) * 4, (5,), (1, 2, 3, 4, 5, 6, 7)])
def test_decode_checkpoints_match_triton_bit_exactly(heads, lengths):
    """The verify regime: every step checkpointed, source slot left alone."""
    case = _case(heads, lengths)
    before = _clone_state(case["state"]).float()
    out, state, mid = _run(kda_varlen_recurrent, case, checkpoints=True)
    ref_out, _, ref_mid = _run(
        glm_kda_varlen_recurrent_npu, case, checkpoints=True
    )

    label = f"H={heads} lengths={lengths}"
    _assert_equal(label, out, ref_out, "outputs")
    _assert_equal(label, mid, ref_mid, "checkpoint elements")
    # Checkpointing replaces the final state write, so the pool is untouched.
    _assert_equal(label, state, before, "state elements")


@pytest.mark.parametrize("heads", HEADS)
def test_decode_leaves_other_slots_untouched(heads):
    """Only the indexed slots are written; the rest of the pool is a live cache."""
    case = _case(heads, (5, 5))
    before = _clone_state(case["state"]).float()
    _, after, _ = _run(kda_varlen_recurrent, case)

    other = [s for s in range(SLOTS) if s not in (1, 2)]
    assert torch.equal(after[other], before[other])
    assert not torch.equal(after[1], before[1])
    assert not torch.equal(after[2], before[2])
