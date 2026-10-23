"""Numerical correctness of the prefill kernels against the Triton baseline.

The kernel is bit-exact against upstream, not merely within tolerance, so these
compare for equality. That is a deliberate bar rather than a lucky one: the
recurrence runs thousands of steps, and the two places a last-bit difference
appeared both compounded along the sequence until they showed up in the output.
Holding equality is what catches a regression in either of them.

  * `tl.sum` over 128 fp32 lanes folds the two 64-lane halves elementwise and
    only then reduces, which a plain 2x64 `WholeReduceSum` pair does not match
    on ~28% of rows. The norm divides q and k on every step.
  * Vector `Div` is 1 ulp on this hardware (its 0-ulp algorithms are 950-series
    only), while upstream's division is IEEE-exact. Beta's reciprocal runs on
    the Scalar unit for that reason.

`tests/ops/test_kda.py` covers dispatch and rejection; this covers values.
Multi-sequence prefill is out of scope by contract — the op rejects num_req > 1
and the caller falls back — so every case here is one packed sequence.

Both head counts run the same cases. H=8 is H=4's token loop run twice per
consumer, so it shares every numerical hazard above and adds one of its own:
round 0's state tile has to reach GM before round 1 loads over the same UB
buffer, and the next round's first token is prefetched across that boundary.
A missed barrier there would show up as a wrong state or a wrong first token
of heads 4..7, which equality on both tensors catches.
"""

import pytest
import torch
import torch_npu  # noqa: F401

import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.attention.glm53.kda_recurrent_npu import (
    glm_kda_varlen_recurrent_npu,
)

from sgl_glm53_ascend_ops.ops import kda_varlen_recurrent

HEADS = (4, 8)
D = 128
SLOTS = 8
SCALE = D**-0.5
LOWER_BOUND = -5.0
DEVICE = "npu"

# (tokens, slot). CHUNK is 16, so these cover a short final chunk, a sequence
# below one chunk, exact multiples, and lengths past the per-core interleave.
# Slots past the first exercise the state-index indirection; slot 0 is the
# padding row and is never written.
CASES = [
    (1, 1),
    (15, 2),
    (16, 1),
    (17, 3),
    (31, 5),
    (127, 2),
    (129, 7),
    (255, 1),
    (513, 4),
    (1000, 2),
    (1023, 6),
    (4096, 3),
]


def _case(heads, tokens, slot):
    torch.manual_seed(31337 + tokens)
    tensors = {
        name: (torch.randn(1, tokens, heads, D, device=DEVICE) * 0.2).bfloat16()
        for name in ("q", "k", "v", "a")
    }
    # Allocated [slots,H,V,K] and viewed [slots,H,K,V] so stride(2)==1: the one
    # state layout that reaches the kernel in production. A nonzero pool is the
    # point — a zero slot is its own transpose and would hide a layout bug.
    state = (torch.randn(SLOTS, heads, D, D, device=DEVICE) * 0.1).transpose(
        -1, -2
    )
    return dict(
        **tensors,
        b=torch.randn(1, tokens, heads, device=DEVICE).bfloat16(),
        A_log=torch.randn(heads, device=DEVICE) * 0.1,
        dt_bias=torch.randn(heads * D, device=DEVICE) * 0.1,
        state=state,
        indices=torch.tensor([slot], dtype=torch.int32, device=DEVICE),
        cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32, device=DEVICE),
    )


def _clone_state(state):
    """Copy a state pool preserving its [V,K] backing.

    A plain ``.clone()`` makes the view contiguous, which both changes what the
    kernel reads and silently selects a different Triton specialization.
    """
    return state.transpose(-1, -2).clone().transpose(-1, -2)


def _run(launch, case):
    """Run one arm on a private copy of the state, returning output and state."""
    state = _clone_state(case["state"])
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
        prefill=True,
    )
    torch.npu.synchronize()
    return out.float(), state.float()


@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("tokens,slot", CASES)
def test_prefill_matches_triton_bit_exactly(heads, tokens, slot):
    case = _case(heads, tokens, slot)
    out, state = _run(kda_varlen_recurrent, case)
    ref_out, ref_state = _run(glm_kda_varlen_recurrent_npu, case)

    assert torch.equal(out, ref_out), (
        f"H={heads} T={tokens}: {int((out != ref_out).sum())}/{out.numel()} "
        f"outputs differ, max {(out - ref_out).abs().max().item():.3e}"
    )
    assert torch.equal(state, ref_state), (
        f"H={heads} T={tokens}: {int((state != ref_state).sum())}/"
        f"{state.numel()} state elements differ, max "
        f"{(state - ref_state).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("heads", HEADS)
def test_prefill_leaves_other_slots_untouched(heads):
    """Only the indexed slot is written; the rest of the pool is a live cache."""
    case = _case(heads, 256, 3)
    before = _clone_state(case["state"]).float()
    _, after = _run(kda_varlen_recurrent, case)

    other = [s for s in range(SLOTS) if s != 3]
    assert torch.equal(after[other], before[other])
    assert not torch.equal(after[3], before[3])
