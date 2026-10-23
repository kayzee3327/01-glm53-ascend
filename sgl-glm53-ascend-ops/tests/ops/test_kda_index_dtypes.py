"""The op's index metadata must give the same answer in int32 and int64.

Every kernel reads `initial_state_indices`, `cu_seqlens` and the two tracking
tensors as `int32_t`, while the op accepts int64 because upstream's callers
pass it -- `glm_kda_varlen_recurrent_npu` validates them as int32-or-int64, and
the Ascend backend holds `mamba_track_indices` in int64. Without a host-side
narrowing an int64 buffer is misread rather than rejected, and the failure is
silent: on little-endian the first element's low word aliases the correct
value, so a one-request `initial_state_indices` works by luck while
`cu_seqlens[1]` lands on a high word and reads as 0. The kernel then walks zero
tokens and returns the zero-filled output allocation, with no error raised.

So these compare int64 against int32 rather than against a recurrence
reference: the dtype must not change the result, and the all-zero output is
called out separately because that is the shape the regression takes.
"""

import itertools

import pytest
import torch
import torch_npu  # noqa: F401

from sgl_glm53_ascend_ops.ops import kda_varlen_recurrent

D = 128
DEVICE = "npu"
TOKENS = 257
HEADS = 4
SLOT = 3


def make_case(index_dtype, starts_dtype, track_dtype=None):
    torch.manual_seed(5)
    tensors = {
        name: (torch.randn(1, TOKENS, HEADS, D, device=DEVICE) * 0.2).bfloat16()
        for name in ("q", "k", "v", "a")
    }
    tensors["b"] = torch.randn(1, TOKENS, HEADS, device=DEVICE).bfloat16()
    # The native pool: allocated [slots,H,V,K] and viewed [slots,H,K,V], so
    # stride(2)==1 as the op requires.
    pool = (torch.randn(6, HEADS, D, D, device=DEVICE) * 0.1).transpose(-1, -2)
    state = pool.transpose(-1, -2).clone().transpose(-1, -2)
    kwargs = dict(
        **tensors,
        A_log=torch.randn(HEADS, device=DEVICE) * 0.1,
        dt_bias=torch.randn(HEADS * D, device=DEVICE) * 0.1,
        lower_bound=-5.0,
        scale=D**-0.5,
        initial_state_source=state,
        initial_state_indices=torch.tensor([SLOT], dtype=index_dtype, device=DEVICE),
        cu_seqlens=torch.tensor([0, TOKENS], dtype=starts_dtype, device=DEVICE),
        prefill=True,
    )
    if track_dtype is not None:
        kwargs.update(
            track_state_indices=torch.tensor([5], dtype=track_dtype, device=DEVICE),
            track_state_lens=torch.tensor([64], dtype=track_dtype, device=DEVICE),
        )
    return kwargs, state


def run(**kwargs):
    out = kda_varlen_recurrent(**kwargs)
    torch.npu.synchronize()
    return out


@pytest.mark.parametrize(
    "index_dtype,starts_dtype",
    [
        pair
        for pair in itertools.product((torch.int32, torch.int64), repeat=2)
        if pair != (torch.int32, torch.int32)
    ],
)
def test_index_dtype_does_not_change_result(index_dtype, starts_dtype):
    base_kwargs, base_state = make_case(torch.int32, torch.int32)
    expected = run(**base_kwargs)

    kwargs, state = make_case(index_dtype, starts_dtype)
    actual = run(**kwargs)

    # The regression's signature, named separately so a failure says which it
    # was: an untouched output allocation rather than a small numeric drift.
    assert actual.float().abs().sum().item() > 0, "op returned its zeroed output"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, base_state, rtol=0, atol=0)


@pytest.mark.parametrize("track_dtype", (torch.int32, torch.int64))
def test_track_dtype_does_not_change_checkpoint(track_dtype):
    base_kwargs, base_state = make_case(torch.int32, torch.int32, torch.int32)
    expected = run(**base_kwargs)

    kwargs, state = make_case(torch.int32, torch.int32, track_dtype)
    actual = run(**kwargs)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Slot 5 is the checkpoint destination; a misread slot index would write
    # elsewhere and leave it untouched.
    torch.testing.assert_close(state, base_state, rtol=0, atol=0)
    assert state[5].float().abs().sum().item() > 0, "checkpoint slot unwritten"
