"""Smoke tests: verify the KDA ops are registered and callable."""

import pytest
import torch
import torch_npu

from sgl_glm53_ascend_ops.ops import kda_varlen_recurrent, prepared_recurrent

H = 4
D = 128
NUM_REQ = 1
DEVICE = "npu"


def test_prepared_recurrent_runs():
    num_tokens = 4

    qn = torch.randn(num_tokens * H, D, dtype=torch.float32, device=DEVICE)
    kn = torch.randn(num_tokens * H, D, dtype=torch.float32, device=DEVICE)
    v = torch.randn(num_tokens * H, D, dtype=torch.bfloat16, device=DEVICE)
    gdecay = torch.randn(num_tokens * H, D, dtype=torch.float32, device=DEVICE)
    beta = torch.randn(num_tokens * H, dtype=torch.float32, device=DEVICE)

    out = torch.zeros(num_tokens * H, D, dtype=torch.bfloat16, device=DEVICE)
    state = torch.zeros(1, H, D, D, dtype=torch.float32, device=DEVICE)

    indices = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    starts = torch.tensor([0, num_tokens], dtype=torch.int32, device=DEVICE)

    prepared_recurrent(
        qn=qn,
        kn=kn,
        v=v,
        gdecay=gdecay,
        beta=beta,
        out=out,
        state=state,
        indices=indices,
        starts=starts,
    )
    torch.npu.synchronize()
    print("prepared_recurrent (no tracking): passed")


def test_prepared_recurrent_track_state():
    num_tokens = 4

    qn = torch.randn(num_tokens * H, D, dtype=torch.float32, device=DEVICE)
    kn = torch.randn(num_tokens * H, D, dtype=torch.float32, device=DEVICE)
    v = torch.randn(num_tokens * H, D, dtype=torch.bfloat16, device=DEVICE)
    gdecay = torch.randn(num_tokens * H, D, dtype=torch.float32, device=DEVICE)
    beta = torch.randn(num_tokens * H, dtype=torch.float32, device=DEVICE)

    out = torch.zeros(num_tokens * H, D, dtype=torch.bfloat16, device=DEVICE)
    state = torch.zeros(1, H, D, D, dtype=torch.float32, device=DEVICE)

    indices = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    starts = torch.tensor([0, num_tokens], dtype=torch.int32, device=DEVICE)

    track_indices = torch.zeros(NUM_REQ, dtype=torch.int32, device=DEVICE)
    track_lens = torch.tensor([num_tokens] * NUM_REQ, dtype=torch.int32, device=DEVICE)

    prepared_recurrent(
        qn=qn,
        kn=kn,
        v=v,
        gdecay=gdecay,
        beta=beta,
        out=out,
        state=state,
        indices=indices,
        starts=starts,
        track_indices=track_indices,
        track_lens=track_lens,
        track_state=True,
    )
    torch.npu.synchronize()
    print("prepared_recurrent (track_state=True): passed")


def _varlen_case(heads, lengths, prefill):
    """Inputs that satisfy every precondition `kda_varlen_recurrent` checks.

    The caller is responsible for upstream's shape contract, so these mirror
    what `glm_kda_varlen_recurrent_npu` would pass at the selection point: bf16
    inputs and an FP32 state pool that is K-contiguous.
    """
    tokens = sum(lengths)
    batch = len(lengths)
    kwargs = {
        name: torch.randn(1, tokens, heads, D, dtype=torch.bfloat16, device=DEVICE)
        for name in ("q", "k", "v", "a")
    }
    # [slots,H,V,K] contiguous, transposed to [slots,H,K,V] — stride(2)==1 and
    # stride(3)==128, the one layout the kernels accept. A contiguous tensor of
    # the final shape has these the other way round.
    state = torch.zeros(batch + 1, heads, D, D, dtype=torch.float32, device=DEVICE)
    return dict(
        **kwargs,
        b=torch.randn(1, tokens, heads, dtype=torch.bfloat16, device=DEVICE),
        A_log=torch.randn(heads, dtype=torch.float32, device=DEVICE),
        dt_bias=torch.randn(heads * D, dtype=torch.float32, device=DEVICE),
        initial_state_source=state.transpose(-1, -2),
        initial_state_indices=torch.arange(
            1, batch + 1, dtype=torch.int32, device=DEVICE
        ),
        cu_seqlens=torch.tensor(
            [sum(lengths[:i]) for i in range(batch + 1)],
            dtype=torch.int32,
            device=DEVICE,
        ),
        lower_bound=-8.0,
        scale=D**-0.5,
        prefill=prefill,
    )


# (heads, lengths, prefill) -> reaches a launcher. Prefill takes one packed
# sequence at the TP16/TP8 widths; decode takes one token per request at
# TP4/TP2/TP1. Together these are the whole supported space.
#
# H=4 prefill computes; the rest still stop at their kernel TODO. Numerical
# correctness for H=4 lives in bench/kda_recurrent, which diffs elementwise
# against the Triton baseline.
IMPLEMENTED = [
    (4, (128,), True),
    (8, (128,), True),
]

STUBBED = [
    (16, (1,) * 3, False),
    (32, (1,) * 3, False),
    (64, (1,) * 3, False),
]


@pytest.mark.parametrize("heads,lengths,prefill", IMPLEMENTED)
def test_varlen_recurrent_dispatch_runs(heads, lengths, prefill):
    """A supported shape with a live kernel dispatches and returns an output."""
    case = _varlen_case(heads, lengths, prefill)
    out = kda_varlen_recurrent(**case)
    torch.npu.synchronize()
    assert out.shape == case["q"].shape
    assert out.dtype == case["q"].dtype
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize("heads,lengths,prefill", STUBBED)
def test_varlen_recurrent_dispatch_reaches_a_kernel(heads, lengths, prefill):
    """A supported shape selects a launcher and stops at the kernel TODO.

    While a kernel is still a stub, "not implemented yet" IS the success
    signal: it is raised only after dispatch has picked a launcher, so any
    other message means the shape was rejected before selection.
    """
    with pytest.raises(RuntimeError, match="not implemented yet"):
        kda_varlen_recurrent(**_varlen_case(heads, lengths, prefill))


@pytest.mark.parametrize(
    "heads,lengths,prefill,message",
    [
        # Widths outside the deployed TP set, and the right width in the wrong
        # regime — H=8 is prefill-only, H=16 is decode-only.
        (2, (128,), True, "prefill supports"),
        (16, (128,), True, "prefill supports"),
        (8, (1,) * 3, False, "decode supports"),
        # Prefill is single-sequence; decode is one token per request.
        (8, (128, 128), True, "one packed sequence"),
        (16, (4,) * 3, False, "one token per request"),
    ],
)
def test_varlen_recurrent_rejects_unsupported(heads, lengths, prefill, message):
    """Shapes no launcher covers are rejected, and say which rule they broke.

    The caller keeps its own fallback for these, so the message has to name the
    constraint rather than report a generic failure.
    """
    with pytest.raises(RuntimeError, match=message):
        kda_varlen_recurrent(**_varlen_case(heads, lengths, prefill))


def test_varlen_recurrent_rejects_intermediate_state():
    """Target verify has no per-token checkpoint path; refuse it explicitly."""
    case = _varlen_case(8, (128,), True)
    case["intermediate_state"] = torch.zeros(
        1, 128, 8, D, D, dtype=torch.float32, device=DEVICE
    )
    with pytest.raises(RuntimeError, match="does not support intermediate_state"):
        kda_varlen_recurrent(**case)


def test_varlen_recurrent_rejects_transposed_state():
    """V-contiguous state is the layout the generic Triton kernel gets wrong.

    It must be refused rather than read: the kernels assume stride(2)==1, so a
    transposed pool would silently produce different values.
    """
    case = _varlen_case(8, (128,), True)
    case["initial_state_source"] = case["initial_state_source"].transpose(-1, -2)
    with pytest.raises(RuntimeError, match="K-contiguous"):
        kda_varlen_recurrent(**case)


if __name__ == "__main__":
    test_prepared_recurrent_runs()
    test_prepared_recurrent_track_state()
    for _heads, _lengths, _prefill in IMPLEMENTED:
        test_varlen_recurrent_dispatch_runs(_heads, _lengths, _prefill)
    for _heads, _lengths, _prefill in STUBBED:
        test_varlen_recurrent_dispatch_reaches_a_kernel(_heads, _lengths, _prefill)
    print("kda_varlen_recurrent dispatch: passed")
