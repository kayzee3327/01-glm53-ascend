"""Smoke test: verify prepared_recurrent is registered and callable."""

import torch
import torch_npu

from sgl_glm53_ascend_ops.ops import prepared_recurrent

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


if __name__ == "__main__":
    test_prepared_recurrent_runs()
    test_prepared_recurrent_track_state()
