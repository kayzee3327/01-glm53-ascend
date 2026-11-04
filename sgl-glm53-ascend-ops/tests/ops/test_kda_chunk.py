"""Numerical correctness of the chunked KDA kernel.

Checked against a PyTorch transcription of `kda_chunk_torch` from
`bench/kda_chunk/kda_python.py` — the reference this operator was written from.
The anchor accumulates in fp32 on the host and the kernel in fp32 on device,
so this is a tolerance comparison, not the equality the recurrent prefill
kernel holds: the kernel reassociates two things deliberately.

  * A and Aqk are factored so the Cube can compute them, with the gate rebased
    per 16 rows. The reference evaluates the same exponents pairwise.
  * The triangular inverse is repeated squaring rather than forward
    substitution, since the strictly-lower matrix is nilpotent.

Both were measured against the reference in fp64 before the kernel was written
(max abs 2.3e-9 on the operator's output), so the tolerance here is covering
fp32 rounding, not an algorithmic gap.

The cases cover what the kernel branches on: one chunk and several, one
sequence and several, the zero slot (which means a zero initial state rather
than a pool read), and every position of the tracked checkpoint — before the
first chunk, mid-chunk, on a chunk boundary, and at the very end.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from sgl_glm53_ascend_ops.ops import kda_varlen_chunk

D = 128
BT = 64
SLOTS = 8
SCALE = D**-0.5
LOWER_BOUND = -5.0
DEVICE = "npu"

# fp32 against fp32 with a different summation order. The kernel's output runs
# ~5e-4 in magnitude on these inputs, so atol is what binds there.
RTOL = 2e-3
ATOL = 2e-5


def _case(heads, cu_seqlens, slots, track=None, seed=0):
    torch.manual_seed(20261103 + seed)
    tokens = cu_seqlens[-1]
    tensors = {
        name: (torch.randn(1, tokens, heads, D, device=DEVICE) * 0.2).bfloat16()
        for name in ("q", "k", "v", "a")
    }
    # Allocated [slots,H,V,K] and viewed [slots,H,K,V] so stride(2)==1: the
    # one state layout that reaches the kernel in production. Nonzero, because
    # a zero slot is its own transpose and would hide a layout bug.
    state = (torch.randn(SLOTS, heads, D, D, device=DEVICE) * 0.1).transpose(
        -1, -2
    )
    return dict(
        **tensors,
        b=torch.randn(1, tokens, heads, device=DEVICE).bfloat16(),
        A_log=torch.randn(heads, device=DEVICE) * 0.1,
        dt_bias=torch.randn(heads, D, device=DEVICE) * 0.1,
        state=state,
        indices=torch.tensor(slots, dtype=torch.int32, device=DEVICE),
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32, device=DEVICE),
        track_indices=(
            None if track is None
            else torch.tensor(track[0], dtype=torch.int32, device=DEVICE)
        ),
        track_lens=(
            None if track is None
            else torch.tensor(track[1], dtype=torch.int32, device=DEVICE)
        ),
    )


def _clone_state(state):
    """Copy a pool preserving its [V,K] backing.

    A plain ``.clone()`` makes the view contiguous, which changes what the
    kernel reads.
    """
    return state.transpose(-1, -2).clone().transpose(-1, -2)


def _reference(case):
    """`kda_chunk_torch`, on the host in fp32, state updated in place."""
    q, k, v, a = (case[n].float() for n in ("q", "k", "v", "a"))
    b = case["b"].float()
    A_log, dt_bias = case["A_log"].float(), case["dt_bias"].float()
    state = case["state"]
    cu = case["cu_seqlens"].tolist()
    slots = case["indices"].tolist()
    track_idx = (
        None if case["track_indices"] is None
        else case["track_indices"].tolist()
    )
    track_len = (
        None if case["track_lens"] is None else case["track_lens"].tolist()
    )

    T, H = q.shape[1], q.shape[2]
    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * SCALE
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    g = LOWER_BOUND * torch.sigmoid(
        A_log.exp()[None, None, :, None] * (a + dt_bias[None, None])
    )
    beta = torch.sigmoid(b)

    out = torch.zeros(1, T, H, D, device=q.device)
    eye = torch.eye(BT, device=q.device)

    for seq in range(len(cu) - 1):
        bos, eos, slot = cu[seq], cu[seq + 1], slots[seq]
        ti = 0 if track_idx is None else track_idx[seq]
        tl = -1 if ti <= 0 else track_len[seq]

        for h in range(H):
            S = (
                state[slot, h].float().clone() if slot > 0
                else torch.zeros(D, D, device=q.device)
            )
            if ti > 0 and tl == 0:
                state[ti, h].copy_(S)

            for c0 in range(bos, eos, BT):
                sl = slice(c0, c0 + BT)
                qi, ki, vi = qn[0, sl, h], kn[0, sl, h], v[0, sl, h]
                bi = beta[0, sl, h]
                gi = g[0, sl, h].cumsum(0)

                A = torch.zeros(BT, BT, device=q.device)
                Aqk = torch.zeros(BT, BT, device=q.device)
                for i in range(BT):
                    A[:, i] = ((ki * (gi - gi[i : i + 1]).exp()) * ki[i]).sum(-1)
                    Aqk[:, i] = ((qi * (gi - gi[i : i + 1]).exp()) * ki[i]).sum(-1)
                Araw = -(A * bi[:, None]).tril(-1)
                Aqk = Aqk.tril(0)

                X = Araw.clone()
                for i in range(1, BT):
                    X[i, :i] = X[i, :i] + (X[i, :, None] * X[:, :i]).sum(-2)
                Ainv = (X + eye) * bi[None, :]

                w = Ainv @ (gi.exp() * ki)
                u = Ainv @ vi
                vnew = u - w @ S
                out[0, sl, h] = (qi * gi.exp()) @ S + Aqk @ vnew

                if ti > 0 and c0 - bos < tl <= c0 - bos + BT:
                    r = tl - (c0 - bos)
                    gr = gi[r - 1]
                    Sr = S * gr.exp()[:, None]
                    Sr = Sr + ((gr - gi[:r]).exp() * ki[:r]).T @ vnew[:r]
                    state[ti, h].copy_(Sr)

                gl = gi[-1]
                S = S * gl.exp()[:, None]
                S = S + ((gl - gi).exp() * ki).T @ vnew
            if slot > 0:
                state[slot, h].copy_(S)
    return out


def _run(case):
    state = _clone_state(case["state"])
    out = kda_varlen_chunk(
        A_log=case["A_log"],
        a=case["a"],
        dt_bias=case["dt_bias"],
        q=case["q"],
        k=case["k"],
        v=case["v"],
        b=case["b"],
        initial_state_source=state,
        initial_state_indices=case["indices"],
        cu_seqlens=case["cu_seqlens"],
        lower_bound=LOWER_BOUND,
        scale=SCALE,
        prefill=True,
        track_state_indices=case["track_indices"],
        track_state_lens=case["track_lens"],
    )
    torch.npu.synchronize()
    return out.float(), state.float()


# (label, heads, cu_seqlens, slots, track)
CASES = [
    ("one chunk", 1, [0, 64], [1], None),
    ("two chunks", 1, [0, 128], [1], None),
    ("four chunks", 2, [0, 256], [1], None),
    ("two sequences", 2, [0, 128, 256], [1, 2], None),
    ("ragged varlen", 1, [0, 64, 192, 256], [1, 2, 3], None),
    ("zero slot", 1, [0, 128], [0], None),
    ("track at 0", 2, [0, 128], [1], ([4], [0])),
    ("track mid-chunk", 2, [0, 128], [1], ([4], [32])),
    ("track on boundary", 2, [0, 128], [1], ([4], [64])),
    ("track at end", 1, [0, 192], [1], ([4], [192])),
]


@pytest.mark.parametrize(
    "label,heads,cu_seqlens,slots,track",
    CASES,
    ids=[c[0].replace(" ", "-") for c in CASES],
)
def test_chunk_matches_reference(label, heads, cu_seqlens, slots, track):
    case = _case(heads, cu_seqlens, slots, track, seed=len(label))
    out, state = _run(case)

    ref_case = dict(case, state=_clone_state(case["state"]))
    ref_out = _reference(ref_case)
    ref_state = ref_case["state"].float()

    torch.testing.assert_close(out, ref_out, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(state, ref_state, rtol=RTOL, atol=ATOL)


def test_chunk_leaves_other_slots_untouched():
    """Only the indexed slot is written; the rest of the pool is a live cache."""
    case = _case(2, [0, 128], [3])
    before = _clone_state(case["state"]).float()
    _, after = _run(case)

    other = [s for s in range(SLOTS) if s != 3]
    assert torch.equal(after[other], before[other])
    assert not torch.equal(after[3], before[3])


def test_chunk_rejects_unaligned_sequence_length():
    """Sequence lengths must be whole chunks; 100 is not."""
    case = _case(1, [0, 100], [1])
    with pytest.raises(RuntimeError, match="multiple of 64"):
        _run(case)
