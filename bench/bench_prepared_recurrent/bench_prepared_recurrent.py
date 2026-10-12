from benchmark import (
    benchmark,
    BenchmarkConfig,
    print_results_table,
    NpuEventMeasurer,
    Reference,
    Comparator,
)

import triton
import triton.language as tl
import math
import torch
import torch_npu

device = "npu"


@triton.jit
def _prepared_recurrent(
    QN,
    KN,
    V,
    GDECAY,
    BETA,
    OUT,
    STATE,
    INDICES,
    STARTS,
    TRACK_INDICES,
    TRACK_LENS,
    TRACK_STATE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BV: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
):
    iv, req, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bos = tl.load(STARTS + req).to(tl.int64)
    eos = tl.load(STARTS + req + 1).to(tl.int64)
    ks = tl.arange(0, D)
    vs = iv * BV + tl.arange(0, BV)
    slot = tl.load(INDICES + req)
    pstate = STATE + slot * S0 + head * S1 + ks[None, :] * SK + vs[:, None] * SV
    state = tl.zeros((BV, D), tl.float32)
    if slot > 0:
        state = tl.load(pstate).to(tl.float32)
    if TRACK_STATE:
        track_slot = tl.load(TRACK_INDICES + req)
        track_len = tl.load(TRACK_LENS + req)
        ptrack = (
            STATE + track_slot * S0 + head * S1 + ks[None, :] * SK + vs[:, None] * SV
        )
        if track_slot > 0 and track_len == 0:
            tl.store(ptrack, state)
    for i in range(eos - bos):
        row = (bos + i) * H + head
        q = tl.load(QN + row * D + ks)
        k = tl.load(KN + row * D + ks)
        v = tl.load(V + row * D + vs).to(tl.float32)
        decay = tl.load(GDECAY + row * D + ks)
        beta = tl.load(BETA + row)
        state *= decay[None, :]
        v -= tl.sum(state * k[None, :], axis=1)
        v *= beta
        state += k[None, :] * v[:, None]
        if TRACK_STATE:
            if track_slot > 0 and i + 1 == track_len:
                tl.store(ptrack, state)
        out = tl.sum(state * q[None, :], axis=1)
        tl.store(OUT + row * D + vs, out.to(OUT.dtype.element_ty))
    if slot > 0:
        tl.store(pstate, state)


from sgl_glm53_ascend_ops.ops import prepared_recurrent as _prepared_recurrent_torch


class _prepared_recurrent_opt:
    """Drop-in for ``kernel[grid](...)`` syntax, forwards to the Ascend C torch op."""

    def __getitem__(self, grid):
        return self._launch

    @staticmethod
    def _launch(
        QN, KN, V, GDECAY, BETA, OUT, STATE,
        INDICES, STARTS, TRACK_INDICES, TRACK_LENS,
        TRACK_STATE, H, D, BV, S0, S1, SK, SV,
        HEAD_OFFSET=0,
        **kwargs,
    ):
        _prepared_recurrent_torch(
            qn=QN, kn=KN, v=V, gdecay=GDECAY, beta=BETA,
            out=OUT, state=STATE, indices=INDICES, starts=STARTS,
            track_indices=TRACK_INDICES if TRACK_STATE else None,
            track_lens=TRACK_LENS if TRACK_STATE else None,
            track_state=TRACK_STATE,
            head_offset=HEAD_OFFSET,
        )


_prepared_recurrent_opt = _prepared_recurrent_opt()


TOKEN_COUNTS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


def bench_one(tokens: int, track_state: bool = False):
    N = 3 if track_state else 2
    requests = 1
    B = requests
    H, D = 4, 128
    BV = 16

    QN = torch.randn([B, tokens, H, D], device=device, dtype=torch.float32) * 0.01
    KN = torch.randn([B, tokens, H, D], device=device, dtype=torch.float32) * 0.01
    V = (
        torch.randn([B, tokens, H, D], device=device, dtype=torch.float32) * 0.01
    ).to(torch.bfloat16)
    GDECAY = torch.sigmoid(
        torch.randn([B, tokens, H, D], device=device, dtype=torch.float32)
    )
    BETA = torch.randn([B, tokens, H], device=device, dtype=torch.float32) * 0.01
    OUT = torch.empty([B, tokens, H, D], device=device, dtype=torch.bfloat16)
    STATE = torch.randn([N, H, D, D], device=device, dtype=torch.float32) * 0.01
    INDICES = torch.tensor([1], device=device, dtype=torch.int32)
    STARTS = torch.tensor([0, tokens], device=device, dtype=torch.int32)

    if track_state:
        TRACK_INDICES = torch.tensor([2], device=device, dtype=torch.int32)
        TRACK_LENS = torch.tensor([tokens // 2], device=device, dtype=torch.int32)
    else:
        TRACK_INDICES = torch.zeros([1], device=device, dtype=torch.int32)
        TRACK_LENS = torch.zeros([1], device=device, dtype=torch.int32)

    TRACK_STATE = track_state

    grid = (triton.cdiv(D, BV), 1, H)

    OUT_ref = OUT.clone().detach()
    OUT_target = OUT.clone().detach()
    STATE_ref = STATE.clone().detach()
    STATE_target = STATE.clone().detach()

    def call_target():
        _prepared_recurrent_opt[grid](
            QN, KN, V, GDECAY, BETA, OUT_target, STATE_target,
            INDICES, STARTS, TRACK_INDICES, TRACK_LENS,
            TRACK_STATE, H, D, BV, *STATE.stride(),
        )

    def call_ref():
        _prepared_recurrent[grid](
            QN, KN, V, GDECAY, BETA, OUT_ref, STATE_ref,
            INDICES, STARTS, TRACK_INDICES, TRACK_LENS,
            TRACK_STATE, H, D, BV, *STATE.stride(),
            num_warps=1, num_stages=3, multibuffer=True,
        )

    return benchmark(
        call_target,
        references=[Reference("triton", call_ref, lambda: (OUT_ref.clone(), STATE_ref.clone()))],
        measurer=NpuEventMeasurer(),
        config=BenchmarkConfig(10, 20, 20),
        output_fn=lambda: (OUT_target.clone(), STATE_target.clone()),
    )


def main():
    results = []
    for tokens in TOKEN_COUNTS:
        label = f"{tokens // 1024}k" if tokens >= 1024 else str(tokens)
        print(f">>> benchmarking tokens={label} (no tracking) ...")
        res = bench_one(tokens, track_state=False)
        results.append((label, res))

    print()
    print("=" * 60)
    print("prepared_recurrent (no tracking) — ascend_c vs triton")
    print("=" * 60)
    print_results_table(results)

    results_track = []
    for tokens in TOKEN_COUNTS:
        label = f"{tokens // 1024}k" if tokens >= 1024 else str(tokens)
        print(f">>> benchmarking tokens={label} (track_state) ...")
        res = bench_one(tokens, track_state=True)
        results_track.append((label, res))

    print()
    print("=" * 60)
    print("prepared_recurrent (track_state) — ascend_c vs triton")
    print("=" * 60)
    print_results_table(results_track)


if __name__ == "__main__":
    main()
