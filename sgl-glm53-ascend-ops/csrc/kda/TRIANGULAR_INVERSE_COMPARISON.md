# The triangular inverse: our Ascend C kernel vs fla's triton_ascend

Both kernels invert the same matrix: the strictly-lower-triangular `Araw`
built from `Akk`, where the chunked KDA forward needs
`(I - Araw)^-1` to form the WY representation. They use **different
algorithms**, and the difference in arithmetic is 22.7x — in the direction
that favours fla.

Compared at the deployed shape: `BT = 64`, `K = V = 128`, bf16 in/out,
fp32 accumulation. Code read at
`fla/ops/kda/backends/triton_ascend/chunk_intra.py` and
`csrc/kda/kda_varlen_chunk_kernel.cpp`.

## The two procedures

### Ours — repeated squaring on nilpotency

`Araw` is strictly lower triangular, so `Araw^64 = 0`, and every `k < 64`
has a unique binary expansion:

```
(I - Araw)^-1 = sum_k Araw^k = prod_s (I + Araw^(2^s)),  s = 0..5
```

Six steps, each two `[64,64] @ [64,64]` Cube matmuls — `X <- X @ (I + D)`
and `D <- D @ D`, the squaring skipped on the last step. Eleven matmuls
total, in [`RunAic`](kda_varlen_chunk_kernel.cpp#L1336-L1348).

Between steps the AIV only renames buffers and adds the identity, in
[`AdvanceInverse`](kda_varlen_chunk_kernel.cpp#L956-L975) — no arithmetic on
the matrix itself.

### fla — blocked forward substitution

`BC = SUB_CHUNK_SIZE = 32` when `BT >= 64 and K <= 256`, so `NC = 2` sub-blocks
at our shape. Two stages, two separate kernel launches:

1. **`chunk_kda_fwd_kernel_diag_solve_npu`**
   ([chunk_intra.py:180-185](../../../../../flash-linear-attention/fla/ops/kda/backends/triton_ascend/chunk_intra.py#L180-L185))
   — literal forward substitution on each 32x32 diagonal block, a scalar
   loop of `BC` steps doing `b_a += tl.sum(b_a[:, None] * b_Ai, 0)`, then
   `b_Ai += m_I`. This is `kda_python.py`'s reference recurrence, in register.

2. **`chunk_kda_fwd_kernel_inter_solve_fused_npu`**
   ([chunk_intra.py:477-481](../../../../../flash-linear-attention/fla/ops/kda/backends/triton_ascend/chunk_intra.py#L477-L481))
   — merges the off-diagonal block algebraically. At `NC = 2` the whole merge
   is one statement:

   ```python
   b_Ai10 = -dot(dot(b_Ai11, b_Akk10), b_Ai00)
   ```

   The `NC >= 3` / `NC >= 4` branches (`b_Ai20`, `b_Ai31`, `b_Ai30`, each
   summing two or three products before the outer multiply) are dead at
   chunk size 64 — they exist for `BC = 16`.

## Arithmetic

MAC counts at `BT = 64`, in units of `64^3`:

| | matmuls | MACs | vs 64^3 |
|---|---|---|---|
| **ours** — 11 x `64^3` | 11 | 2,883,584 | 11.00 |
| **fla** — diagonal substitution | — | 61,440 | 0.23 |
| **fla** — 2 x `32^3` merge | 2 | 65,536 | 0.25 |
| **fla** total | 2 | 126,976 | **0.48** |

**22.7x more arithmetic on our side.** Repeated squaring treats the matrix as
dense: every one of the 11 matmuls is full `64x64x64`, even though `Araw^(2^s)`
gets sparser each step and half of every operand is structurally zero.
Substitution exploits the triangularity directly.

## Parallel structure

This is the larger difference, and it does not show up in the FLOP count.

| | ours | fla |
|---|---|---|
| chunks in flight | 1 per block, serial | all, gridded |
| inverse launches | — (inline) | 2 kernels |
| ordering cost | **6 cross-core rendezvous** | none |
| where the math runs | Cube, operands via GM | vector core, in register |

Our six squaring steps are serial *inside one block*, and each one costs a
`CrossCoreSetFlag`/`CrossCoreWaitFlag` round trip because the AIV has to
build the next factor between Cube calls. fla's two solve kernels launch on
grids over `(nt, nc, bh)` — every chunk's inverse runs concurrently, and no
handshake exists because the stages are separate launches.

fla also keeps the substitution off the Cube entirely: `tl.dot` appears only
in the block merge. The diagonal solve is register-resident vector work.

## Precision

Ours is the more conservative of the two. We keep the inverse in fp32 in the
workspace from end to end. fla accumulates the diagonal blocks in fp32
(`Akkd` is `torch.float32`) but stores the merged result to `Akk`, allocated
`dtype=k.dtype` — **bf16** on these inputs. Both pass against the fp32 anchor
at the bench tolerance, so this is a margin difference, not a correctness one.

Our 11-matmul chain does accumulate over more steps, but each is an exact
fp32 Cube product and the operands are bounded by the gate rebase, so the
depth has not cost measurable accuracy: 12/12 chunk tests pass with worst
case 0.85 of tolerance, and the residual there is bf16 output quantization.

## What this is worth, measured

**The FLOP gap is not the bottleneck.** Measured on card 4, H8 T4096:

- The entire 6-step inverse is **~25 of 115 us per chunk**. Cutting
  `kInverseSteps` from 6 to 3 (wrong answer, pure timing) saved 12.5 us/chunk.
- `aic_mac_time` is **5.2%** of AIC time. The Cube is idle, not saturated.
- AIC accounted work (43 us/chunk) + AIV accounted work (63 us) ~= the 115 us
  wall, so the two cores never overlap — the 9 rendezvous per chunk serialize
  them and we pay the sum rather than the max.

So eliminating 95% of the inverse's arithmetic would buy at most a few percent.
This matches the earlier Triton measurement that hierarchical inversion cuts
78% of FLOPs and runs **5-8% slower**: the inverse is not where the time goes,
on Ascend C either.

**What fla's structure actually buys is point 3, not point 2.** Our latency is
linear in T (2.05 / 3.92 / 7.09 ms at T1024/2048/4096) while fla's is nearly
flat (2.00 / 2.05 / 2.34), because `blockDim = num_req * num_heads` gives us
8 blocks at H8 against the card's 20 AIC, with the whole chunk loop serial
inside each. fla splits the forward into five launches, each gridded over all
chunks, and only the state scan is chunk-serial.

## Conclusion

Adopting fla's inversion algorithm is **not** the lever. It is a 22.7x
arithmetic cut on a stage that costs ~22% of the chunk, running on a unit
that is 5% utilized, and the same change measured slower in Triton.

The lever is the parallel decomposition: decouple the chunk-serial stages so
only the state scan is serial in chunks, and raise `blockDim` past 20. That
would change `blockDim` and workspace sizing but not the op signature. If that
restructure happens, the inverse's 6 rendezvous disappear with it — at which
point the algorithm choice becomes a second-order question about whether the
Cube or the vector core should own the stage.
