# Why `RecurrentKda` is not an arm in this benchmark

CANN 9.2.0's `recurrent_kda` from
[ops-transformer](https://gitcode.com/cann/ops-transformer/tree/9.2.0/attention/recurrent_kda)
was wired in as a fourth arm, measured, and removed on 2026-10-27. It is kept
out because it is slow enough on this hardware to be uninformative, not because
it is wrong — it passed validation on every row it could run.

The sibling bench in `../kda_chunk` kept a CANN arm for a while, and that was
not an inconsistency: `ChunkKdaFwd` is 3-6x *faster* than the Triton arm there,
so it was worth comparing against, where this op is the opposite result. That
arm has since been removed too, for an unrelated reason — see that README.

## What it measured

910B4 (20 AIC / 40 AIV), card 4, CANN 9.0.0 toolkit with the op built from
9.2.0 source as a custom vendor package. `ours` is
`kda_varlen_recurrent`; all three arms PASS.

| row | ours (ms) | triton (ms) | cann (ms) | cann vs ours | cann vs triton |
| --- | --- | --- | --- | --- | --- |
| decode H16 B1 | 0.121 | 0.262 | 0.422 | 3.48x slower | 1.6x slower |
| decode H16 B32 | 0.133 | 0.277 | 0.535 | 4.01x slower | 1.9x slower |
| decode H16 B128 | 0.303 | 0.480 | 1.281 | 4.23x slower | 2.7x slower |
| decode H32 B1 | 0.118 | 0.263 | 0.450 | 3.81x slower | 1.7x slower |
| decode H32 B32 | 0.146 | 0.270 | 0.771 | 5.27x slower | 2.9x slower |
| decode H32 B128 | 0.633 | 0.943 | 2.257 | 3.57x slower | 2.4x slower |
| decode H64 B1 | 0.140 | 0.269 | 0.456 | 3.27x slower | 1.7x slower |
| decode H64 B32 | 0.301 | 0.485 | 1.247 | 4.14x slower | 2.6x slower |
| decode H64 B128 | 1.225 | 1.834 | 4.039 | 3.30x slower | 2.2x slower |

It is 3.3-5.3x slower than ours and ~2x slower than the generic Triton
fallback. An arm that loses to the fallback it is meant to replace does not
discriminate between the other two, which is the whole job of a reference here.

## It could only ever cover a third of the rows

Both exclusions are scope, not failure, and together they are most of why the
arm was not worth its maintenance cost.

**No checkpoint output.** It writes state per token into the pool and has no
per-step checkpoint tensor, so there is nothing to validate `verify` against —
and `verify` is the shape the deployed profile is dominated by.

**At most 8 tokens per sequence.** `RKDA_MAX_MTP = 8` in
`op_host/recurrent_kda_tiling_processor.h:49`, documented in
`aclnnRecurrentKda.md` as the limit for a dense sequence and for each varlen
segment alike. It is a decode/MTP kernel by design, so every `prefill` row
(1k-16k tokens) is out of scope.

That exclusion needs care rather than a failing row, because **the op does not
say no**: the kernel checks `cu_seqlens` on entry and returns without writing
anything. An over-long launch is a *successful* call that computed nothing, so
comparing against it measures whatever was in the output allocation — which can
read as a plausible diff, or even as a pass if the allocator handed back a freed
buffer still holding another arm's result. The removed `cann_supported()` gate
existed to draw that line.

## Measured against the *patched* op, which matters

`/data/wangkaize/ops-transformer` (branch 9.2.0) carries an uncommitted local
fix to `CalcComputeUbCoeff` in `recurrent_kda_tiling_processor.h`. Upstream
budgets `broadTmp` only when `K != 128`, but it is used on every path and the op
only accepts `K == 128`, so `tmpBuff` is under-reserved by ~30 KB of 192 KB and
the kernel writes past its UB allocation — upstream's own
`tests/pytest/test_accuracy.py` faults. The fix drops the conditional.

So the numbers above are the only honest ones: unpatched, the op either faults
the device or returns an unwritten buffer, and timing that would be timing
nothing.

**The fix costs a tile of V parallelism, and that is the first reason it is
slow.** On 910B4 (UB 192 KiB, K=V=128, HV=16, fp32 state,
`state_v_first=False`):

| build | vStep coefficient | vStep | repeatTime |
| --- | --- | --- | --- |
| upstream 9.2.0 (overflows UB) | 2570 | 64 | 2 |
| patched (what was measured) | 3082 | 48 | 3 |

It walks V=128 in three passes where a correct op with a tighter buffer plan
would do two. That is a genuine upstream defect rather than a tuning choice, but
the way out is a kernel change, not a host knob.

Two host-side knobs recover `repeatTime=2` even with the fix applied —
`state_v_first=True` (the K-first state transpose buffer at line 545 is what
costs the third pass) or a bf16 state. Neither closes a 3.3-5.3x gap, which is
why this was not pursued.

**A second, independent UB cost: the budget is sized for 8 tokens even when
there is 1.** `recurrent_kda_tiling_processor.h:519-531` multiplies every input
queue by `RKDA_MAX_MTP` unconditionally, and decode is one token per sequence.
This inflates the fixed and working reservations (12.9 KB / 29.8 KB here) on
every decode launch. It is not what pushed `repeatTime` from 2 to 3 — the patch
above is — but it shrinks the budget the tiling has to work within.

**Core occupancy is coarse.** `UpdateDynamicBlockDimByTaskUnits`
(`recurrent_kda_tiling_processor.h:479-490`) sets `blockDim = min(B*HV, aivNum)`
and nothing else — there is no value-dim split. H64/B128 is 8192 task units over
40 AIVs = **205 sequential waves**, each paying the repeatTime penalty above.
This is consistent with the one-head-per-core/BV=128 finding behind our own
kernel.

**The cubes are idle.** `op_kernel/recurrent_kda.cpp:30` is
`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`, so all 20 AIC sit out the
whole launch. Defensible for a recurrence, but it means the op cannot trade its
way out of a vector-bound regime.

## The local ops-transformer patch outlives this benchmark

Removing this arm does **not** make the `CalcComputeUbCoeff` fix removable. It is
still uncommitted in the working tree of `/data/wangkaize/ops-transformer`, and
a fresh clone or a checkout that resets that file brings the overflow back.
Upstream's own `recurrent_kda` pytest suite needs it to pass at all. (The
`kda_chunk` bench used to build `chunk_kda_fwd` from the same tree and no longer
does, so that is one fewer consumer, not zero.)

Diagnosing a recurrence: an all-NaN `out` whose NaNs are confined to even V
columns at ~1/256 density means the kernel never wrote the buffer, i.e. the
patch is missing. Re-apply, then rebuild.

## Restoring the arm

`git log -- bench/kda_recurrent/` — the arm was `glm_kda_varlen_recurrent_cann`
plus `cann_supported`/`cann_state` in `bench_kda_varlen_recurrent.py`, and two
helper files, `cann_op.py` and `build_cann_op.py`. `../kda_chunk` used to carry
the same two scripts apart from `OP_NAME`, `VENDOR` and `NEEDS_WHEEL`, and was
the better starting point than this history — they were deleted with that
bench's arms, so both copies now live only in `git log`.

Two things about the build are worth not rediscovering:

* `recurrent_kda` needs the vendor wheel (`NEEDS_WHEEL = True`) — its Python
  entry point is not in the source tree, unlike `chunk_kda_fwd`, which needed a
  hand-written bridge instead.
* The two ops must install under *different* vendor names. The installer clears
  same-named files out of the vendor directory, and every single-op build ships
  its own `libcust_opapi.so`, so a shared name means whichever built second
  silently deletes the other's operator.

The argument mapping that took the longest to get right: `layout="TND"` (our
tensors are `[1,T,H,D]`, which is TND with the leading 1 squeezed, not BSND —
BSND reads `T` as the batch); `ssm_state_indices` is per *token*, not per
sequence, so per-sequence indices need `repeat_interleave`; `safe_gate=True`
selects `lower_bound * sigmoid(exp(A_log) * x)`, which is what sglang's
preparation computes and therefore the setting that makes the two arms the same
function.
