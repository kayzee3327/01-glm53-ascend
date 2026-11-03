# recurrent KDA varlen benchmark

Compares our Ascend C KDA varlen recurrence against the two paths the caller
has for the same work, on the shapes GLM-5.3 serving actually runs.

| arm | what it is | status |
| --- | --- | --- |
| `ascendc` | `kda_varlen_recurrent` from the sglang GLM-5.3 plugin — the target | working |
| `triton` | the generic Triton recurrence, one program per value tile per head, BV=64 | working |
| `prepared` | FP32 input preparation + the Ascend C `prepared_recurrent` kernel, the deployed prefill path | working; prefill rows only |

The target is ours, so every speedup column reads as "ours vs theirs". Both
references are the caller's own paths, which is the whole arm list now.
`prepared` is the comparison that decides whether the kernel is worth shipping:
it is the path production takes today, roughly 3x the generic one, so beating
`triton` alone does not settle anything. Its preparation is counted in, because
the caller pays for it — it exists only to feed that kernel.

## Removed outside arms

Both are out, and neither is coming back as part of the current work.

CANN's `RecurrentKda` was evaluated and dropped on its measurements. See
[DISCARDED-cann-recurrent-kda.md](DISCARDED-cann-recurrent-kda.md), which also
covers the ops-transformer patch that outlives the arm.

`fla` — flash-linear-attention-npu's `recurrent_kda` — was removed alongside the
chunk bench's outside arms; `git log -- bench/kda_recurrent/` has it. Its result
is worth keeping even though the arm is not, because it was the one arm that was
not our own code and it reached our own conclusion independently:

It was 4.4-9.1x slower on prefill and at parity on decode, and the split between
those is the finding. Prefill time was linear in tokens and **flat in heads**
(H4 T1k and H8 T1k both ~5.5ms, so doubling the work cost nothing) — the
signature of a launch that does not fill the device.
`UpdateDynamicBlockDimByTaskUnits` in its tiling sets `blockDim = seq_num * HV`,
and a packed prefill is one sequence, so H4 occupied 4 of 40 vector cores.
Decode is where that stopped binding and the gap closed completely: 1.70x at H16
B1 down to **0.96x at H32 B128 and 0.97x at H64 B128**, marginally *faster* than
ours on the two shapes with the most task units to spread. So the prefill gap
was a parallelization decision, not a slower recurrence — the same
one-head-per-core/BV=128 conclusion our own kernel is built around.

## Run it

```bash
python bench_kda_varlen_recurrent.py
```

No options. Every parameter is a module constant at the top of the script, so a
run is reproducible from the file alone and two runs are always comparable — to
measure something else, edit the constants.

Operator work on this machine is restricted to physical card 4, so run it as
`ASCEND_RT_VISIBLE_DEVICES=4 python bench_kda_varlen_recurrent.py` — `DEVICE`
below is `npu:0`, an index into the *visible* set.

The sibling benchmark in `../kda_chunk` covers the chunked KDA prefill op. It is
now a bare frame around one unimplemented arm, and its README records what the
outside kernels measured there before they were removed.

## Regimes

Shapes are the deployed parallel strategies, `H = 64 / attn_tp_size`:

| regime | what it exercises |
| --- | --- |
| `prefill` | H=4/8 (TP16/TP8) over 1k–16k packed tokens, one sequence |
| `decode` | one token per request, no checkpoints — `forward_decode`, at H=16/32/64 and 1/32/128 requests |
| `verify` | `draft_token_num=4` tokens per request with the state checkpointed after every step — EAGLE target verification, and the shape the profile is dominated by |

The `verify` rows are the profiled deployment maxima rather than a sweep; what
varies across them is the block count the kernel gets (`requests * heads`)
against 40 vector cores. `req` there is per DP replica, which is what one launch
sees — not the deployment-wide concurrency the serving configs are quoted in. See
`VERIFY_SHAPES`, which records both.

Every row validates the state pool alongside the output, and `verify` validates
the checkpoint buffer too — the only output of that path the caller reads, since
a checkpointing launch deliberately leaves the state slot alone. They are
in-place arguments, so a kernel that computes the right output while corrupting
a neighbouring slot fails here.

## Tolerance

`RTOL = 2e-3`, `ATOL = 1e-6`, one bar for every arm now that all of them are our
own code.

Outputs here are small — `|out|` runs ~1e-4 with a ~1e-2 tail, because q and k
are L2-normalized before the scale. So `atol` has to stay well under that or it
passes everything on its own and the comparison means nothing; the real bar is
`rtol`.

The prepared prefill path is bit-exact against the generic kernel, so a correct
Ascend C recurrence should be too: tighten both to 0 once the kernel is expected
to match exactly. Both arms being our own code is what makes that reachable.

The removed `fla` arm needed its own looser bar, and why is worth remembering
before adding another outside arm: a separately written kernel landed 1-2 bf16
ulps out on ~20 of 2M output elements while its state pool matched inside `RTOL`
everywhere — a different summation order, not a different recurrence.
`Reference` takes a per-arm `comparator` for that case. It matters because
measurement is gated on validation: holding such an arm to the strict bar fails
the row over rounding and discards *every* arm's timings with it.
