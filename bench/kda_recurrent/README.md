# recurrent KDA varlen benchmark

Compares our Ascend C KDA varlen recurrence against the two paths the caller
has for the same work, on the shapes GLM-5.3 serving actually runs.

| arm | what it is | status |
| --- | --- | --- |
| `ascendc` | `kda_varlen_recurrent` from the sglang GLM-5.3 plugin — the target | working |
| `triton` | the generic Triton recurrence, one program per value tile per head, BV=64 | working |
| `prepared` | FP32 input preparation + the Ascend C `prepared_recurrent` kernel, the deployed prefill path | working; prefill rows only |
| `fla` | `recurrent_kda` from flash-linear-attention-npu (Tianjin University) | working; no `verify` rows |

The target is ours, so every speedup column reads as "ours vs theirs".
`prepared` is the comparison that decides whether the kernel is worth shipping:
it is the path production takes today, roughly 3x the generic one, so beating
`triton` alone does not settle anything. Its preparation is counted in, because
the caller pays for it — it exists only to feed that kernel.

`fla` is the only arm that is not our own code, which is the reason it is here:
`triton` and `prepared` tell us whether the kernel beats the paths this caller
already has, and neither can tell us whether a gap is the op or the hardware.
It computes the gate, the L2 norm and the beta sigmoid in-kernel from the
model's own tensors, so it is the same function with the same inputs.

It is 4.4-9.1x slower on prefill and roughly at parity on decode — and the
split between those two is the useful result, not either number alone.

Prefill time is linear in tokens and **flat in heads**: H4 T1k and H8 T1k both
take ~5.5ms, so doubling the work costs nothing. That is the signature of a
launch that does not fill the device.
`UpdateDynamicBlockDimByTaskUnits` in its tiling sets
`blockDim = seq_num * HV`, and a packed prefill is one sequence, so H4 occupies
4 of 40 vector cores and H8 occupies 8.

Decode is where that stops binding, and the gap closes completely: 1.70x at
H16 B1 down to **0.96x at H32 B128 and 0.97x at H64 B128** — i.e. it is
marginally *faster* than ours on the two shapes with the most task units to
spread. So the prefill gap is a parallelization decision, not a slower
recurrence; given enough independent work the two kernels land within a few
percent. That is the same one-head-per-core/BV=128 conclusion our own kernel is
built around, reached independently.

Those decode figures are worth treating as parity rather than as a ranking:
a few percent either way is within what a differently-tuned block mapping
explains, and the `verify` shapes the deployed profile is actually dominated by
are out of this arm's scope entirely.

CANN's own `RecurrentKda` was evaluated as an arm and dropped. See
[DISCARDED-cann-recurrent-kda.md](DISCARDED-cann-recurrent-kda.md) for the
measurements and why. `fla` is kept where that one was not, for two reasons: it
is 2-6x faster than CANN's was on the same decode shapes, so it beats rather
than loses to the `triton` fallback it would replace, and it covers prefill and
decode rather than decode only.

Both vendor packages register op type `RecurrentKda`, so they could not have
coexisted here anyway — dropping the CANN arm is what leaves one live claimant.
The sibling `kda_chunk` bench has both and has to run one process per vendor;
see its README.

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

The sibling benchmark in `../kda_chunk` covers the chunked KDA prefill op, and
does keep a CANN arm: `ChunkKdaFwd` wins there by 3-6x, which is the opposite
result and the reason that one is still wired up.

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

`RTOL = 2e-3`, `ATOL = 1e-6` — except the `fla` arm, which carries
`FLA_RTOL = 1.6e-2`, `FLA_ATOL = 1e-5`.

Outputs here are small — `|out|` runs ~1e-4 with a ~1e-2 tail, because q and k
are L2-normalized before the scale. So `atol` has to stay well under that or it
passes everything on its own and the comparison means nothing; the real bar is
`rtol`.

The prepared prefill path is bit-exact against the generic kernel, so a correct
Ascend C recurrence should be too: tighten both to 0 once the kernel is expected
to match exactly. Both arms being our own code is what makes that reachable.

`fla` is the predicted exception, and measuring it confirmed the prediction: a
separately written kernel lands 1-2 ulps out on ~20 of 2M output elements at
~1e-4 magnitude, while its state pool matches inside `RTOL` everywhere. That is
a different summation order, not a different recurrence, so it gets bf16's own
`assert_close` default. Per-arm tolerances exist because measurement is gated on
validation: holding this arm to the strict bar would fail the row over bf16
rounding and discard every arm's timings with it.
