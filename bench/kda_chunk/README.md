# chunk KDA benchmark

Measures our Ascend C chunked KDA prefill kernel on the shapes GLM-5.3 prefill
runs, against the two outside kernels that implement the same operator.

| arm | what it is | status |
| --- | --- | --- |
| `ascendc` | the kernel in the sglang GLM-5.3 plugin — the target | runs, C64 only |
| `cann` | CANN 9.2.0's `ChunkKdaFwd`, via `ref_ops.cann` | runs, C64 and C128 |
| `fla` | flash-linear-attention-npu's, via `ref_ops.fla_npu` | runs, C64 only |
| `torch` | the correctness anchor, `kda_python.kda_chunk_torch` | runs, NPU fp32, never timed |

The two reference arms come from the [`ref_ops`](../../../../ref_ops) package.
That package is the reason they can both be columns in one table: both upstreams
register op type **`ChunkKdaFwd`**, CANN's launcher registry is keyed by that
string, and the second to resolve used to lose — `cann` segfaulting, `fla`
raising `ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaFwd failed`, with path ordering
unable to fix it in either direction. `ref_ops` ships *renamed* copies of the
CANN-side ops, so the collision is dissolved rather than worked around and no
per-vendor process or re-exec is needed here. `ref_ops/AGENTS.md` is the install
and pin-bump procedure.

`import ref_ops` happens at module scope and must stay there: it builds the
aclnn dlsym search path, and `GetCustOpApiHandlers` caches that path in a
static on the *first* operator launch in the process. An import deferred into an
arm would run after the target had already launched, leaving both vendors
unreachable for the rest of the run.

## Where the target stands

A full run on an idle card 4 at the constants currently in the file (H8, C64,
T1k–4k), with every arm taking raw `BSND` inputs and fusing what it can. Every
row `PASS`es, target and references alike:

| shape | `ascendc` | `cann` | vs `cann` | `fla` | vs `fla` |
| --- | --- | --- | --- | --- | --- |
| H8 C64, T1024 | 2.03 ms | 0.95 ms | 0.47x | 1.99 ms | 0.98x |
| H8 C64, T2048 | 3.83 ms | 1.40 ms | 0.37x | 2.10 ms | 0.55x |
| H8 C64, T4096 | 7.36 ms | 2.59 ms | 0.35x | 2.34 ms | 0.32x |

So the target is ~2-3x slower than `cann` and the gap widens with `T`. Against
`fla` it is level at T1024 and falls behind from there — `fla` scales much
better across this range (1.99 → 2.34 ms for 4x the tokens, against the
target's 2.03 → 7.36 ms), so the target is close to flat-cost at T1k and
increasingly per-token-bound after.

Two notes before reading those as kernel-vs-kernel ratios:

* **`cann`'s column includes host work the other two fuse** — see the fusion
  table below. Its kernel alone is faster than its column says.
* The target's own `MAD` is 0.001-0.013 ms against medians of 2-7 ms, so these
  are stable measurements, not noise.

Earlier revisions of this file quoted H4 and T16k rows and a `cann`-as-stand-in
bar. Those came from different constants and, for the reference arms, from the
host-prepared `TND` call this file no longer makes; `git log` has them.

## Removed arms

`triton`, plus `reference.py`, `test_reference.py`, `test_triton.py`,
`triton_kda_chunk.py`, `cann_op.py` and `build_cann_op.py` — the last two being
the per-bench build machinery that `ref_ops` replaced. `git log --
bench/kda_chunk/` has all of it.

## Run it

```bash
python bench_kda_chunk.py
```

No options. Every parameter is a module constant at the top of the script, so a
run is reproducible from the file alone and two runs are always comparable — to
measure something else, edit the constants. The script also pins
`ASCEND_RT_VISIBLE_DEVICES=4` itself, since operator work on this machine is
restricted to that card and forgetting the pin benchmarks on someone else's.

With no target kernel it does not exit 1 any more: the first live arm stands in
as the denominator and the reference arms are measured anyway, labelled in the
table so what is quoted against what stays readable. Only an empty arm set is
fatal.

The same substitution happens per row when the target is live but does not cover
the shape, which is the C128 half of the table — `ascendc`'s chunk size is a
compile-time constant of the kernel, not an argument, so there is no C128 to
measure. Those rows keep reporting `cann` against itself at 1.00x rather than
going blank, and the target's cells read `-`. Measuring the C64 kernel in a C128
row would have read as a speedup.

## Inputs

Shapes, dtypes and distributions are `make_case` from
`../kda_recurrent/bench_kda_varlen_recurrent.py`, down to the scale factors and
the padding row at slot zero — `q`/`k`/`v`/`a` as `[1, T, H, D]` bf16, `b` as
`[1, T, H]`, and the state pool allocated `[slots,H,V,K]` and viewed
`[slots,H,K,V]`. That bench measures the recurrence this kernel replaces, so the
two have to agree on their inputs or comparing them says nothing. The only
addition here is `chunk_size`.

## What each arm is given, and what it fuses

Every arm is called with the inputs **raw and unmodified**: rank-4
`[1, T, H, D]` as `BSND`, un-normalized `q`/`k`, raw gate `a`, raw `beta`
logits, `A_log`, `dt_bias`, and the state pool in its own orientation. There is
no shared host preparation step any more.

| arm | L2 norms | beta sigmoid | bounded gate |
| --- | --- | --- | --- |
| `ascendc` | in kernel | in kernel | in kernel |
| `fla` | in kernel | in kernel | in kernel |
| `cann` | **host** | **host** | in kernel |

`cann` is the exception and it is a capability gap, not a calling mistake. That
operator has exactly seven attributes — `layout`, `scale`, `chunk_size`,
`safe_gate`, `lower_bound`, `use_gate_in_kernel`, `state_v_first` — and none is
a normalization, an epsilon or a sigmoid. `beta` is consumed as a plain
multiplicative coefficient on every kernel path, so passing raw logits would
silently compute the wrong thing rather than activate them. Nothing closes the
gap: `recurrent_kda` has both flags but is the decode kernel, and
`kda_input_proj` does fuse the beta sigmoid but is the projection stage and
subsumes the QKV GEMMs. So the `cann` column is that arm's honest end-to-end
cost on these inputs, with `host_l2norm` and a host sigmoid inside its timed
region — worth ~0.1 ms of 0.87 ms at T1024, shrinking with `T`.

This replaced a single `prepare_tnd` that normalized on the host for both
reference arms. That was wrong in two ways, both measured:

* **It charged `fla` for work it can fuse.** Fully fused, `fla` runs 2.12 ms
  against 2.77 ms host-prepared at H8 T1024 — the host form was inflating it by
  ~24%. The file justified the host path by claiming `fla`'s in-kernel
  normalization "refuses every chunk size but 64"; the constraint is real but
  C64 is the only size this benchmark runs and the only one `ascendc` supports.
* **It called both arms in `TND`.** Both accept rank-4 `BSND` with `cu_seqlens`
  at `B=1`, bit-identically (`0.0e+00` on both outputs, for both arms). `BSND`
  is also cheaper for `cann`: the aclnn layer normalizes every layout to
  head-major before launching and materializes one transpose per tensor it
  moves, so `TND` moves five (q, k, v, g, beta) where `BSND` moves only `beta`.
  Worth 9.2% at T1024, fading to 0.2% by T4096 — fixed overhead, not per-token.

Reaching `BSND` on `cann` needed a `ref_ops` change: that bridge is ours, not
upstream's, and it hardcoded `TORCH_CHECK(q.dim() == 3)`. It now accepts rank-3
`TND`/`NTD` and rank-4 `BSND`, sizing `attn_out`/`Aqk`/`Akk` by input rank,
which is what the aclnn layer checks against.

## State layout

All four arms read and write the state in the **pool's own orientation** — `K`
fastest, which is what `native_state` allocates, what the Ascend C op requires
and checks for with `stride(2)==1 && stride(3)==128`, and what the deployed
backend normalizes both of its pools to. The reference ops spell that
`state_v_first`: a contiguous `[N, HV, V, K]`. `native_live_state` gathers it,
and `k_first` transposes their result back into the `[N, HV, K, V]` arm
contract. `ascendc` and `run_torch` take the pool itself and need neither — they
write their state where the pool already keeps it and read it back as a view.

Before this, the reference arms were gathered into a contiguous `[N, HV, K, V]`,
so the two outside kernels were handed a transposed copy of the pool rather than
the pool. That is a correctness-of-comparison point rather than a speed one: the
repack measured 0.025 ms per call at H8 against medians of 1–7 ms, and removing
it moved no arm outside its `MAD`. It is worth having because `state_v_first` is
a tiling attribute of both reference kernels, not a host-side transpose, so an
arm fed a layout the deployment never produces could be measuring a different
kernel path for free. Both were checked under the flag and return results
*bit-identical* to the K-first spelling on these inputs, so it buys the shared
layout and changes nothing else.

`run_ascendc` does no host preparation at all: `kda_varlen_chunk` accepts the
caller's `[1, T, H, D]` tensors and the raw gate and beta, and does both L2
norms, both sigmoids and the bounded gate in its AIV prologue. The two
conversions it does need are layout, not math: `dt_bias` back to `[HV, K]`, and
a `clone_state` of the pool — the op updates state in place where the chunk
references return a new tensor, so without the copy each timed call would start
from the previous one's final state.

The gate stays raw: `A_log`, `dt_bias` and the bounded sigmoid are applied
in-kernel under `use_gate_in_kernel`, as the deployment runs it. The convention
was confirmed end to end by checking `cann.chunk_kda` against
`glm_kda_varlen_recurrent_npu` on the same inputs.

## The remaining hook

Every arm returns `(attn_out, final_state)` — `attn_out` as `[T, HV, V]` in the
input dtype, `final_state` as fp32 `[N, HV, K, V]`. Verified for all three.

`run_torch(inputs, case)` returns the same pair, and is never timed. It is
`kda_chunk_torch` from [`kda_python.py`](kda_python.py) — a literal
transcription of the chunked formulas, building the full `[BT, BT]` `A` column
by column and inverting it by forward substitution, `O(C^2 K)` temporaries and
all. Being obviously right is the only requirement, since nothing times it.

It runs **on the NPU, in fp32**, which is not a compromise on either count. fp32
is what the reference *is*: `kda_chunk_torch` calls `.float()` on `q`/`k`/`a`/`b`
and allocates its accumulator `torch.float32`, so handing it fp64 would downcast
at the first line. And on device it reads the same inputs as the arms, bit for
bit — the anchor is the only thing standing between a layout bug and a `PASS`,
so there is no reason to move it off the pool the kernel reads. `bench_one` still
pulls it to the host, because that is where the comparison happens.

`v` is the one input passed as fp32 rather than bf16, purely for the return
dtype: the function casts its output back to `v.dtype` on the way out, and the
bf16 round trip would cost up to ~3.9e-3 relative for nothing. The upcast is
exact and everything else is upcast internally anyway, so the arithmetic is
unchanged.

The import is deferred into the function, and `kda_python` must not import this
module at top level: the two would otherwise be circular. `kda_python.run()`
imports from here the same way, for its own chunk-vs-recurrent check.

The chunked algebra is where the subtlety lives — the WY representation, the
triangular inverse, the chunk-local gate rebasing — so the anchor is not
self-validating. `kda_python.run()` is the check that covers it: it runs
`kda_chunk_torch` against `kda_recur_torch`, a token-at-a-time scan in the same
file that shares none of that machinery, and asserts they agree at this
benchmark's tolerances. The reference arms are the second opinion on top of
that — an anchor that disagreed with both of them would be more likely wrong
than they are.

This matters more than usual here because the kernel was written *from* this
reference, so a misreading of the operator in `kda_chunk_torch` would be
reproduced in `ascendc` and the comparison between them would pass.

## Correctness

Each arm is checked against `run_torch` independently, in `bench_one`, not
against the target. An arm that disagrees is reported `FAIL` rather than quietly
becoming the baseline — which also means the framework's own
target-vs-reference comparison is deliberately bypassed with a comparator that
accepts anything. With two reference arms this matters more than it did with
none: they agree with each other to a bf16 ulp, so a shared misreading of the
inputs would look like consensus.

`RTOL = 2e-2`, `ATOL = 2e-4`, for bf16 inputs accumulated in fp32.

Measured against the anchor at H8 T1024 C64, all three arms clear those by a
wide margin, and by very different amounts:

| arm | `attn_out` max | `final_state` max |
| --- | --- | --- |
| `ascendc` | 3.1e-5 | 3.6e-8 |
| `cann` | 7.8e-5 | 4.9e-4 |
| `fla` | 8.3e-5 | 4.9e-4 |

`ascendc` sits four orders of magnitude closer on `final_state` than either
reference, and it is worth being clear about why that is not a quality claim:
the kernel was written from this same reference, so the two share their
summation order over the state update where the references' differs. That is
also why the reference columns are worth having — an error the anchor and the
kernel share is exactly the kind this comparison cannot see, and the kind they
can.

The two references agreeing with each other to 4.9e-4 on `final_state` while
differing from the anchor by the same amount is the expected shape of that: they
are two independent kernels with the same summation order, not two readings of
one.

Which bound binds was measured on these inputs at H4 T1024 C64: `|attn_out|`
runs ~5e-4 mean with a ~1.6e-2 max, so most of it sits under `atol` and `atol`
is what decides there. `final_state` is larger (mean ~5e-3) and is where `rtol`
does the work. The disagreement the tolerances have to admit is measured, not
guessed: `cann` and `fla` differ by ~1 bf16 ulp on identical inputs, and `cann`
against the deployed Triton recurrence differs by 6.1e-5 on `attn_out` and
4.4e-4 on `final_state`. The pair is not vacuous — zeroing or halving
`attn_out`, or zeroing `final_state`, each produced `FAIL` — but a regime that
only perturbed the small bulk of `attn_out` would need `atol` tightened to be
caught.

The validation path was exercised by standing `cann`'s own output in as the
anchor (both arms `PASS`) and then halving it (both `FAIL`). Doing that found a
latent bug, now fixed: `bench_one` moved each arm's output to the host but left
the anchor wherever it was produced, and `assert_close` treats device as an
attribute to match — so an NPU-resident anchor failed every comparison on the
device alone, including against a bit-identical tensor. That fix is what lets
the anchor run on the NPU now.

## Measurement caveat

A non-monotone latency column is a reason to re-measure before it is a reason to
investigate the kernel. An early run produced 14.2ms and 45.8ms at H8
T4096/T8192 C64 against 5.45ms at the *larger* T16384; re-measured on an idle
card they were 1.50ms and 2.81ms, monotone in T. The outliers were another
process on card 4. The pin at the top of the script stops you using someone
else's card; it does not stop someone else using yours.

The sibling benchmark in `../kda_recurrent` covers the recurrent KDA op and
keeps two arms, `triton` and `prepared` — both the caller's own paths.
