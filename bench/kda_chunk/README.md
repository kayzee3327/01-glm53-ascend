# chunk KDA benchmark

Measures our Ascend C chunked KDA prefill kernel on the shapes GLM-5.3 prefill
runs, against the two outside kernels that implement the same operator.

| arm | what it is | status |
| --- | --- | --- |
| `ascendc` | the kernel in the sglang GLM-5.3 plugin — the target | not implemented; `run_ascendc` raises |
| `cann` | CANN 9.2.0's `ChunkKdaFwd`, via `ref_ops.cann` | runs, C64 and C128 |
| `fla` | flash-linear-attention-npu's, via `ref_ops.fla_npu` | runs, C64 only |
| `torch` | the correctness anchor | blank by design; `run_torch` raises |

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

## Measured bar

From a full run on an idle card 4, with `ascendc` absent so `cann` stands in as
the denominator:

| shape | `cann` | `fla` |
| --- | --- | --- |
| H4 C64, T1k → T16k | 0.85 → 6.80 ms | 1.38 → 7.10 ms |
| H8 C64, T1k → T16k | 0.95 → 8.83 ms | 1.26 → 9.01 ms |
| H4 C128, T1k → T16k | 0.82 → 6.28 ms | — |
| H8 C128, T1k → T16k | 0.95 → 8.43 ms | — |

`cann` leads `fla` by 1.6x at T1k narrowing to 1.02-1.04x at T16k, so the two
converge where the shapes are large enough to stop being launch-bound. C128 buys
`cann` a few percent over C64. This is the bar the target is aiming at.

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

## Inputs

Shapes, dtypes and distributions are `make_case` from
`../kda_recurrent/bench_kda_varlen_recurrent.py`, down to the scale factors and
the padding row at slot zero — `q`/`k`/`v`/`a` as `[1, T, H, D]` bf16, `b` as
`[1, T, H]`, and the state pool allocated `[slots,H,V,K]` and viewed
`[slots,H,K,V]`. That bench measures the recurrence this kernel replaces, so the
two have to agree on their inputs or comparing them says nothing. The only
addition here is `chunk_size`.

`prepare_tnd` does the per-arm adaptation: squeeze to TND, and L2-normalize
`q`/`k` and sigmoid `beta` in fp32. The normalization is not a choice — the
`ref_ops` CANN bridge exposes no `use_qk_l2norm_in_kernel` flag, and `fla`'s
in-kernel path refuses every chunk size but 64, so the host form is the only one
both arms can be measured in. It lives inside the arms rather than in
`make_inputs` because the deployed path pays it per call, which also leaves the
target free to fuse it and be credited.

The gate stays raw: `A_log`, `dt_bias` and the bounded sigmoid are applied
in-kernel under `use_gate_in_kernel`, as the deployment runs it. The convention
was confirmed end to end by checking `cann.chunk_kda` against
`glm_kda_varlen_recurrent_npu` on the same inputs.

## The two hooks

`run_ascendc(inputs, case)` returns `(attn_out, final_state)` — `attn_out` as
`[T, HV, V]` in the input dtype, `final_state` as fp32 `[N, HV, K, V]`, which is
what both reference arms were verified to return. Nothing else needs to change:
the arm is already wired into the probe, the validation and the table.

`run_torch(inputs, case)` returns the same pair, and is never timed. It is left
blank deliberately — a hand-written anchor is the point of it. Two things about
it are decided by how the frame uses it:

* Being obviously right is the only requirement, so it can be a literal
  transcription of the formulas — `O(C^2 K)` temporaries and all — rather than
  anything a kernel could run.
* It should accumulate in fp64, which means running on CPU, because the NPU has
  no fp64. `torch.testing.assert_close` wants matching dtypes, so either round
  to the arm's dtype there or upcast at the comparison site in `bench_one`. The
  tolerances below were set with the former, where the bf16 rounding alone costs
  up to ~3.9e-3 relative.

Until it exists, every `ok` column reads `-` rather than `PASS`. Timing a kernel
before it can be checked is useful; calling it correct because nothing
contradicted it is not — and the two reference arms agreeing is not a substitute,
since they would agree on a shared misreading too.

Note that the chunked algebra is where the subtlety lives — the WY
representation, the triangular inverse, the chunk-local gate rebasing — so a
reference is not self-validating either. The deleted `test_reference.py` checked
it against a token-at-a-time delta-rule scan that shares none of that
machinery, which is the check worth repeating. The reference arms are a second
opinion available now that it was not before: an anchor that disagrees with both
of them is more likely wrong than they are.

## Correctness

Each arm is checked against `run_torch` independently, in `bench_one`, not
against the target. An arm that disagrees is reported `FAIL` rather than quietly
becoming the baseline — which also means the framework's own
target-vs-reference comparison is deliberately bypassed with a comparator that
accepts anything. With two reference arms this matters more than it did with
none: they agree with each other to a bf16 ulp, so a shared misreading of the
inputs would look like consensus.

`RTOL = 2e-2`, `ATOL = 2e-4`, for bf16 inputs accumulated in fp32.

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
device alone, including against a bit-identical tensor. `run_torch` runs on CPU
anyway, but the frame no longer depends on that.

## Measurement caveat

A non-monotone latency column is a reason to re-measure before it is a reason to
investigate the kernel. An early run produced 14.2ms and 45.8ms at H8
T4096/T8192 C64 against 5.45ms at the *larger* T16384; re-measured on an idle
card they were 1.50ms and 2.81ms, monotone in T. The outliers were another
process on card 4. The pin at the top of the script stops you using someone
else's card; it does not stop someone else using yours.

The sibling benchmark in `../kda_recurrent` covers the recurrent KDA op and
keeps two arms, `triton` and `prepared` — both the caller's own paths.
