# chunk KDA benchmark

Compares four implementations of the chunked KDA prefill forward on the shapes
GLM-5.3 prefill runs.

| arm | what it is | status |
| --- | --- | --- |
| `cann` | `ChunkKdaFwd` from CANN 9.2.0's [ops-transformer](https://gitcode.com/cann/ops-transformer/tree/9.2.0/attention/chunk_kda_fwd) | built from source, see below |
| `triton` | `triton_kda_chunk.py`, written for this comparison | working |
| `fla` | `chunk_kda_fwd` from [flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu) (Tianjin University) | working; `chunk_size=64` only |
| `ascendc` | the kernel in the sglang GLM-5.3 plugin | not implemented; reports `-` |

`cann` and `fla` cannot run in the same process — see
[One process per vendor](#one-process-per-vendor), which is why a run prints two
tables.

## Run it

```bash
python bench_kda_chunk.py
```

No options. Every parameter is a module constant at the top of the script, so a
run is reproducible from the file alone and two runs are always comparable — to
measure something else, edit the constants. The script also pins
`ASCEND_RT_VISIBLE_DEVICES=4` itself, since operator work on this machine is
restricted to that card and forgetting the pin benchmarks on someone else's.

The target is the arm speedups are quoted against. It defaults to `ascendc`,
falling back to `triton` while that kernel does not exist. An unavailable arm is
a skipped column rather than a failure.

## One process per vendor

`cann` and `fla` both register the op type **`ChunkKdaFwd`**, and CANN's
launcher registry is keyed by that name. One slot, two claimants: whichever
resolves second loses, and it does not lose cleanly — `cann` **segfaults**,
while `fla` raises `ADD_TO_LAUNCHER_LIST_AICORE ChunkKdaFwd failed`. Path
ordering cannot fix it, in either direction.

So the arm is selected per process by `KDA_CHUNK_VENDOR`, and running the script
bare re-executes it once per vendor:

```bash
python bench_kda_chunk.py                    # both, one table each
KDA_CHUNK_VENDOR=fla python bench_kda_chunk.py   # just this one
```

The selection happens at import, before `torch_npu` loads, because two things
are already settled by the first aclnn call: the dlsym handler list is cached in
a function-local `static` for the life of the process, and `fla_npu`'s installed
`.pth` prepends its own vendor at *interpreter start*, before any user code —
so selecting `cann` means setting `FLA_NPU_DISABLE_PTH=1` as well as pointing
`ASCEND_CUSTOM_OPP_PATH` at the glm53 vendors explicitly.

`triton` and the target arm run under both vendors, which is what keeps the two
tables comparable: the vendor-specific column changes, the baseline does not.

A deselected arm is reported as `not selected: KDA_CHUNK_VENDOR=...` and is
never probed. That matters more than it reads — the probe runs an arm to find
out whether it exists, and probing the registration loser is the segfault.

The sibling benchmark in `../kda_recurrent` covers the recurrent KDA op the same
way, so the two directories share the same four files: `bench_*.py`,
`build_cann_op.py`, `cann_op.py`, `README.md`. The two `build_cann_op.py` are the
same script apart from `OP_NAME`, `VENDOR` and `NEEDS_WHEEL`; the two
`cann_op.py` share everything down to `available()` and then diverge, because
only one of the two operators ships a Python binding.

The four files here with no counterpart next door — `triton_kda_chunk.py`,
`reference.py`, `test_reference.py`, `test_triton.py` — exist because this
benchmark had to write its own comparison arms. The recurrent bench gets both of
its non-CANN arms from sglang, which ships a Triton fallback for that op and
nothing for this one.

The `fla` arm exists in both benches and needs no per-vendor split next door:
that one's CANN arm was dropped, so nothing there contests `RecurrentKda`. The
collision is specific to having two live claimants on one op type.

## The CANN arm needs a build first

The installed toolkit is **CANN 9.0.0**, which does not ship this operator:
`libopapi.so` has no `aclnnChunkKdaFwd` symbol and there is no kernel binary
for it. The operator landed later, so the only way to measure the official
implementation here is to compile the 9.2.0 sources against the 9.0.0 toolkit
and install them as a **custom op package**, which loads alongside the built-in
operators without modifying the installed toolkit.

```bash
python build_cann_op.py --check   # status, builds nothing
python build_cann_op.py           # build and install (tens of minutes)
```

This runs, from the checkout at `/data/wangkaize/ops-transformer` (branch
`9.2.0`):

```
bash build.sh --pkg --ops=chunk_kda_fwd --vendor_name=glm53chunk \
              --soc=ascend910b --disable-check-compatible -jN
```

then installs the resulting `.run` package under
`$ASCEND_OPP_PATH/vendors/glm53chunk_transformer/`.

`--opkernel` is deliberately **not** passed, despite the name suggesting it is
what builds the device kernel. It sets `ENABLE_OPKERNEL`, which `build.sh` tests
*before* the `--ops`/`--vendor_name` branch of its dispatch chain, so it wins and
runs `build_kernel` alone: kernel binaries get compiled, CPack is never invoked,
and the build reports success having produced no `.run` at all. Without it the
custom-op branch sets `ENABLE_OPS_KERNEL=ON` itself, so the kernel is still
built and is also packaged.

Why a cross-version build is expected to work: the operator declares
`ascend910b` support directly (`op_host/chunk_kda_fwd_def.cpp` calls
`AICore().AddConfig("ascend910b", ...)` alongside `ascend910_93` and
`ascend950`), and the repo's own compatibility gate is already forced `OFF` in
`cmake/config.cmake`. `--disable-check-compatible` is passed anyway so this does
not depend on that line staying as it is. If a future checkout starts depending
on a 9.2-only toolkit header, that surfaces here as a compile error naming the
header.

The build is serialized by a lock file in the ops-transformer checkout, because
that tree has a single shared `build/` directory and a concurrent build from
the same checkout — the neighbouring bench builds `recurrent_kda` from it — would
corrupt it.

The two benches install under *different* vendor names (`glm53chunk` here,
`glm53recurrent` next door), and that is load-bearing rather than cosmetic. The
installer clears same-named files out of the vendor directory it is installing
into, and every single-op build ships its own `libcust_opapi.so`, so a shared
vendor name would mean whichever bench built second silently deleted the other's
operator. Separate names let both sit in `opp/vendors/` at once; the aclnn
`dlsym` search walks every vendor directory, so each bench still finds its own
op. `cann_op.py` checks for `aclnnChunkKdaFwd` by name for that reason — the
sibling's package alone is enough to make a directory scan succeed.

## The fla arm needs no build

It comes from the installed `flash-linear-attention-npu` wheel plus its run
package, so there is no loader and no JIT bridge here — just an import. The
install is scenario 1 of that repo's `docs/开发者指南.md`:

```bash
cd /data/wangkaize/flash-linear-attention-npu
bash build.sh --soc=ascend910b --pkg --vendor_name=fla_npu \
              --ops=recurrent_kda,chunk_kda_fwd
cd torch_custom/fla_npu && python setup.py bdist_wheel
pip install --force-reinstall --no-cache-dir --no-deps dist/*.whl
./build_out/fla_npu_linux-aarch64.run --install
```

Both ops go in one `.run`: each single-op build ships its own
`libcust_opapi.so`, so building them separately under one vendor name would
make the second install delete the first.

Two things that are not obvious from that guide. The wheel is **not** optional
even though scenario 1 describes replacing an op in an already-installed wheel —
the `.run` carries only Ascend C artifacts, so a venv without `fla_npu` needs
both. And `scripts/check_npu_env.py` fails here on `torch_npu` 2.10.0 vs the
2.10.0.post2 it wants; that gate only applies under
`FLA_NPU_BUILD_LEGACY_EXTENSION=1`, and the default Stable-ABI build skips the
torch checks entirely, so it does not block this.

`chunk_size=128` is rejected by the operator's own parameter validation
(`chunkSize only supports 64.`) and documented as out of range, so the C128 rows
report `-` rather than `FAIL`. The arm is dropped per row for that, not per run.

### How the Python binding works

`chunk_kda_fwd` has no `torch_extension/` wrapper upstream, unlike its sibling
`recurrent_kda`, so there is no ready-made entry point to import and no wheel to
build (`NEEDS_WHEEL = False`). `cann_op.py` supplies one instead: a small C++
bridge calling `aclnnChunkKdaFwd` through ops-transformer's own `ACLNN_CMD`
macro, JIT-compiled by `torch.utils.cpp_extension.load` using `OpBuilder`'s
include and link paths.

That macro resolves the symbol with `dlsym` and searches the custom vendor
packages' `libcust_opapi.so` *before* the built-in `libopapi.so`, which is
precisely what lets a 9.2.0-built operator run on a 9.0.0 toolkit. It looks in
`ASCEND_CUSTOM_OPP_PATH` and `LD_LIBRARY_PATH`, and CANN's `set_env.sh` points
neither at the vendors directory, so `cann_op.py` sets both itself once it has
found the package that exports the symbol. It exposes *every* vendor package it
found, not just the one it wanted: the handler list is built on the first aclnn
call in the process and cached in a function-local `static`, so whatever is on
the path at that moment is all the process will ever search.

## Correctness

Not decided between the arms. Each is checked against `reference.py`, a direct
transcription of the operator's documented formulas, and an arm that disagrees
is reported `FAIL` rather than quietly becoming the baseline.

`reference.py` is in turn verified against a token-at-a-time recurrence — the
delta rule with no chunking anywhere — in `test_reference.py`. That is the real
check on the chunked algebra: the WY representation, the triangular inverse and
the chunk-local gate rebasing are where the subtlety lives, and a sequential
scan shares none of it.

```bash
python test_reference.py   # chunked algebra vs sequential scan, CPU only
python test_triton.py      # the Triton kernels vs the fp64 reference, on NPU
```

Both run standalone or under pytest. The reference runs on CPU because the NPU
has no fp64, and fp64 is the point of it.

## Notes on the Triton implementation

Five kernels, because that is what the data dependencies and the hardware
budget allow:

1. `kda_gates` — activate the gate, cumulative-sum it within each chunk, emit
   the gate-scaled `q` and `k`. Tiled over K as well as (chunk, head).
2. `kda_attn` — the chunk-local pairwise matrices `Aqk` and `L`.
3. `kda_akk` / `kda_akk_merge` — invert `I + L`.
4. `kda_scan` — the serial part: carry the state across a sequence's chunks.
5. `kda_output` — `qg @ h + Aqk @ v_new`, chunk-parallel.

Three things were forced by the target rather than chosen:

**Pairwise decay must be rebased.** Reaching a matmul needs the separable form
`(q 2^gk) (k 2^-gk)^T`, but the chunk-local cumulative gate falls by roughly
175 in log2 across a 64-token chunk under the deployed bounded gate, and
`2^175` overflows fp32 — the naive factoring yields `inf * 0 = NaN`. `kda_attn`
splits each chunk into `BC`-row sub-tiles and rebases each pair to the first row
of the column tile, so both factors stay bounded while their product is
unchanged.

**The triangular inverse is a doubling series, not row substitution.** `L` is
strictly lower triangular, so `(I + L)^-1 = sum (-L)^p` terminates exactly;
accumulating it by doubling needs `log2(BT)` steps against 63 serial ones, whose
unrolled live temporaries overflow UB by about 2.5x. At `chunk_size=128` even
that does not fit, so the inverse is hierarchical: `kda_akk` inverts `BB`-wide
diagonal blocks and `kda_akk_merge` joins adjacent pairs one level at a time via
`[[A,0],[C,D]]^-1 = [[A^-1,0],[-D^-1 C A^-1, D^-1]]`. Blocking below the 64 that
UB forces is also a work reduction, since doubling costs `2 log2(n)` matmuls of
`n^3` against two per merge — `BB=32` is about 22% of `BB=64`'s
multiply-accumulates at `BT=128`.

**Output is split from the scan.** Folding the output matmuls into the serial
walk costs an extra `[BT, BT]` operand tile and overflows cbuf at
`chunk_size=128`. The output is chunk-parallel anyway, so it does not belong in
the one kernel that must run serially.

## Results, and a measurement caveat

Both outside kernels beat the Triton arm by a wide margin on C64, and by
margins that widen with tokens: `cann` from 0.23x at T1k to 0.16x at T16k,
`fla` from 0.29x to 0.17x. The two are close, `cann` ahead by 7-14% on
matching rows. Triton is also the better arm at C128 than at C64 in absolute
terms, which is what makes `cann`'s C128 ratios (0.25-0.34x) look weaker than
its C64 ones without the kernel being slower.

One thing worth not re-discovering: an early run produced 14.2ms and 45.8ms for
`cann` at H8 T4096/T8192 C64 — against 5.45ms at the *larger* T16384, so
plainly not a property of the kernel. Re-measured on an idle card they are
1.50ms and 2.81ms, monotone in T. The outliers were contention from another
process on card 4. The pin at the top of the script stops you using someone
else's card; it does not stop someone else using yours, so a non-monotone
latency column is a reason to re-measure before it is a reason to investigate
the kernel.

## Adding the Ascend C arm

Implement `run_ascendc` in `bench_kda_chunk.py` to dispatch to the op and return
`(attn_out, final_state)` — `attn_out` as `[T, HV, V]` in the input dtype,
`final_state` as fp32 `[N, HV, K, V]`. Nothing else needs to change: the arm is
already wired into the probe, the validation and the table.
