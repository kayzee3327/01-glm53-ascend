# kda_varlen_chunk — design notes

Long-form notes for `kda_varlen_chunk_kernel.cpp` and `kda_varlen_chunk.h`.
The source carries the banner comments that map code to the reference; the
reasoning, the measurements and the traps live here. Headings are named by the
code that points at them.

The kernel is a MIX specialization of `kda_chunk_torch`
(`bench/kda_chunk/kda_python.py`) to `BT == 64`, `K == V == 128`, `H == HV`.
One block owns one (sequence, head) and walks its chunks in order; one AIC and
two AIV per block.

## Comment convention

`Process` is banner-commented against that reference: a `══ NAME ══` banner
opens a stage and lists the reference lines it covers, and a `── expr ──`
banner marks the instructions computing that one expression. Code between two
banners implements the upper one.

## Departures from the reference

Two, both forced by the hardware and both measured bit-close (max abs 2.3e-9
over the whole operator):

* **A and Aqk are factored so the Cube can compute them.** The reference's
  pairwise `(g_c - g_i).exp()` keeps every exponent <= 0; the factored form
  splits it into `exp(gc-m) * exp(m-gc)`, which overflows fp32 unless `m` is
  rebased often. The cumulative gate spans ~189 over 64 tokens, so the rebase
  is per 32 rows, anchored on the block's midpoint. See `kRebaseRows`.
* **The triangular inverse is blocked**, as FLA's is: the two 32x32 diagonal
  blocks by forward substitution on the AIVs, the off-diagonal block by two
  Cube matmuls, `C10 = -C11 B10 C00`. The reference solves all 64 rows in one
  serial chain.

## No on-chip UB->L1 path

Everything the Cube consumes travels through GM. `DataCopy` UB->L1 with ND2NZ
is a software simulation on this architecture that stages through the
high-level Matmul API's workspace and needs `REGISTER_MATMUL`, so a basic-API
kernel has no on-chip UB->L1 path. That is why the AIV's results reach the AIC
through this block's workspace slice.

## `CubeMatmul`

`GM(ND) -> L1(Nz) -> L0A(Zz)/L0B(Zn) -> L0C(Nz) -> GM(ND)`, the chain the
devkit's `load_data_l12l0` sample documents. `Load3Dv2` does both L1->L0
moves: with `l1H=1` and a 1x1 filter it degenerates to a reformat,
`enTranspose` transposes into L0A, and a destination of L0B transposes
unconditionally.

| parameter | meaning |
|---|---|
| `A_TRANS` | A is stored `[K, M]` and the operand is its transpose |
| `B_TRANS` | B is stored `[N, K]` and needs no transpose into L0B |
| `dstRowStride` | elements between consecutive rows of C in GM. Defaults to N (a packed `[M, N]` result); pass the real row pitch when C is a block of a wider tile |
| `first` | set on the block's first matmul, where there is no previous call whose buffers have to be waited on |
| `aRowStride`, `bRowStride` | elements between consecutive rows of A / B in GM. Default 0 means packed |

`aRowStride`/`bRowStride` work because the ND2NZ load takes the row pitch as
`srcDValue` independently of the column count (`dValue`) it moves, so a
sub-block of a wider tile goes to the Cube in place; the L1 layout is
unaffected. The blocked triangular inverse reads 32x32 blocks of 64x64 tiles
this way, which is what lets it skip a packing pass.

`B` stored `[N, K]` is already the Zn orientation, so it goes in with
`Load2D`; `B` stored `[K, N]` needs the transpose `Load3Dv2` applies for L0B.

For `Fixpipe`: `srcStride` is `mSize` rounded up to 16, in C0 units; under
NZ2ND `dstStride` is the destination row's element count.

### Reverse barriers between consecutive matmuls

L1, L0A/L0B and L0C are fixed addresses reused by every call, so each stage
has to wait for the previous call's consumer to release the buffer it is about
to overwrite:

| event | what it guards |
|---|---|
| `MTE1_MTE2` | L1 refill waits for the last L1 -> L0 load |
| `M_MTE1` | L0A/L0B load waits for the last `Mmad` |
| `FIX_M` | `Mmad` waits for the last `Fixpipe` to drain L0C |

A single matmul needs none of these and is correct without them; a chain of
them faults the core, which is how they were found.

## UB map

Laid out by hand at absolute addresses as in the recurrent kernels. ~187 KB of
the 192 KB an AIV gets in MIX mode, as the recurrent kernels' maps also
assume. Overrunning it faults the core rather than wrapping: `aivector error`
with no further detail.

Notes on individual buffers:

* **`BRAW`/`BETA`/`NRM`** — beta's `1 + exp(-b)` denominators and the two L2
  norms, a whole 64-lane repeat each so `Exp` never reads uninitialized UB.
* **`RED`** — `ReduceRows128_AS` folds a row's two halves into this and
  reduces it, so one 64-lane repeat is all it needs.
* **`LB`** — `lower_bound` as a tensor: `Div` takes both operands as tensors
  and `Divs` is unsupported, so the scalar has to be materialized.
* **`SCR`** — Scalar/Vector exchange for `exp(A_log)`: no scalar exp here.
* **`NORM`** — a second `[64, 128]` working tile, for the stages that need two
  live at once (the rebased factors, the auxiliary operands, the output sum,
  the state's row decay).
* **`STAGE`** — the bf16 staging tile, reused for q, k, a and v in turn. Half
  a chunk at a time: every use is a load immediately followed by the `Cast`
  that consumes it, so a half tile costs one extra copy and leaves the mask
  its own 16 KB.
* **`MASK`** — the lower-triangular mask, `[64, 64]` fp32, built once per
  block. Its own space, not aliased onto the staging tile: the staging tile is
  refilled mid-chunk, and an alias there made the mask depend on an ordering
  that is easy to break.

## `Slice`

A member rather than a lambda: inside a `this`-capturing lambda the member
tensor is const-qualified and `GlobalTensor::operator[]`, declared non-const,
will not bind.

## `LoadRows` / `StoreRows` / `StoreBlock`

`V_MTE2` in `LoadRows` is the write-after-read on `dst`: the caller's previous
Vector ops read the same tile and only `PIPE_V` barriers separate them, which
do not order MTE2. The matching `MTE2_V` stays with the caller, which issues
several loads before one wait.

`StoreBlock` moves a `[rows, cols]` sub-block of a wider UB tile into the
workspace, packed. `blockLen` and the gaps count 32 B datablocks, so
`srcRowStride - cols` has to be a multiple of 8 floats; every caller here
steps 32.

## `AddIdentity`

`X += I` on a `[BT, BT]` fp32 tile, on the Scalar unit. The diagonal has an
element stride of `BT + 1 == 65` and no vector form reaches it: `repeatStride`
counts 32 B datablocks, so a strided `Adds` can only step by multiples of 8
elements, and a per-row `Adds` of one element at offset `r*BT + r` is 32 B
aligned only when `r % 8 == 0` — an unaligned vector destination faults the
core with `aivector error`.

Both callers hand the tile straight to `StoreRows`, so the scalar writes need
`S_MTE3` as well as `S_V`.

## `StageCast`

Two half-chunk copies, because the staging tile is half a chunk: the mask and
a whole tile do not both fit in UB. Rows are strided by H in GM, so each half
is one `DataCopy` with a row gap of `(H-1)` rows; `blockLen` and gaps count
32 B datablocks, and a 128-lane bf16 row is 8 of them.

`V_MTE2` is the write-after-read on `stageLocal`: `MTE2_V` orders the `Cast`
after its own copy, but nothing orders the next copy after the `Cast` has
read, and `PipeBarrier<PIPE_V>` only orders V against V. Without it MTE2 runs
the whole queue ahead and every `Cast` reads the last copy's rows — measured
as all six `Cast`s in `LoadChunk` producing `a[32:64]`.

## `LoadChunk`

`b_beta` is one bf16 per row, strided by H, so a `DataCopy` would move one
useful element per datablock; Scalar gathers the chunk instead.

## `PrepareFactors`

`kda_pre_torch`, for the whole chunk at once. The reference's association is
kept: the reduction folds the two 64-lane halves before reducing, and the
reciprocal and square root run on Scalar, both of which the recurrent kernels
found to be bit-exactness requirements.

* `tl.sum` over 128 fp32 lanes folds the halves elementwise and only then
  reduces; a plain 2x64 `WholeReduceSum` pair does not match it.
* Scalar `sqrt`, not Vector: `Sqrt` and `Div` are 1 ulp on this hardware. The
  epsilon goes inside, as the reference places it.
* The per-row factor broadcast is a row at a time because the factor is per
  row and `Muls` takes a scalar.
* The gate's `lower_bound / (...)` is a real division to match the reference,
  with `lbRowLocal` holding the numerator broadcast over a row.
* `g_c = g.cumsum(-2)` is along the token axis, so each row adds the row above
  it. Serial in the row index; 63 adds of 128 lanes.

## `WriteRebased`

`A[r, c] = sum_k kn[r,k] kn[c,k] exp(gc[r,k] - gc[c,k])` written so the Cube
can compute it:

```
row[r,k] = kn[r,k] exp(gc[r,k] - m[k])      per rebase block
col[c,k] = kn[c,k] exp(m[k] - gc[c,k])      same m
A[r, c]  = sum_k row[r,k] col[c,k]
```

with `m = gc` of the block's midpoint row. The product reconstructs the
reference's exponent exactly; the rebase is what keeps the positive half
inside fp32, and the midpoint anchor is what bounds it at half the block's
span rather than all of it — see `kRebaseRows`. `kWsRowQ` is the same thing
with `qn`, for `Aqk`.

The row factors are computed one row at a time: a single strided `Sub` with
`src1RepStride 0` would express the broadcast in two instructions instead of
SB, but that form faults the vector core here. The two row factors share
`exp(gc - m)`, so it is multiplied into `kn` and `qn` in turn with the product
landing in `norm`, the second working tile.

The column factors are written as a whole `[r1, D]` tile because the matmul
reads them as the B operand in `[N, K]` orientation.

## `MaskAndNegate`

The Cube wrote both `A` and `Aqk` as full `[BT, BT]` tiles; everything on or
above the diagonal is noise from the padded columns and is zeroed here. Row r
keeps r columns of `Araw` and r+1 of `Aqk`, so the two differ by one lane.

`Araw` is masked first and stored to `kWsDpow`; `Aqk` is masked second and
leaves *it* resident in `tmp`, which is why `SolveDiagBlocks` opens with a
reload (below). Only the substitution path stores `Araw`, and it stores
nothing else: the identity enters inside each diagonal solve, and both halves
of `kWsX` are written in full before the Cube reads it, so there is no factor
or `X` to seed here.

`tril(-1)`: row r keeps columns `[0, r)`. Offsetting the destination by one
row against an unshifted mask gives each row r the mask of row r-1, which is 1
exactly on `[0, r)`. Row 0 keeps nothing.

Under `KDA_CHUNK_INVERSE_SQUARING`, X seeds as the identity, not as
`I + Araw`: the chain's six products are `prod_{s=0..5} (I + Araw^(2^s))` and
the `s == 0` factor is `I + Araw` itself, so seeding X with it applies that
factor twice — measured as exactly 2x on the linear term.

## The triangular inverse

Two implementations, selected by `KDA_CHUNK_INVERSE_SQUARING` in the header
(CMake: `-DKDA_CHUNK_INVERSE=substitution|squaring`). Both build, both are
validated against the reference, and the blocked substitution is the default.

Reference, in both cases:

```python
for i in range(1, BT):
    A[..., i, :i] += (A[..., i, :, None] * A[..., :, :i]).sum(-2)
A = A + eye(BT)
```

### Blocked forward substitution (default)

FLA's shape, from `chunk_intra.py`:

* `C00`, `C11` — the two 32x32 diagonal blocks, by forward substitution on the
  AIVs, one block per core.
* `C10 = -C11 B10 C00` — two 32x32 Cube matmuls, which is
  `chunk_kda_fwd_kernel_inter_solve_fused_npu`'s off-diagonal merge for the
  one pair BT=64 with 32-row blocks has.

`kWsX` is laid out as the Cube will finish it:

```
[ C00   0  ]
[ C10  C11 ]
```

The two AIVs own 32 whole rows of the tile each — rows, not blocks — so each
side's store is one contiguous run and neither needs a strided destination.
The Cube reads `C00` and `C11` in place at a row pitch of `BT` and writes
`C10` into the hole at (32, 0), so the assembled inverse needs no second AIV
pass. The strictly-upper zero is load-bearing: the auxiliary matmuls take the
whole `[BT, BT]` tile, not its blocks, and (0, 32) otherwise holds `Araw`'s
masked-out noise.

`B10` arrives already negated — `Araw` carries the sign — so the Cube's
`+C11 B10 C00` is the reference's `-C11 Akk10 C00` without a sign flip. `Mmad`
has no sign control and `cmatrixInitVal` only chooses overwrite against
accumulate.

**The substitution loop** is `chunk_kda_fwd_kernel_diag_solve_npu`:

```python
b_Ai = -tril(b_Akk, -1)
for i in range(2, BC):
    b_a = -Akk[i, :]  masked to columns [0, i)
    b_a += sum_j b_a[j] * b_Ai[j, :]
    b_Ai[i, :] = b_a
b_Ai += I
```

`Araw` already carries the negation and the strict-lower mask, so the row the
loop reads is the reference's `b_a` as-is, and the identity goes in once at
the end as `b_Ai += m_I` does. Row 0 is zero and row 1 is already its own
answer, so the loop starts at 2 as `range(2, BC)` does.

`tl.sum(b_a[:, None] * b_Ai, 0)` is a column reduction over the rows above i,
each weighted by that row's coefficient. No vector op reduces across rows, so
it is `i - 1` `Axpy` calls into the row being solved — which is what that
reduction is. Row j is nonzero only on columns `[0, j)`, so each covers the
leading j lanes.

The coefficients are read for the whole row in **one** Scalar visit rather
than one per `Axpy`. That is sound because an `Axpy` at step j writes lanes
`[0, j)` while the coefficient read at step j is lane j: no `Axpy` on this row
ever touches a lane this row has yet to read, so the in-place reads see the
original `Araw` entries the reference's `b_a` holds.

There is no `PipeBarrier<PIPE_V>` between the accumulations. They are a RAW
chain on one row, which is what the Bisheng compiler's `--cce-auto-sync`
inserts PIPE_V sync for: it is on by default for a `<<<>>>` launch and covers
PIPE_V interfaces operating on basic `LocalTensor`s, which these are.
Spelling them out measured the same (122.8 against 122.6 us/chunk) and the
results were identical, so the cost there is the instruction count, not the
sync.

**The helper AIV** has no UB copy of the chunk, so it reads `B11` back out of
the workspace. It lands it at (0, 32) of its own `tmp` tile, which is where
that block sits in the `[BT, BT]` layout — so its solve, identity and store
are the worker's code at the same offsets, with the zero to its left being the
inverse's strictly-upper block. Its row indices are spelled out rather than
reusing `SubstituteBlock`'s `(r0, r0)` anchor, which does not apply when the
block is at (0, 32) instead of (32, 32).

Both AIVs call `InitSocState()`, not just the worker: it sets the vector mask
to all-ones, and the helper runs vector work of its own. Nothing it
initializes is per-core state the two would contend over.

**Two sync points this path needs that the squaring path does not**, both
found by bisecting a 19.4% output mismatch confined to rows 32-63:

`SolveDiagBlocks` reloads `Araw` from `kWsDpow` before it starts. The seeding
block stores `Araw` there and then loads `Aqk` into the same `tmpLocal` for
the mask step, so by the time the solve runs the resident tile is `Aqk`, not
`Araw` — the mask ordering alone does not keep the buffer.

The handoff to the Cube is **one** `CrossCoreSetFlag<0x2>` per AIV on
`kFlagAivToAic`, unconditional on both cores, and **one** `CrossCoreWaitFlag`
on the AIC. Mode 2 is a quorum — the AIC's wait clears only once both AIVs
have set that ID — so the pair of sets is the mechanism that orders `C00` and
`C11` against the `C10` matmuls, not a double delivery to be split apart.
Giving the helper its own flag ID, or guarding either set with `if (worker)`,
starves the quorum and the kernel times out in the AIC. Mode 4 is the
single-AIV variant and A3 does not support it.

The `C10` pair also needs an explicit `FIX_MTE2` barrier between the two
matmuls. `kWsInvTmp` is the first one's Fixpipe output and the second's MTE2
input, and `CubeMatmul`'s own barriers order on-chip buffers only — nothing in
it covers a GM round trip. The squaring path never hits this because its
paired matmuls read `kWsDpow`, which the first does not write, which is why it
validated while this path did not.

### Repeated squaring

`Araw` is strictly lower triangular so `Araw^BT == 0`, and every `k < BT` has
a unique binary expansion, hence

```
(I - Araw)^-1 = sum_k Araw^k = prod_s (I + Araw^(2^s))
```

Six steps for BT=64, eleven Cube matmuls. The AIV adds the identity into each
factor before the product, so what the Cube multiplies is `(I + Araw^(2^s))`,
and between steps it turns the Cube's two results back into the next step's
operands: `X <- X @ (I + D)` is already in the scratch tile, and `D <- D^2`
needs the identity added before the next product. The last step's factor is
never used, so the squaring is skipped there.

### Why the substitution wins

Measured H8 T4096 C64 on card 4, through `bench/kda_chunk`'s frame:

| variant | us/chunk |
|---|---|
| repeated squaring | 110.8 |
| substitution, one AIV | 122.6 |
| substitution, split across the AIV pair | **104.3** |

Three things that were measured rather than assumed:

1. **The rendezvous count is not the lever.** Substitution cuts the inversion
   from six AIV<->AIC pairs to one. Priced directly by adding four empty
   mode-2 pairs per chunk: 1.6 us/chunk, so ~0.4 us a pair, and the five
   saved are worth ~2 of the 6.5. What buys the rest is the Cube work
   dropping from eleven 64x64 matmuls to two 32x32 ones.
2. **One AIV doing both diagonal blocks is a net loss** (122.6 against
   110.8). The 31-`Axpy` serial chain costs 35 us/chunk on one core — measured
   by stubbing the chain out, which gave 87.5. Splitting the two independent
   blocks one per AIV is what makes the change win at all.
3. **The explicit PIPE_V barriers in the chain are free** (see above).

### Accuracy

The blocked formula is exact. Verified in fp64 on the host: the two diagonal
substitutions plus `C10 = -C11 B10 C00` agree with the reference's serial
64-row loop to 4e-13, which is the roundoff the serial form itself carries
against `torch.linalg.inv`. In fp32 the blocked form is often *more* accurate
than serial.

On device, against an fp64 host reference over the test's cases: substitution
runs 7.3e-7 mean / 4.7e-5 max, squaring 6.9e-7 / 2.7e-5, on a signal whose
mean magnitude is ~5e-4. `/data/wangkaize/tmp/probe_inv_accuracy.py`
re-measures (the reference runs on CPU — the NPU has no fp64 kernels,
`aclnnEye` refuses `DT_DOUBLE`).

`tests/ops/test_kda_chunk.py`'s `atol` is set from that fp64 measurement, not
from the anchor: the anchor is itself an fp32 computation with a third
summation order, so a tolerance tuned to it tracks the anchor's rounding as
much as the kernel's. At the old `atol=2e-5` it admitted squaring and refused
substitution on 5 of 32768 elements with both equally close to the fp64 truth.

## `BuildAuxOperands`

`A` is the inverse scaled by beta on the right. Rather than scaling the
`[BT, BT]` inverse, beta is folded into the right operands, which is the same
product: `(X * beta[None, :]) @ M == X @ (beta[:, None] * M)`.

`v` is re-staged here rather than held across the chunk: keeping it resident
would need another `[64, 128]` tile and the UB map does not have one.

For the state update's left operand, its transpose is what the matmul takes,
which `A_TRANS` handles.

## `ApplyStateUpdate`

The decay is indexed by the key axis, which in the transposed tile is the
column, so each `[V, K]` row takes the whole `exp(g_c[refRow])` vector as an
elementwise multiplier. One row at a time: the tile is 64 KB and no UB buffer
here is larger than 32 KB.

The two `[D]` landing buffers are rows of `tmp`, not `qn`/`kn`: the
tracked-prefix path calls this and then `RebaseKDecay`, which still needs
`kn`.

## State layout

`S` and its increment are both stored transposed: `[V, K]` row-major, not
`[K, V]`. The state pool is K-contiguous (`[slots,H,K,V]` with
`stride(2)==1`), so a verbatim copy of a slot is already `[V][K]` and the
alternative is a 128x128 transpose on the AIV. The three matmuls that touch S
take it as a `B_TRANS` operand instead, and the increment is computed as
`vnew^T @ kdecay`:

```
(kdecay.T @ vnew).T == vnew.T @ kdecay
```

`CopyStateOut` is therefore a verbatim copy, and this block owns the whole
head so no other block shares a datablock.

## `CopyWs`

Workspace-to-workspace copy, through UB because there is no GM-to-GM move.
`tmp` is the staging buffer, so the caller's next Vector write to it has to
wait for the last store to have read it.

## Block indexing

On an AIV, `GetBlockIdx()` is a global AIV index (0 .. 2*blockDim-1), not the
Cube block: the two AIVs of block b report 2b and 2b+1. The block is therefore
the index halved. `blockDim` counts Cube blocks in `KERNEL_TYPE_MIX_AIC_1_2`
and each gets two AIVs, one block per (request, head).

The AIV pair splits the work unevenly: subBlockIdx 0 (the "worker") does all
the elementwise stages, because they are too tangled by the gate rebasing to
split and the second AIV would need its own UB copy of the whole chunk. The
one exception is the inverse's two diagonal solves, which share no data.

## Tracked prefix state

```
S_r = S * g_r.exp() + ((g_r - g_i[:r]).exp() * k_i[:r]).T @ v_i[:r]
```

Same shape as the chunk update with r rows instead of BT, run only by the
chunk containing `track_len` (`i*BT < track_len <= (i+1)*BT`); the AIV decides
and signals. The AIV rebuilds `kWsKDecay` against row r-1 instead of the last
row and the Cube redoes the increment over r rows.

The chunk's own update still has to run from the state as it was at the
chunk's start, so the prefix version is computed into a copy and the original
put back from `kWsStateSave`. The prefix increment overwrites `kWsStateUpd`,
so the Cube redoes the chunk's own over all BT rows afterwards.

## Entry points

Two kernels differing only in `TRACK_STATE`, a template parameter rather than
a runtime flag because it changes what the chunk loop does at a chunk
boundary. The macro exists because a `__global__` kernel cannot be a template
and the signature has to match the launcher exactly.

## Workspace must be zeroed

`at::zeros`, not `at::empty`. The A and Aqk tiles are written by one matmul
per rebase block covering only columns `[0, r1)`, so the strictly-upper part
of each tile is never written; the mask multiply that follows then turns
whatever was there into `0 * x`, which is NaN for uninitialized garbage that
happens to decode as NaN or Inf.
