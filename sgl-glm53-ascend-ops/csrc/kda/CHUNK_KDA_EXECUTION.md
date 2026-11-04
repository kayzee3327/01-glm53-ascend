Written for: an engineer reading `kda_varlen_chunk_kernel.cpp` for the first
time, who knows the chunked KDA formulas but not this kernel.

# Chunk KDA on Ascend: execution trace

Maps the formulas to `csrc/kda/kda_varlen_chunk_kernel.cpp`. One MIX block per
`(sequence, head)`: one AIC (Cube) and two AIV (Vector), of which only
`GetSubBlockIdx() == 0` does vector work. Chunks run in order — the state
carries across them.

`BT = C = 64` tokens per chunk, `K = V = 128`, `H == HV`. Activations bf16,
everything else fp32.

Notation follows the formula sheet: $\Gamma_{[t]}^{1\rightarrow C}$ is the
per-token cumulative gate, $\gamma_{[t]}^r$ its row $r$.

## Where the two cores meet

Every Cube operand travels through a per-block GM workspace slice. Basic
Ascend C has **no on-chip UB→L1 path**: the `DataCopy` UB→L1 with ND2NZ that
appears to be one is a software simulation that stages through the high-level
Matmul API's workspace and requires `REGISTER_MATMUL`. So the split is not a
tuning choice — Vector writes a tile to GM, hands over, Cube reads it back.

Nine rendezvous per chunk, each a strict pair:
`CrossCoreSetFlag<0x2, PIPE_MTE3>` / `CrossCoreWaitFlag` on the AIV,
`CrossCoreWaitFlag` / `CrossCoreSetFlag<0x2, PIPE_FIX>` on the AIC.

## Prologue, once per block

| Step | Computation | Core | Ascend C |
|---|---|---|---|
| P0 | $\mathbf S_{[0]} \leftarrow$ pool slot, or $\mathbf 0$ | Vector | 128 row `DataCopy`s GM→UB→workspace. The pool is `[slots,H,K,V]` with `stride(2)==1`, so a verbatim copy lands as `[V,K]`, i.e. $\mathbf S^\top$. Every matmul below is written against that transposed convention |
| P1 | `track_len == 0` → checkpoint the initial state | Vector | `CopyStateOut` |

## Per chunk

### 1 — load

| Computation | Core | Ascend C |
|---|---|---|
| $\mathbf Q, \mathbf K, \mathbf V, \mathbf a$ bf16 → fp32; raw $\beta$ | Vector | `StageCast`: row-strided `DataCopy` (rows are `H`-strided in GM) + `Cast`, in two half-chunks through one 8 KB staging tile. $\beta$ is one element per row, so Scalar `GetValue` beats a `DataCopy` that would move one useful element per datablock |

### 2 — normalization, gate, Decay Cumsum

$$
\mathbf Q_{[t]} = \frac{\mathbf q}{\lVert \mathbf q\rVert}\cdot\text{scale},\quad
\mathbf K_{[t]} = \frac{\mathbf k}{\lVert \mathbf k\rVert},\quad
\beta_{[t]} = \sigma(b)
$$

$$
\log\alpha_{[t]}^r = \text{lower\_bound}\cdot\sigma\!\left(e^{A_{\log}}(\mathbf a^r + \mathbf{dt}_{\text{bias}})\right),
\qquad
\gamma_{[t]}^r = \sum_{i=1}^{r}\log\alpha_{[t]}^i
$$

| Core | Ascend C |
|---|---|
| Vector | `Mul` + `ReduceRows128_AS` for the two norms — it folds the 128 lanes as two 64-halves before reducing, which is what `tl.sum` does; a plain 2×64 `WholeReduceSum` pair does not match. One Scalar visit per chunk computes both `sqrt`s and $\beta$'s reciprocal, because Vector `Div`/`Sqrt` are 1 ulp here. The gate is `Add`/`Muls`/`Exp`/`Adds`/`Div`. **Decay Cumsum is 63 serial row `Add`s** — the prefix sum your notes call for, along the token axis |

### 3 — the rebased factors

The Cube cannot evaluate $A_{c,i}=\sum_d k_{c,d}k_{i,d}e^{g_{c,d}-g_{i,d}}$
column by column. It needs an outer product, which forces the split exponent
your notes reject:

$$
e^{g_{c,d}-g_{i,d}} = e^{g_{c,d}-m_d}\cdot e^{m_d-g_{i,d}}
$$

Taking $m = \gamma_{[t]}^{C}$ for the whole chunk is exactly the overflow your
notes predict. **The fix is to rebase per 16-row block**, with
$m = \gamma_{[t]}^{r_1-1}$ for rows $[r_0, r_1)$:

$$
\mathrm{row}^{(R)} = \mathbf K_{[t]}^{[r_0:r_1]}\odot e^{\gamma^{[r_0:r_1]}-m},
\qquad
\mathrm{col}^{(R)} = \mathbf K_{[t]}^{[0:r_1]}\odot e^{m-\gamma^{[0:r_1]}}
$$

Measured on deployed inputs, $\gamma$ spans $\approx 177$ over 64 tokens, and
the largest exponent either factor sees is:

| rebase granularity | max exponent | result |
|---|---|---|
| whole chunk | — | **NaN** |
| 32 rows | +88.4 | at the fp32 cliff (+88.7) |
| **16 rows** | **+43** | err $\approx 10^{-8}$ vs the pairwise form |

So 16 is a correctness bound, not a knob. The raw products legitimately reach
$5\times10^{16}$ above the diagonal and are removed by the mask in step 5;
inside the kept triangle $\lvert A\rvert \le 0.023$.

| Core | Ascend C |
|---|---|
| Vector | Per-row `Sub`/`Exp`/`Mul`, then `StoreRows` to the workspace. A single strided `Sub` with `src1RepStride = 0` would express the broadcast in two instructions instead of 16, but that form faults the vector core |

### 4 — $A_{kk}$ and $A_{qk}$

$$
A_{kk} = \mathrm{row}_K\,\mathrm{col}_K^\top,
\qquad
A_{qk} = \mathrm{row}_Q\,\mathrm{col}_K^\top
$$

| Core | Ascend C |
|---|---|
| **Cube** | 8 × `CubeMatmul<false, true>`, `[16,128] @ [r1,128]ᵀ`, `dstRowStride = BT` so each result lands as a `[16, r1]` block of a `[64,64]` tile. `B_TRANS` because `col` is stored `[N, K]`, already the Zn orientation |

### 5 — strict-triangular construction

$$
\mathbf A^{\text{raw}} = -\operatorname{StrictTril}\left(\operatorname{Diag}(\beta_{[t]})\,A_{kk}\right),
\qquad
A_{qk} \leftarrow \operatorname{Tril}(A_{qk})
$$

| Core | Ascend C |
|---|---|
| Vector | `Muls` by $-\beta_r$ per row. The `tril` is a whole-tile `Mul` against a 0/1 mask: multiplying rows $1..BT\!-\!1$ against the *unshifted* mask gives row $r$ the mask of row $r-1$, which is exactly `tril(-1)`, and row 0 is zeroed outright. The obvious per-row `Duplicate(x[r*BT + r], 0, BT-r)` is only 32 B aligned when `r % 8 == 0` and faults the core |

### 6 — Lower Triangular Inverse

$$
\mathbf M_{[t]} = \left(\mathbf I + \mathbf A^{\text{raw}}\right)^{-1}\operatorname{Diag}(\beta_{[t]})
$$

Your notes solve this by forward substitution, row by row:
$X_{ij} = \delta_{ij} - \sum_{k<i} A_{ik}X_{kj}$. That is 64 dependent steps,
all on the Vector unit.

This kernel uses the fact that $\mathbf A^{\text{raw}}$ is **strictly** lower
triangular, hence nilpotent with $(\mathbf A^{\text{raw}})^{C} = 0$. Every
$k < C$ has a unique binary expansion and the powers commute, so the Neumann
series telescopes into a product:

$$
\left(\mathbf I - \mathbf N\right)^{-1} = \sum_{k=0}^{C-1}\mathbf N^k
= \prod_{s=0}^{\log_2 C - 1}\left(\mathbf I + \mathbf N^{2^s}\right),
\qquad \mathbf N = -\mathbf A^{\text{raw}}
$$

Six steps for $C=64$: 11 matmuls, **no serial vector chain**. Accuracy on real
KDA matrices is equal to or better than blocked substitution across input
scales from 0.2 to 10.

| Core | Ascend C |
|---|---|
| **Cube** | 11 × `CubeMatmul` `[64,64] @ [64,64]`: `X ← X · Factor` each step, plus `Dpow ← Dpow²` for all but the last |
| Vector | Between steps, `Factor ← I + Dpow`. `Dpow` and `Factor` are **separate buffers** — squaring the factor instead would give $\mathbf I + 2\mathbf N^{2^s} + \mathbf N^{2^{s+1}}$. `X` seeds as $\mathbf I$, not $\mathbf I + \mathbf N$: the $s=0$ factor already *is* $\mathbf I+\mathbf N$, and seeding with it applies that factor twice (measured as exactly 2× on the linear term). `AddIdentity` runs on Scalar — a stride-65 diagonal is unreachable by any vector repeat, since repeat strides count 32 B datablocks |

$\operatorname{Diag}(\beta)$ is **not** applied to $\mathbf M$. Since
$\mathbf M = X\operatorname{Diag}(\beta)$ and
$X\operatorname{Diag}(\beta)\,\mathbf Z = X(\operatorname{Diag}(\beta)\mathbf Z)$,
$\beta$ is folded into the right operands in step 7 — one `Muls` over a
`[64,128]` tile instead of over `[64,64]` plus a second pass.

### 7 — right operands

$$
\operatorname{Diag}(\beta)\left(\Gamma^{1\rightarrow C}\odot\mathbf K\right),\quad
\operatorname{Diag}(\beta)\mathbf V,\quad
\Gamma^{1\rightarrow C}\odot\mathbf Q,\quad
\left(\Gamma^{(i+1)\rightarrow C}\odot\mathbf K\right)
$$

| Core | Ascend C |
|---|---|
| Vector | `Exp`/`Mul`/`Muls`. The last one is $e^{\gamma^{C}-\gamma^{r}}\odot\mathbf K$ — the $(i{+}1)\rightarrow C$ form from your notes, excluding token $i$'s own gate, matching fla. $\mathbf V$ is re-staged from GM here rather than held across the chunk: keeping it resident would need another `[64,128]` tile and the UB map has none (186.5 KB of the 192 KB an AIV gets in MIX mode) |

### 8 — Auxiliary Matrix

$$
\mathbf W_{[t]} = \mathbf M_{[t]}\left(\Gamma^{1\rightarrow C}\odot\mathbf K_{[t]}\right),
\qquad
\mathbf U_{[t]} = \mathbf M_{[t]}\mathbf V_{[t]},
\qquad
\mathbf W_{[t]}\mathbf S_{[t]}
$$

| Core | Ascend C |
|---|---|
| **Cube** | 3 × `CubeMatmul`. The $\mathbf W\mathbf S$ one uses `B_TRANS`, because the state tile holds $\mathbf S^\top$ |

### 9 — Pseudo-value term

$$
\tilde{\mathbf V}_{[t]} = \mathbf U_{[t]} - \mathbf W_{[t]}\mathbf S_{[t]}
$$

| Core | Ascend C |
|---|---|
| Vector | One `Sub` over `[64,128]` |

### 10 — Chunk Output

$$
\mathbf O_{[t]} = \left(\Gamma^{1\rightarrow C}\odot\mathbf Q_{[t]}\right)\mathbf S_{[t]}
+ \operatorname{Tril}(A_{qk})\,\tilde{\mathbf V}_{[t]}
$$

| Core | Ascend C |
|---|---|
| **Cube** | 2 × `CubeMatmul` for the two terms, plus the state increment $\tilde{\mathbf V}^\top(\Gamma^{(i+1)\rightarrow C}\odot\mathbf K)$ — computed with its operands swapped, so the result is already transposed to match $\mathbf S^\top$ |
| Vector | `Add` the two terms, `Cast` to bf16, store out in two halves with the row stride `H` |

### 11 — Chunk State Update

$$
\mathbf S_{[t+1]} = \operatorname{Diag}\!\left(e^{\gamma_{[t]}^{C}}\right)\mathbf S_{[t]}
+ \left(\Gamma^{(i+1)\rightarrow C}\odot\mathbf K_{[t]}\right)^{\!\top}\tilde{\mathbf V}_{[t]}
$$

| Core | Ascend C |
|---|---|
| Vector | 128 rows of `Mul` + `Add`. Because the tile is $\mathbf S^\top$, the decay is a **vector** `Mul` against the whole $e^{\gamma^C}$ row, not a per-row scalar `Muls` — that transpose detail was a real bug, caught by measuring `kWsOut` against `qg @ S0.T` (7e-10) versus `qg @ S0` (3.5e-3) |

### 11t — tracked prefix state, when `track_len` lands in this chunk

$$
\mathbf S_{\text{track}} = \operatorname{Diag}\!\left(e^{\gamma^{r-1}}\right)\mathbf S_{[t]}
+ \left(e^{\gamma^{r-1}-\gamma^{i}}\odot\mathbf K^{[0:r]}\right)^{\!\top}\tilde{\mathbf V}^{[0:r]}
$$

| Core | Ascend C |
|---|---|
| both | Save $\mathbf S^\top$, rebase `kDecay` on row $r-1$, one Cube `[128,128]` over $r$ rows, write the checkpoint, restore the saved state, **then recompute the chunk's own increment** — `kWsStateUpd` is a single buffer and the prefix pass overwrites it |

## Epilogue

`pool[slot, head] ← S^T` when `slot > 0`.

## Summary of the split

| | work |
|---|---|
| **Cube** | 25 matmuls per chunk: 8 for $A_{kk}$/$A_{qk}$, 11 for the inverse, 3 for $\mathbf W$/$\mathbf U$/$\mathbf W\mathbf S$, 3 for the output and state increment |
| **Vector** | normalization, gate and Decay Cumsum, the 16-row rebasing, triangular masking, identity insertion, all accumulation, the bf16 output cast, and the state decay |

The inverse is the one structural departure from the reference, and it is what
moves the chunk's serial dependency onto the Cube instead of leaving 64
dependent row-steps on the Vector unit.

## Accuracy

12/12 chunk tests pass, 105/105 for the suite. Worst case over six independent
input draws is 0.85 of the test tolerance, and that residual is **bf16 output
quantization, not kernel error**: rounding the fp32 reference to bf16
reproduces the gap to three digits, and the kernel adds only 0.01–0.09 of the
bound beyond what the dtype costs.
