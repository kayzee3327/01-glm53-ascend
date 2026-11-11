# chunk KDA 的核间分工与流水

补全并校正一份按 `RunAiv` / `RunAic` 展开的串行执行梳理。配套读
[CHUNK_KDA_EXECUTION.md](./CHUNK_KDA_EXECUTION.md)（公式到代码的映射）和
[kda_varlen_chunk.md](./kda_varlen_chunk.md)（设计理由、踩坑、实测）。

代码口径：`kda_varlen_chunk_kernel.cpp`，`BT = 64`、`K = V = 128`、`H == HV`，
`KERNEL_TYPE_MIX_AIC_1_2`（1 AIC + 2 AIV / block），`blockDim = num_req * num_heads`。

## 先说三处代码已经变了的

原表里有三条"待办"已经落地，表格按新代码重写：

| 原表的说法 | 现状 |
|---|---|
| 每隔 16 行取新的 $g_j$，后续按 fla 调整为 32 行 | 已是 32 行，且锚点取**块中点**而非末行（`kRebaseRows = 32`，`mRow = (r0 + SB/2) * D`）。约束是**正指数**不能超 fp32 的 +88.7，中点锚把 32 行块的裕量压到 +49，与 16 行/末行锚等价。A/Aqk 的 matmul 由 8 次降到 4 次，实测只值 3.5% |
| 求逆是反复在 AIV/AIC 同步的过程，改 fla 方式减少同步开销 | 已改完并作为默认（`KDA_CHUNK_INVERSE_SQUARING = 0`）。两个 32×32 对角块在**两个 AIV 上各解一块**，off-diagonal 由 Cube 两次 32×32 完成。实测 110.8 → 104.7 us/chunk。但**省下来的不是同步开销**：一个 mode-2 会合实测只值 ~0.4 us，5 个会合约 2 us；真正的收益是 Cube 工作量从 11 次 64×64 降到 2 次 32×32。详见 [TRIANGULAR_INVERSE_COMPARISON.md](./TRIANGULAR_INVERSE_COMPARISON.md) |
| 另一个 core 偶尔参与 | 仍然基本成立，但它现在有了一段**实质工作**：`SolveDiagBlockHelper` 解 C11，是整个 kernel 里唯一真正双 AIV 并行的向量段，实测这一拆分值 18 us/chunk |

"每组 aicore 处理一个 req 的一个 head，由 cu_seqlens 给出"——准确。
`head = blockIdx % numHeads`、`req = blockIdx / numHeads`，`bos/eos` 取自
`startsGm`（即 cu_seqlens）。AIV 上 `GetBlockIdx()` 是全局 AIV 号，所以
block 号要减半。

另外 `CHUNK_KDA_EXECUTION.md` 本身还停在旧实现上（16 行 rebase、8 次 A/Aqk
matmul、重复平方求逆、每 chunk 九个会合），读的时候以本文和头文件为准。

## Prologue：每个 block 一次，不在 chunk 循环里

原表把这两行排在串行序列开头，容易误读成每 chunk 的代价。它们**整个
(req, head) 只做一次**，摊到所有 chunk 上：

| 计算/访存 | RunAic | RunAiv (subcore-0) | RunAiv (subcore-1) |
|---|---|---|---|
| 下三角 mask | — | `BuildMask`：`Duplicate` 0.0f 清 `[64,64]`，再 **2080 次 scalar `SetValue`** 写 1.0f，两侧各一次 V↔S 同步。不能用向量写法：行尾 `Duplicate(x[r*BT+r], ...)` 的元素偏移只在 `r % 8 == 0` 时 32 B 对齐，否则 `aivector error` | — |
| `dt_bias` / `lower_bound` 张量化 | — | 一次 `DataCopy` + 一次 `Duplicate`。`Div` 两个操作数都要张量，没有 `Divs` | — |
| load state 至 workspace | — | `slot > 0` 时 128 行 GM→UB→workspace；否则 `Duplicate` 0。pool 是 `[slots,H,K,V]` 且 `stride(2)==1`，逐字拷进来就是 $\mathbf S^\top$，后面所有 matmul 都按这个转置约定写 | — |
| `track_len == 0` 时落 checkpoint | — | `CopyStateOut` | — |

注意 state **始终不驻留 UB**，它住在 workspace（GM）里，`[128,128]` fp32 = 64 KB，
而 UB 已用到 187/192 KB。这决定了下面第 11 步的形态。

## 每 chunk 的串行执行

一行一个阶段，顺序就是 `RunAiv` / `RunAic` 里 chunk 循环的顺序。
`══` 标记一次 mode-2 会合（AIV 侧 set-then-wait，AIC 侧 wait-then-set）。
每 chunk **4 次 AIV→AIC 发送、3 次回执**：求逆那次故意单向，让 AIC 的两次
32×32 和 AIV 的辅助算子段重叠；tracked prefix 命中的那个 chunk 再加 2 对。
除此之外，会合两侧严格互等，这就是"流水完全没有重叠"的机制。
会合必须严格成对，差一个就是 0% AICore 的静默死锁。

| 计算/访存 | RunAic | RunAiv (subcore-0, worker) | RunAiv (subcore-1, helper) |
|---|---|---|---|
| load qkva 至 ub 并 cast fp32 | | `LoadChunk`/`StageCast`：行在 GM 里按 `H` 跨步，每张量两次半 chunk `DataCopy` + `Cast`，共用一个 8 KB staging tile。**必须有 `V_MTE2`**，否则 MTE2 跑在 `Cast` 前面把 tile 盖掉，六次 `Cast` 全读到最后一次拷贝的行 | 空转 |
| 生成 beta 的原始值 | | `b` 每行只有 1 个 bf16 且按 `H` 跨步，`DataCopy` 一个 datablock 只搬 1 个有用元素，所以 scalar `GetValue` 逐行取 | 空转 |
| 预计算 qn / kn / g / beta / cumsum | | `PrepareFactors`，语义同 recurrent 的 `kda_pre_torch`，但一次做整 chunk 64 行。精度点：`tl.sum` 的两半折叠——用两次 `WholeReduceSum(mask=64, repeat=rows, srcRepStride=16)` 一次性处理所有行的 lo/hi 两半，再 `Add` 合并，匹配 `tl.sum` 的折叠顺序；`sqrt` 和倒数走 **Scalar**（向量 `Div`/`Sqrt` 差 1 ulp）、gate 用真除法。**cumsum 是 63 次串行行 `Add`**，每次夹一个 `PipeBarrier<PIPE_V>` | 空转 |
| 逐行广播、per-row `Muls` | | 原表说的"rows 16→64 引入不必要的操作"指这里：归一化因子和 beta 都是 per-row 标量，`Muls` 只吃标量，所以 64 行就是 64 条指令，recurrent 的 16 行只要 16 条。同理 `Add` bias、`Div` gate 各 64 条 | 空转 |
| $\mathrm{row}^{(R)}, \mathrm{col}^{(R)}$ 的构造 | | `WriteRebased`，**2 个 rebase 块**（不是 4 个）。每块：32 次 `Sub` 作 $\gamma - m$（带 `src1RepStride = 0` 的广播 `Sub` 会让 vector core 故障，所以只能逐行）、一次 `Exp`、两次 `Mul` 分别乘进 kn/qn，再 `StoreRows` 到 workspace。$A_{kk}$ 与 $A_{qk}$ 共用 $e^{\gamma-m}$，列因子只算一份 | 空转 |
| ══ 会合 1 ══ | | | |
| $A_{kk} = \mathrm{row}_K\mathrm{col}_K^\top$，$A_{qk} = \mathrm{row}_Q\mathrm{col}_K^\top$ | **4 次** `CubeMatmul<false,true>`，`[32,128] @ [r1,128]ᵀ`，`dstRowStride = BT` 直接落成 `[64,64]` tile 的一个 `[32, r1]` 块。`B_TRANS` 是因为 col 存成 `[N,K]`，已经是 Zn 朝向 | 等 Cube | 空转 |
| ══ 会合 1 回 ══ | | | |
| 构造 $-\operatorname{StrictTril}(\operatorname{Diag}(\beta)A_{kk})$ 与 $\operatorname{Tril}(A_{qk})$ | | $A_{kk}$ 先做：64 次 `Muls(-beta_r)`，再逐行 `Duplicate` 掉 $[r, BT)$ 得 `tril(-1)`，存到 `kWsDpow`——**只存这一个**，`SolveDiagBlocks` 会从这里读回来。$A_{qk}$ 后做（逐行清 $[r+1, BT)$），所以留在 `tmp` 里的是 $A_{qk}$。注意**没有"对角线 scalar 加 1"**——单位阵是在每个对角块求逆末尾 `AddIdentity` 一次性加的 | 空转 |
| 求逆对角块 $C_{00}$ / $C_{11}$ | | `SolveDiagBlocks`：先把 B10、B11 `StoreBlock` 发出去，mode-1 会合一次，然后是前代循环：`i = 2..31` 共 30 行，第 i 行 i-1 次 `Axpy`，一块 465 次；每行**一次** scalar 访问取齐系数（第 j 步写 `[0,j)`、读的是 lane j，不冲突）。最后 `AddIdentity` + 清掉 (0,32) 的 mask 残噪 | `SolveDiagBlockHelper`：同一段代码，块落在 (0,32)。它**没有 chunk 的 UB 副本**，所以 B11 要从 workspace 读回来。这是唯一的真并行向量段 |
| ══ 会合 2（单向，无回执）══ | | | |
| $C_{10} = -C_{11}B_{10}C_{00}$ | 2 次 `CubeMatmul` 32×32，直接写进 `kWsX` 的 (32,0) 空洞，`aRowStride`/`bRowStride` 让它原地读 64×64 tile 的子块，省掉打包。B10 带着负号来的，所以 Cube 的 `+` 就是参考实现的 `-` | 不等回执，直接往下做辅助算子——这两次 Cube 和下面一格是**本 kernel 唯一的 AIC/AIV 重叠** | 空转 |
| 辅助矩阵前置：$\operatorname{Diag}(\beta)(\Gamma^{1\to C}\odot\mathbf K)$、$\operatorname{Diag}(\beta)\mathbf V$、$\Gamma^{1\to C}\odot\mathbf Q$、$\Gamma^{(i+1)\to C}\odot\mathbf K$ | （同时在算 $C_{10}$） | `BuildAuxOperands`，拆法同 fla。$\beta$ 折进**右操作数**而不是去缩放 `[64,64]` 的逆（`(X\cdot\beta)\,M = X(\beta\,M)`），一次 `[64,128]` 的 `Muls` 就够。`v` 在这里**重新从 GM staging**：UB 腾不出第三块 `[64,128]`，所以整个 chunk 期间 v 不驻留 | 空转 |
| ══ 会合 3（走 `kFlagAivToAicAux`）══ | | | |
| $\mathbf W = \mathbf M(\Gamma^{1\to C}\odot\mathbf K)$，$\mathbf U = \mathbf M\mathbf V$，$\mathbf W\mathbf S$ | 3 次 `CubeMatmul`。$\mathbf W\mathbf S$ 用 `B_TRANS`，因为 state tile 存的是 $\mathbf S^\top$ | 等 Cube | 空转 |
| ══ 会合 3 回 ══ | | | |
| $\tilde{\mathbf V} = \mathbf U - \mathbf W\mathbf S$ | | 一次 `[64,128]` `Sub`，前面两次 `LoadRows` | 空转 |
| ══ 会合 4 ══ | | | |
| $\mathbf O$ 的两项 + state 增量 | 3 次 `CubeMatmul`：$(\Gamma\odot\mathbf Q)\mathbf S$（`B_TRANS`）、$\operatorname{Tril}(A_{qk})\tilde{\mathbf V}$、以及 $\tilde{\mathbf V}^\top(\Gamma^{(i+1)\to C}\odot\mathbf K)$——**操作数对调**，结果直接就是转置好的、与 $\mathbf S^\top$ 对齐的形状 | 等 Cube | 空转 |
| ══ 会合 4 回 ══ | | | |
| chunk 输出 | | `FinishChunk`：一次 `Add` 合两项，`Cast` 回 bf16，分两半按 `H` 跨步写出 | 空转 |
| chunk state update：$\mathbf S\leftarrow\operatorname{Diag}(e^{\gamma^C})\mathbf S + \ldots$ | | `ApplyStateUpdate`：**128 轮**，每轮两次 `LoadRows` + `Mul` + `Add` + `StoreRows`。因为 tile 是 $\mathbf S^\top$，衰减沿列，所以是整行**向量** `Mul` 而不是 per-row `Muls`（这个转置细节曾是真 bug，`kWsOut` 对 `qg @ S0.T` 是 7e-10、对 `qg @ S0` 是 3.5e-3）。一行一行是因为 64 KB 的 tile 塞不进任何一个 ≤32 KB 的 UB buffer | 空转 |

tracked prefix（`i*BT < track_len <= (i+1)*BT` 的那一个 chunk）额外插两对会合：
AIV 把 `kWsKDecay` 按第 r-1 行重算 → Cube 按 r 行重做增量 → AIV 落 checkpoint、
从 `kWsStateSave` 把 state 还原、再把 `kWsKDecay` 按末行重算 → Cube 按 BT 行
重做本 chunk 自己的增量。`kWsStateUpd` 只有一块，所以必须重做。

## 为什么没有重叠

两个独立原因，都是实测的（H8 T256，4 chunk，`blockDim` 8）：

**1. 会合把两个核串成一条。** AIC 记账工作 43 us/chunk + AIV 记账工作
63 us/chunk ≈ 115 us 的 wall time，即 kernel 付的是**和**而不是 max。
但要注意归因：flag 指令本身很便宜（~0.4 us 一对，4 对约 1.6 us），
贵的是"一侧干活时另一侧只能等"。

**2. `blockDim = num_req * num_heads` 把卡喂不饱。** card 4 有 20 个 AIC，
H8 只用 8 个，而每个 block 内部整条 chunk 链是串行的。所以 wall time 精确等于
`(每 block 的 chunk 数) × 115 us`，延迟对 T 线性（T1024/2048/4096 →
2.05 / 3.92 / 7.09 ms），而 fla 近乎平的（2.00 / 2.05 / 2.34）——fla 把 forward
拆成五次 launch，每次在所有 chunk 上开 grid，只有 state scan 是 chunk 串行的。

算力完全不是瓶颈：

| pipe | AIC（447 us 合计） | AIV（481 us） |
|---|---|---|
| cube / vec | **23.3（5.2%）** | 54.0（11.2%） |
| scalar | 70.3（15.7%） | 80.5（16.7%） |
| mte2 | 32.5 | 51.5 |
| mte3 | — | 64.7 |

Cube 只有 5% 利用率。这解释了为什么"减少同步开销""削减求逆的算术量"这类
优化都只值几个百分点：

| 做过的优化 | 收益 |
|---|---|
| rebase 32 行 + 中点锚（A/Aqk matmul 腰斩） | 3.5%（7.36 → 7.09 ms） |
| 整个 6 步求逆砍到 3 步（答案是错的，纯计时） | 12.5 us/chunk，即 ~11% |
| 换成 fla 的分块前代求逆（算术量 −22.7×） | 110.8 → 104.7 us/chunk，≈5.5% |
| 双 AIV 分摊两个对角块 | 122.6 → 104.7 us/chunk，18 us |

注意最后两行的对比：**求逆换算法本身是亏的**（单 AIV 做两块是 122.6，比原来的
重复平方 110.8 还慢），是双核拆分把它救回来的。这和 Triton 侧早先那条测量一致
——层次化求逆砍掉 78% FLOPs 却慢 5-8%。

## 结论：杠杆在哪

不在单个阶段的算术量，也不在 flag 指令数。两个方向，按收益排：

1. **拆 launch。** 把 chunk-串行的阶段解耦，只让 state scan 在 chunk 维度串行，
   其余阶段各自在所有 chunk 上开 grid，把 `blockDim` 推过 20。这会改 `blockDim`
   和 workspace 的尺寸计算，但不动算子签名。做了这一步，求逆的会合自然消失，
   算法选择才退化成"该放 Cube 还是向量核"的二阶问题。
2. **在当前结构内，啃 AIV 的 scalar 和 MTE3。** AIV 的 scalar 占 16.7%、MTE3 占
   13.5%，都比 vector 的 11.2% 大。候选：`BuildMask` 的 2080 次 `SetValue`
   （每 block 一次，可以预计算成常量搬进来）、`ApplyStateUpdate` 的 128 轮
   小搬运、`WriteRebased` 的逐行 `Sub`、以及 `PrepareFactors` 里那一堆 per-row
   `Muls`/`Add`/`Div`（64 条指令各一遍）。

## 读代码时容易踩的

* `CHUNK_KDA_EXECUTION.md` 描述的是旧实现（16 行 rebase、8 次 A/Aqk matmul、
  重复平方求逆、9 个会合），别拿它对当前代码。重复平方的路径还在，
  `-DKDA_CHUNK_INVERSE=squaring` 能编出来，但不是默认。
* workspace 必须 `at::zeros`。A/Aqk 的严格上三角从来没被写过，紧接着的 mask
  乘法会把未初始化的垃圾变成 `0 * NaN`。
* 没有片上 UB→L1 通路。看起来像的那个带 ND2NZ 的 `DataCopy` 是高层 Matmul API
  的软件模拟，要 `REGISTER_MATMUL`。所以 Cube 的每一个操作数都绕 GM。
* AIV 的 UB 在 MIX 模式下是 **192 KB**，不是 256；当前用到 187 KB。
* 连续 `CubeMatmul` 之间需要三个反向 barrier（`MTE1_MTE2` / `M_MTE1` /
  `FIX_M`），单次不需要，连起来不加就故障。
* `msprof op` 看不见这个 kernel——它用裸 `<<<>>>` launch。要用 application 模式：
  `msprof --output=DIR --aic-metrics=PipeUtilization --ai-core=on
  --application="python driver.py"`，然后读
  `mindstudio_profiler_output/op_summary_*.csv`。
