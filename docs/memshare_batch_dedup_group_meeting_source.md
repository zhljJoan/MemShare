# MemShare Batch 去重与分层复用：组会叙事底稿

> 用途：先固定组会故事线和证据边界，再据此重做 PPT。详细实现、公式和完整日志索引见 [最终方案设计](../journal_extension_block_unique_final_method.md)。

## 0. 先讲结论

这项工作不是提出一个新的分布式 GNN 架构，而是对 **MemShare Batch 去重表示和执行链**的升级。

一句话主线是：

> 原 MemShare 的直接去重解决“少读、少存”；升级版把相同去重关系定义成跨阶段共享的 Batch 契约，进一步实现“少重建、少展开、少计算”。

它经历了两个版本，但数据本身包含三种计算粒度：

1. **object**：去重后的节点 UID 或图边 EID，适合读取基础数据和复用与时间无关的计算；
2. **temporal state**：唯一的 `(UID, timestamp)`，同一 UID 在不同时间仍是不同状态；
3. **edge occurrence**：采样器在本 batch 中对一条边的一次使用，所有 occurrence 都必须保留。

版本二没有比版本一删除更多训练边，也没有改变 temporal state 的等价关系。变化在于：**同一套 object/state/occurrence 映射成为统一 Batch 契约，并继续被 Block、memory updater 和 attention 使用。** 训练路径用 sampler-native producer 生成它；evaluation 与 fallback 仍可由原 Python/GPU compaction 生成兼容表示。

## 1. 问题：直接去重之后，复用为什么仍会丢失

时序邻域采样会在一个 batch 内重复遇到同一节点、同一图边和同一 `(UID,timestamp)` 状态。三种重复的处理边界不同：

| 粒度 | 可以合并什么 | 不能合并什么 |
| --- | --- | --- |
| 节点/图边 object | node feature、edge feature、当前 memory、mailbox 的读取；部分与时间无关的投影 | 同一 UID 在不同 timestamp 下的完整状态更新 |
| temporal state | 完全相同的 `(UID,timestamp)` 状态行 | 同一 UID 的不同 timestamp |
| edge occurrence | 不合并，只用映射读取 object/state | destination、时间差、softmax、dropout 或训练角色不同的边使用 |

原 MemShare 已经识别这些等价关系，但去重发生在采样返回之后：Python/GPU 再执行 UID、EID 和 `(UID,timestamp)` 的 `unique`、顺序恢复与 Block 构造。它有效减少了远程读取和状态行，却留下两个执行问题：

- sampler 已经遍历过 occurrence，返回后又做一次全局 unique、remap 和结构重建；
- legacy 映射主要服务 fetch 和 Block 数据附着，虽然 Block 中仍保存部分映射，但它们尚未形成统一 producer/consumer 契约，也未被用于 GRU object-level projection 和显式 segment execution。

因此核心问题不是“MemShare 没有去重”，而是：

> **去重结果的复用范围有限，还没有成为贯穿整个 batch 的统一执行契约。**

## 2. 两个版本的演进

### 2.1 版本一：原 MemShare 直接去重

```text
Sampler 输出完整 occurrence
        ↓
Python/GPU: UID unique + EID unique + (UID,timestamp) unique
        ↓
object 表 + temporal-state 表 + inverse mapping
        ↓
按 object 读取 feature / memory / mailbox，构造 DGL Block
        ↓
下游按 state / occurrence 展开并执行通用算子
```

这一版已经做到：

- 同一节点/图边的基础数据在 batch 内只读取一份；
- 相同 `(UID,timestamp)` 只保留一个状态行；
- 正样本、负样本、root 角色和全部采样边 occurrence 仍通过 inverse mapping 恢复。

它的局限不是语义，而是执行边界：已有映射面向 fetch/Block 保存，没有被定义成所有 consumer 共同遵守的 Batch 契约。

### 2.2 版本二：分层紧凑执行

版本二把原有去重语义显式组织成一份 Batch 中间表示。下面的箭头表示“映射指向”，不是 object 展开方向：

```text
temporal state ── state_to_uid ──► node object
edge occurrence ─ edge_to_object ► edge object
root occurrence ─ root_to_state ─► temporal state
edge occurrence ── src/dst state ► temporal state
```

这份契约有两个 producer：

```text
Batch contract
├─ 训练优化 producer：sampler-native thread-local compaction + stable merge
└─ evaluation / fallback producer：原 Python/GPU compaction

两者 ──► Direct Block / Compact GRU / Fused Segment Attention
```

训练 producer 在返回 Python 前，对每个采样线程的局部结果做 hash compaction，再按 root-first、线程顺序和线程内首次出现顺序稳定合并。它输出：

- UID/EID object 表；
- temporal-state 表和 `state_to_uid`；
- `root_to_state`；
- 保持原采样顺序的 occurrence COO，以及 occurrence 到 edge object 的映射。

下游不再重新猜测这些关系，而是按自身正确粒度消费同一套映射。

| 维度 | 原 MemShare 直接去重 | 分层紧凑执行升级 |
| --- | --- | --- |
| 构造位置 | sampler 返回后的 Python/GPU | 训练用 sampler-native producer；evaluation/fallback 仍用 legacy producer |
| object/state/occurrence 数量 | 已去重 object/state，保留 occurrence | **语义数量相同** |
| 表示方式 | 面向 fetch/Block 的 consumer-specific tensor | 显式、可由多个 consumer 复用的 Batch 映射 |
| Block | 高层 DGL 构造并按需组织邻接 | 用原序 COO 创建同语义 DGL Block，并随 Block 保存本 batch 的 segment plan |
| GRU | 整个 GRU 按 temporal state 执行 | mailbox/hidden 投影按 UID object 复用；时间编码和最终更新仍按 state 执行 |
| Attention | DGL 的 score、softmax、dropout、加权和与聚合算子链 | edge occurrence 不删；按 destination segment 融合执行 |

这里的 Direct Block 不执行新的去重，也没有绕开 DGL。它用已有 occurrence COO 创建 DGL Block，让 DGL 在 builder 阶段准备所需稀疏格式，并显式取得 `dst_ptr` 与 `csc_to_occurrence`；这两个数组只在当前 batch/Block 生命周期内保存，供 Fused Segment Attention 的前向和反向复用。

Direct Block **没有消除 COO→CSC 转换**。DGL 本身也会缓存已生成的稀疏格式，因此贡献不是“避免 DGL 重复转换”，更不是跨 batch cache；变化是把本 batch 的 segment plan 显式暴露给融合 attention 使用。Fused Segment 开关同时切换这条 builder 路径，所以相应增量结果始终是 builder + plan + kernel 的联合路径。

## 3. 完整执行链与各组件职责

```text
Sampler
  ↓
训练：sampler 内分层紧凑化
评估：原 MemShare 直接去重
  ↓
Object 表 + temporal state + edge occurrence COO
  ↓
Direct Block / segment plan + object 数据读取
  ↓
Compact GRU raw update
  ↓
原 MemShare smooth / Filter / feature attach
  ↓
Fused Segment Attention
  ↓
原 decoder、loss、反向传播与持久状态写回
```

三个最终组件只是同一个 Batch 表示的 producer/consumer：

| 组件 | 只负责什么 | 明确不做什么 |
| --- | --- | --- |
| sampler 内分层紧凑化 | 提前生成 object/state/occurrence 表及映射，移除 post-sampling 全局 unique/remap | 不减少 legacy 已有的 state、edge 或 remote object 数量 |
| Direct Block + Fused Segment Attention | 保存 destination segment plan；融合 score、softmax、dropout、加权和与聚合 | 不删除 occurrence，不把不同 timestamp 合并，不消除 occurrence K/V |
| Compact GRU | 对 UID 不变的 mailbox/hidden affine 只算一次，再按 `state_to_uid` 组合 state-specific time 分量 | 不把同 UID 不同 timestamp 的 raw update 合并 |

Compact GRU 之后的 smooth、Filter、feature attach、latest-state 选择和持久 memory/mailbox 写回继续使用原 MemShare 语义。它们不是被 kernel 跳过，而是接收 Compact GRU 产生的 state-level raw update 后原样执行。

## 4. 不变量：这项优化没有改变什么

当前最终对比固定保持：

- 图划分、节点/边 owner 和热点集合不变；
- 正样本、负样本及其训练角色不变；
- boundary-decay 邻居采样规则和概率不变；
- 每轮全局 local/remote node/edge 计数在 OFF/ON 间 bit-exact；
- temporal state 仍按 `(UID,timestamp)` 区分；
- edge occurrence 的数量、顺序、EID、时间差和 destination 不变；
- smooth、Filter、feature attach 和持久状态更新规则不变；
- Fused Segment 的 native dropout 与 PyTorch 路径对齐 mask 形状和 CUDA RNG 消耗。

浮点归约顺序和 kernel 编排发生变化，因此不能宣称逐 bit 等价。最终质量采用 AP/AUC 的直接完整训练结果判断。

## 5. 目前已经覆盖的实验

### 5.1 最终端到端设置

- **模型**：`TGN_large`，historical memory，time-aware GRU；
- **硬件**：双机 8 张 NVIDIA A40，每机 4 卡；
- **网络**：10GbE `ens4f1np1`，NCCL Socket，IB/GDR 关闭；
- **软件**：PyTorch 2.1.1+cu118、CUDA 11.8、Triton 3.1.0；
- **固定参数**：`alpha=0.3`、`beta=0.1`，冻结 adaptive parameters；
- **数据集**：WIKI 50 epoch × 3 seeds（6773、95842、12345）；LASTFM 50 epoch、WikiTalk 10 epoch、Flight 10 epoch，后三者均为 seed 6773。

最终结果是统一源码中的直接 A/B：

- **MemShare 直接去重基线**：分层紧凑化、Fused Segment、Compact GRU 全关；
- **升级链路**：三项全开，训练 sampler 使用分层紧凑化，evaluation sampler 保持原直接去重。

这个 baseline 代表统一代码中的 MemShare 直接去重链路，不是未经同步的历史 master。最终百分比也不是三个组件的边际收益相加。

### 5.2 效率结果

表中均为耗时下降；`critical` 是每个 epoch 先取 8 个 rank 的最大值，再在相应 epoch 范围内求平均。`steady` 排除 epoch 0 的首次 JIT/预热，作为主口径；`tail` 对 WIKI/LASTFM 取最后 10 轮，对 WikiTalk/Flight 取最后 4 轮。`total wall` 包含每轮 train、validation、test、同步与本轮簿记，但不包含 epoch 计时开始前的进程/NCCL/模型/dataloader 初始化，也不包含 checkpoint 写盘，因此不是整个作业从启动到退出的 wall-clock。

WIKI 的 `±` 是相同 seed 先组成 OFF/ON 配对后，三个 paired 百分比的均值 ± 样本标准差；不是 rank/epoch 波动或置信区间。其余数据集只有一个 seed，所以不报告 `±`。

| 数据集 | full train critical | steady train critical | tail train critical | steady total-wall critical |
| --- | ---: | ---: | ---: | ---: |
| WIKI（3 seeds） | 0.1887 → 0.1841 s（2.29% ± 5.96%） | **0.1673 → 0.1510 s（9.67% ± 2.72%）** | 0.1667 → 0.1510 s（9.33% ± 3.78%） | **0.2635 → 0.2447 s（7.07% ± 2.64%）** |
| LASTFM | 1.2784 → 1.1314 s（11.48%） | **1.2600 → 1.1012 s（12.59%）** | 1.2480 → 1.1000 s（11.86%） | **2.1476 → 1.9339 s（9.93%）** |
| WikiTalk | 7.9580 → 7.1570 s（10.07%） | **7.8700 → 7.0067 s（10.97%）** | 7.7975 → 7.0075 s（10.13%） | **12.7933 → 11.7800 s（7.93%）** |
| Flight | 2.5670 → 2.4180 s（5.84%） | **2.4644 → 2.2689 s（7.97%）** | 2.4650 → 2.2700 s（7.81%） | **4.0233 → 3.7944 s（5.69%）** |

箭头左侧为 all-off、右侧为 all-on，单位为 `s/epoch`。WIKI 的绝对时间是三个 seed 的算术均值，括号内百分比仍按相同 seed 配对后求均值 ± 样本标准差；逐 epoch 日志只打印到 `0.01 s`，表中更多小数来自多轮平均。仓库只归档了 `gpu07` 四个本地 rank 的 stdout，绝对值是包含 8-rank barrier 的已归档 rank critical 近似；严格的 8-rank 最大值需要另一节点原始 stdout。各 seed 的绝对值见[完整方案记录](../journal_extension_block_unique_final_method.md#121-效率)。

因此当前可用的效率结论是：

> 在固定全局采样工作量下，稳态训练关键路径下降 **7.97%--12.59%**，包含 validation/test 的整轮关键路径下降 **5.69%--9.93%**。

WIKI 的 full 结果受 epoch 0 编译/初始化影响较大，不能把 steady 的约 10% 写成所有 full run 的结果。WIKI 单个 steady epoch 约 0.15 秒，而逐 epoch 日志只保留到 0.01 秒，因此 steady critical 的具体百分比存在明显量化误差；高精度 full `avg_time` 只确认方向一致。

OFF/ON 每轮经 all-reduce 得到的全局 local/remote node/edge 计数 bit-exact。steady shared count 的变化为 WIKI `+0.01%`、LASTFM `-1.40%`、WikiTalk `-0.31%`、Flight `-0.88%`；shared count 不是通信字节或耗时。后三个数据集存在小幅有利轨迹漂移，所以不能把全部效率收益归因于计算去冗余。

### 5.3 质量与 time-to-target

`pp` 表示百分点：例如 AP 从 95.0% 到 94.8% 是 `-0.2 pp`。checkpoint 只按最高 validation AP 选择，再报告对应 test。TTT 从 epoch 0 累计至首次命中表中 validation AP 阈值的 epoch，并包含命中 epoch 的 epoch-critical total wall；负号表示更快达到，正号表示更慢。阈值是描述性分析而非预注册指标，不同数据集之间不能直接比较。

| 数据集 | best-val checkpoint 对应 test AP / AUC 变化 | TTT 的 validation AP 阈值 | 含评估 TTT 变化 |
| --- | ---: | ---: | ---: |
| WIKI（3 seeds） | -0.0687 ± 0.2062 / -0.0883 ± 0.2469 pp | 95% | 11.36 → 11.78 s（+3.96% ± 8.39%） |
| LASTFM | -0.1556 / -0.2869 pp | 90% | 78.60 → 75.18 s（-4.34%） |
| WikiTalk | -0.0246 / -0.0364 pp | 97% | 103.45 → 95.79 s（-7.42%） |
| Flight | +0.0008 / -0.0051 pp | 96% | 33.18 → 31.86 s（-3.98%） |

当前允许的表述是：

- WIKI 三 seed 的 best-val checkpoint 对应 test 平均回退小于 0.1 pp；
- 所有已测 run 的 best-val checkpoint 对应 test 最坏回退小于 0.5 pp；
- 不能写“严格无损”或“统计非劣”；LASTFM、WikiTalk、Flight 仍是单 seed；
- WIKI 三个 seed 都晚 1--3 个 epoch 达到 95% validation AP，所以 WIKI 的 TTT 没有稳定改善。该结果不否定每轮吞吐提升，但不能宣称全面加速收敛。

### 5.4 结构、组件与 Kernel 证据

| 证据层次 | 已覆盖结果 | 解释边界 |
| --- | --- | --- |
| 复用机会 | WIKI 四卡、单配置、1 epoch 聚合：580,153 个 state / 66,287 个 UID object（`S/U=8.75×`）；4,284,775 个 occurrence / 325,717 个 edge object（`E/O=13.15×`） | 没有 per-batch 分布或其他数据集复现；只说明 legacy 去重后的跨层放大，不是版本二又删除了 8.75×/13.15× 数据 |
| sampler 内分层紧凑化增量（内部实验代号 P0） | WikiTalk seed 6773，八卡 ABBA（OFF→ON→ON→OFF），`--skip_eval`，每次 10 epoch：steady critical 下降 4.01%，相邻配对 4.07%/3.95% | 单 seed；Fused Segment 与 Compact GRU 两侧已开启；shared count 同时下降 0.96% |
| Direct Block + Fused Segment 历史增量 | 四数据集各单 seed，八卡 steady critical 下降 4.70%--6.33% | 最终 native-dropout RNG 对齐前的历史结果；开关包含 builder/segment-plan，steady shared count 约 `+0.16%` 至 `+5.48%`，不能称纯 kernel A/B |
| Compact GRU 历史增量 | 在 Fused Segment 已开启的链路上，四数据集各单 seed，steady critical 下降 2.26%--4.40% | UID affine 复用与 gather/pointwise 融合的联合结果；shared count 约 `-1.34%` 至 `+5.63%`，不能与其他百分比相加 |
| Segment 局部 benchmark | A40、预构建 Block、固定 dropout mask、random edge permutation，F+B：1.8555 ms → 0.7843 ms，下降 57.7% | kernel reference microbenchmark；不含 mask 生成、完整 Q/K/V 投影和 builder |
| Compact GRU 局部 benchmark | A40、`S=13865,U=1579`，F+B：1.0236 ms → 0.7865 ms，下降 23.2% | baseline 已是 object/state 拆分但 gather/ATen pointwise 未融合的 compact reference，不是原 MemShare state-level `GRUCell` |

### 5.5 正确性覆盖

当前记录包括：

- sampler 对 50 组 recent 与 30 组 boundary-decay 随机输入的 parity；
- occurrence/state/root/edge mapping，以及同 UID 同/异 timestamp 边界；
- Fused Segment 前向、Q/K/V 反向、空段、非均匀 degree、degree 630、非 identity edge permutation；
- fixed dropout mask 与 native-dropout CUDA RNG state；
- Compact GRU 输出、time/GRU 参数梯度和 fallback；
- smooth/Filter、persistent selector、feature attach 及实验性 RNN/Transformer compact consumer；
- 该最终实现修订的测试记录为 36 passed，完整训练 loss 均有限，无运行或 NCCL 错误。实验性 RNN/Transformer/smooth compact consumer 只完成算子 parity、默认关闭，未进入最终端到端配置。

### 5.6 仍未覆盖的实验

- LASTFM、WikiTalk、Flight 的多 seed；
- 1/4/16 GPU 强扩展或弱扩展；
- IB 和其他网络带宽、其他 GPU 架构；
- 峰值显存、通信字节和通信时间的系统报告；
- JODIE/APAN 等其他 memory updater 的最终端到端收益；
- Direct Block 的独立端到端归因；
- 四数据集统一的 Batch 重复率与跨阶段开销画像；
- 与未经修改 upstream/master MemShare 的直接 A/B；当前 baseline 是统一源码中的三个最终开关全关；
- P0 在 evaluation、多层、无负样本、identity/reversed 链路上的覆盖；当前 P0 只用于一层、带负样本、非 identity/non-reversed 的训练 sampler。

### 5.7 证据位置

| 内容 | 路径 |
| --- | --- |
| WIKI 复用机会探针 | `outputs/dual_dedup_view/wiki_4gpu_1epoch.log` |
| Segment / Compact GRU 局部 benchmark | `outputs/kernel_skill_loop/20260818/` |
| Direct Block + Fused Segment 四数据集消融 | `outputs/fused_segment_attention_ab/20260817/8gpu_multidataset/` |
| Compact GRU 四数据集增量消融 | `outputs/compact_gru_fused_ab/20260818/8gpu_multidataset/` |
| P0 WikiTalk ABBA | `outputs/p0_hierarchical_8gpu_20260818/` |
| 最终 all-off / all-on 完整训练 | `outputs/p0_hierarchical_full_8gpu_20260818/` |

## 6. 需要补充的 Motivation 实验

### 6.1 要回答的问题

只展示“重复率高”还不够。最关键的问题是：

> **原 MemShare 已经完成 UID/EID/temporal-state 去重，为什么仍需要新的 Batch 表示？**

Motivation 实验应同时证明：

1. legacy 去重后，object 仍会被放大为更多 temporal state 和 edge occurrence；
2. 保持 legacy producer 不变，只升级去重后的 builder/consumer，是否仍有稳定收益；
3. 保持 compact consumers 不变，再把 producer 从 legacy 换成 sampler-native，是否进一步减少 post-sampling compaction。

### 6.2 一个实验：Legacy-Dedup Residual Reuse

实验由一组结构统计和两组分阶段 A/B 组成，使用相同 workload 配置。它分别检验 **post-compaction execution reuse** 与 **sampler-native construction**，不是纯 consumer kernel A/B。

#### A. 四数据集 Batch redundancy census

在统一源码的 MemShare 直接去重基线上，使用 8 GPU、seed 6773，至少运行 2 epoch，只观测 epoch 1 的前 100 个 batch或完整 epoch 1。现有 `dual_dedup_view` 只有 epoch 聚合的 `S/U/E/O`，缺少 `N`、per-batch 分位数和下述分段时间；本实验需要新增只读计数/计时探针，但不能新增一次 `unique`。

| 记号 | 记录内容 |
| --- | --- |
| `N` | temporal unique 之前的 root + sampled-node occurrence 数 |
| `S` | unique `(UID,timestamp)` temporal-state 数 |
| `U` | unique node UID object 数，直接取 `dist_nid.numel()` |
| `E` | 保留的 edge occurrence 数 |
| `O` | unique graph EID object 数，直接取 `dist_eid.numel()` |

报告：

$$
R_{state}=S/U,\qquad R_{edge}=E/O,\qquad R_{temporal}=N/S.
$$

对每个同步 global batch，先跨 rank 累加实际处理的行数，再计算该 batch 的比值，例如：

$$
r_b=\frac{\sum_r S_{b,r}}{\sum_r U_{b,r}}.
$$

最后对 $r_b$ 报告 median 与 P10--P90。这里的分子、分母是各 rank 实际处理行数之和，不是跨卡再次全局去重后的 UID/EID 数；也不能直接平均各 rank ratio。

同时记录 legacy preparation 的四段时间，host wall 与 CUDA event 分开报告：

1. UID/EID object unique；
2. temporal unique + first-occurrence 顺序恢复；
3. metadata/COO remap；
4. 实际 DGL Block 构造；COO→CSC/稀疏格式准备只通过 profiler 观察原 DGL attention 真正触发的工作，**不能为了计时额外调用** `adj_tensors("csc")` 改变 baseline。

由 shape 直接推导两类计算放大，不额外物化 tensor：

- GRU 中 UID-invariant mailbox/hidden affine 的可复用行比例；
- edge-object feature 展开到 occurrence 的行数和字节数。

已有 WIKI 先导结果是 `S/U=8.75×`、`E/O=13.15×`。在 `TGN_large` 的 `mail/time/hidden=372/100/100` 下，GRU affine 中理论可省的行计算比例约为

$$
\frac{372+100}{372+100+100}\left(1-\frac{1}{8.75}\right)=73.1\%.
$$

这里复用的是 mailbox 与 hidden 两类矩阵乘，time 分量仍按 state 计算；73.1% 是忽略 kernel 拆分、padding、launch、访存和 backward 的理想 FLOP 上界，不是 GRU 总耗时下降或可实现 speedup。作为量级参照，已保留的 gather-pointwise Compact GRU 局部 F+B 相对未融合 compact reference 实测只下降 23.2%。

`E/O=13.15×` 只表示 edge feature 展开压力，**不表示可以删除 13.15 倍采样边或 attention 计算**。最终 all-on 仍会把 edge feature 和 K/V 物化到 occurrence，因此该比值也不能直接解释 Fused Segment 的收益。

#### B. 三配置分阶段 ABBA

选择 WikiTalk、8 A40，每次 10 epoch、`--skip_eval`，正式计时关闭所有 probe。用三个配置把 producer 与去重后的执行链分开：

| 配置 | Batch producer | 去重后的 builder / consumers |
| --- | --- | --- |
| A：原直接去重链 | legacy Python/GPU compaction | legacy DGL Block/attention + 原 GRU 路径 |
| B：只升级执行 consumer | **仍为 legacy compaction** | Direct Block/segment plan + Fused Segment + Compact GRU |
| C：完整分层紧凑执行 | sampler-native compaction | 与 B 相同的 Direct Block/Fused Segment/Compact GRU |

执行两组独立的顺序抵消测试：

```text
A → B → B → A：回答 legacy 去重后是否仍有 builder/consumer 执行空间
B → C → C → B：回答 sampler-native producer 是否减少 post-sampling compaction
```

两组都保持相同图划分、seed、alpha/beta、采样与负样本。主要指标为 steady epoch-critical train time；同时核对每轮全局 local/remote node/edge 计数、shared count 和两组相邻配对的方向。如果要宣称采样结果逐项相同，还必须额外记录每个 batch 的 root、负样本、EID、timestamp/COO 哈希并验证两侧 bit-exact；只有计数相同只能证明 sampled workload counts 相同。

这组实验的解释边界是：

```text
A→B: same legacy producer; builder + compact consumers changed
B→C: same compact consumers; Batch producer changed
A→C: full MemShare Batch upgrade
```

若 A→B 稳定更快，只能支持“legacy compaction 后仍存在下游执行优化空间”；它同时改变 Direct Block、segment plan、attention 和 GRU，不能把差值归因于映射持久化或某一颗 kernel。若 B→C 也稳定更快，才支持 sampler-native producer 减少 post-sampling compaction 的主张。

现有证据中，B→C 的 WikiTalk ABBA 已观察到 steady critical 下降 4.01%；A→B 在最终统一源码下尚未做联合 ABBA。历史单组件消融给出 4.70%--6.33% 的 Direct Block+Fused Segment 和 2.26%--4.40% 的 Compact GRU 正信号，但不能把两者相加代替 A→B。

### 6.3 最终图表只做一张双面板

- **左图：Cardinality waterfall**——四数据集的 `N → S → U` 与 `E → O`，标注 median 和 P10--P90；
- **右图：Staged execution A/B**——A/B/C 三柱，另标注 A→B 的 builder/consumer 差值与 B→C 的 producer 差值。

图中央只保留一句话：

> Direct dedup reduces object fetch rows; batch-lifetime mappings enable downstream compact execution.

### 6.4 判定标准

以下条件同时满足，才把它作为主 Motivation：

- 至少 3/4 数据集的 global-batch median `S/U > 4×`，而不是由少数异常 batch 拉高均值；
- A→B 与 B→C 的两组相邻配对各自 steady critical 均至少下降 3%；
- OFF/ON 每轮全局 node/edge workload count 一致；若采集了 occurrence 哈希，则哈希也必须一致；
- steady shared-count 的绝对变化小于 1%，不足以解释主要收益。

若某一阶段的两组配对都在 1%--3%，只作为支持性结果，不进入 headline；小于 1% 或方向翻转时，只能说“存在表示冗余”，不能说该阶段是主要执行瓶颈。

## 7. 当前最稳妥的组会结论

> MemShare 原有 Batch 去重已经减少了对象读取与 temporal-state 存储；本次升级没有改变去重等价类，而是把 object/state/occurrence 关系定义为 batch-lifetime contract，并在训练路径把其构造前移到 sampler 边界。Direct Block、Compact GRU 与 Fused Segment Attention 因而能在各自正确粒度上复用或融合计算。当前四数据集 8-GPU 直接 A/B 显示稳态训练关键路径下降 7.97%--12.59%、整轮关键路径下降 5.69%--9.93%；已测 AP/AUC 最坏回退小于 0.5 pp。下一项最关键证据是四数据集的 Batch redundancy census，以及最终统一代码下 A→B 的 legacy-producer/compact-consumer ABBA。
