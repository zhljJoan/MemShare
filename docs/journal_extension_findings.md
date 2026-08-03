# MemShare 期刊扩展：阶段性实验发现与决策记录

> 更新时间：2026-07-28
> 当前实验分支：`driftshare-io-cache-benchmark`
> 本文档用于汇总当前代码上的实验事实、反例、无效方向和下一步低成本验证方案。它不是论文定稿，所有结论均按证据强度标注。
>
> **2026-07-28 口径修正：**旧 remote profiler 混用了 top-0 的 node home 与 top-\(K\) 的 edge route。此前记录的 0.8735/0.9093 remote ratio、remote coverage 和 drift oracle 数值不代表实际远程访问，现已撤销。第 4、5 节记录修正后的定义和结果。

## 1. 结论摘要

目前最重要的发现可以收敛为以下八点。

1. **现有 smooth aggregation 的平均精度收益是真实存在的。**
   在同一模型、同一时刻、同一 memory 状态下进行 raw/smooth 配对预测，WIKI 和 LASTFM 的 AP、AUC、accuracy 都有正向提升。

2. **smooth aggregation 不是对所有样本都有效。**
   WIKI 有 392 个样本被修正、290 个样本被破坏；LASTFM 有 3,513 个样本被修正、2,522 个样本被破坏。负作用存在，但小于总体收益。

3. **负作用主要是事件和上下文相关的，不是固定节点属性可以稳定解释的。**
   固定节点策略在时间外推测试中几乎没有超过 `all-smooth`；burstiness、remote ratio、staleness、方向一致性等标量的 AUC 大多接近 0.5。

4. **“高 burst 节点不适合 smooth”不能作为统一规则。**
   burst 能描述数据集和热点节点的时序结构，但 WIKI、LASTFM、Flight、WikiTalk 上的最优静态 burst 策略不一致，且独立训练结果受 seed 和训练轨迹影响明显。

5. **现有 drift oracle 不能用于判断动态热点是否值得实现。**
   它把 top-0 node home 与 top-\(K\) edge route 混在同一计数中，节点换分区也会被误计为远程访问。GDELT“接近 oracle”和 StackOverflow“存在较大上界”的旧结论均暂停使用。

6. **热点集合同时参与图划分和 edge route，是在线更新方案的主要系统风险。**
   直接改变共享节点集合会使 memory cache、历史事件分区和未来事件路由语义不一致。当前证据不足以支撑为此实现版本化在线迁移和增量重划分。

7. **StackOverflow 的一次有效双机测试中，关闭 IB 只增加约 0.84% 的 epoch 时间。**
   这次配置下网络传输不是明显瓶颈。GDELT 尚未得到有效的 IB/Socket 对照，因为运行在超大数据准备阶段被终止。

8. **当前值得继续验证的低成本方向，是把“路由热点”和“共享 memory 热点”解耦。**
   保持原有 top-\(K\) 节点参与图划分、feature cache 和 edge route，不在线改变路由；只在这个固定候选集合内部判断哪些节点值得复制并同步 memory。该方向目前只是候选假设，尚未被实验验证。

## 2. 证据等级

| 等级 | 定义 | 当前实验 |
| --- | --- | --- |
| A：强证据 | 同状态配对、定义正确的完整 trace 统计或完成的系统实测 | raw/smooth shadow；StackOverflow 双机测试 |
| B：中等证据 | 完整训练但 seed 较少，或机制消融已有重复 | WIKI/LASTFM 50 epoch；Filter source 三 seed |
| C：初步证据 | 训练较短、单 seed 或样本量很小 | 四数据集 2 epoch burst 策略；Flight/WikiTalk 20 epoch changed samples |
| D：无效/未完成 | 被中断、定义错误、饱和或无法公平归因 | 旧 remote/drift profiler；GDELT 双机 epoch；饱和 oracle |

论文正文只能把 A 级结果写成确定事实。B 级结果需要明确 seed 限制；C 级结果只适合决定下一步实验；D 级结果不能用于支持方法收益。

## 3. 当前方法语义

当前 MemShare 的 top-\(K\) 共享节点不是一个独立 cache，它同时影响：

- 图划分时的共享节点处理；
- 节点 feature 和 memory 的副本布局；
- 边界采样和跨分区访问；
- 未来事件的 edge route；
- shared memory 的同步与历史补偿。

当前 memory 链路为：

```text
显著性检查（SSIM threshold）
    -> Filter 预测历史增量
    -> gamma 调节历史分量
    -> local 最新 memory 与 remote/stale 历史状态平滑融合
```

因此，改变“哪些节点共享”会同时改变效率、训练轨迹和图路由，不能把它当成普通 cache replacement。

### 3.1 实验标签的准确含义

smooth policy 与同步过滤是两个独立维度：

| 标签 | 含义 |
| --- | --- |
| `none` | eligible shared node 都不应用历史 smooth aggregation |
| `all` | eligible shared node 全部应用 smooth aggregation |
| `low_burst_only` | 只对低 burst 分组应用 smooth |
| `mid_burst_only` | 只对中 burst 分组应用 smooth |
| `high_burst_only` | 只对高 burst 分组应用 smooth |
| `fullsync` | 不使用 SSIM=0.3 过滤，显著性阈值不减少同步 |
| `ssim03` | 使用 SSIM threshold=0.3 过滤 shared-memory 同步 |

因此：

- `none_fullsync` 才是“不做 smooth + 不做 SSIM 过滤”的标准全同步对照；
- `none_ssim03` 是“不做 smooth，但同步仍经过 SSIM=0.3 过滤”；
- `all_fullsync` 用于单独观察 smooth 在全同步状态下的影响；
- `all_ssim03` 是当前完整 MemShare 组合。

后文凡是比较独立训练模型，必须同时注明这两个维度。

## 4. 数据与分区结构

### 4.1 时序性质概况

以下时序统计来自 [summary_patterns.csv](../outputs/temporal_pattern/summary_patterns.csv)。GDELT 这一行只分析了前 10,000,000 个事件，不是完整训练流。该文件中的 `remote_ratio` 和 `remote_coverage` 使用了已经撤销的旧口径，本文不再引用；事件数、inter-arrival 和 burst 分类不受影响。

| 数据集 | 统计事件数 | 全体 burst P50 | 共享节点 burst P50 | 共享节点中 bursty 比例 |
| --- | ---: | ---: | ---: | ---: |
| WIKI | 110,232 | 0.3293 | 0.5129 | 0.9486 |
| LASTFM | 905,172 | 0.3609 | 0.6143 | 0.9319 |
| Flight | 1,349,001 | 0.6177 | 0.8826 | 1.0000 |
| WikiTalk | 5,483,197 | 0.4385 | 0.6030 | 0.8660 |
| StackOverflow | 44,447,934 | 0.5438 | 0.6534 | 0.9798 |
| GDELT 前 10M | 10,000,000 | 0.5779 | 0.9156 | 1.0000 |

按当前 inter-arrival 规则得到的节点类型占比：

| 数据集 | 全体 regular | 全体 Poisson-like | 全体 bursty | 全体样本不足 | 共享 regular | 共享 Poisson-like | 共享 bursty | 共享样本不足 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WIKI | 0.0118 | 0.1435 | 0.3208 | 0.5239 | 0 | 0.0444 | 0.9486 | 0.0070 |
| LASTFM | 0 | 0.1629 | 0.8226 | 0.0145 | 0 | 0.0681 | 0.9319 | 0 |
| Flight | 0.0088 | 0.1058 | 0.7527 | 0.1327 | 0 | 0 | 1.0000 | 0 |
| WikiTalk | 0.0095 | 0.0593 | 0.1535 | 0.7777 | 0.0090 | 0.0758 | 0.8660 | 0.0492 |
| StackOverflow | 0.0405 | 0.0729 | 0.5381 | 0.3485 | 0.0023 | 0.0055 | 0.9798 | 0.0124 |
| GDELT 前 10M | 0.0290 | 0.0702 | 0.6911 | 0.2097 | 0 | 0 | 1.0000 | 0 |

这里的 `regular` 只表示 inter-arrival dispersion 较低，并不等价于经过频谱检验确认的强周期。当前分类器没有在 Flight 共享热点中识别出独立的 regular/periodic 群体，而是将它们全部归为 bursty；若论文要声称 Flight 存在强周期节点，需要另做 autocorrelation/periodogram 检验。

可支持的结论：

- 当前高 degree/shared 节点通常比全体节点更 bursty。
- Flight 和 GDELT 的共享热点尤其集中在高 burst 区域。
- burstiness 是有意义的数据描述量，但不能据此决定是否应用 smooth。
- Flight 的训练事件数为 1,349,001，约为 WIKI 的 12.2 倍；当前 Flight 现象不能归因于训练事件数量不足。

### 4.2 纯 METIS 下的 Degree Hot 与 Cross-Partition Hot

该实验专门回答：

> 原始 degree top-\(K\) 热点是否也是朴素图划分后承担最多跨分区边的节点？

为了避免热点复制和特殊 edge route 反过来改变结果，实验只使用 plain METIS top-0 的节点分区 \(P(u)\)，不使用 top-\(K\) 的 `edge_output`。对原始边 \(e=(u,v)\) 定义：

\[
\operatorname{cut}(e)=\mathbb I[P(u)\ne P(v)]
\]

\[
D_u=\#\{e:e\ni u\},\qquad
X_u=\#\{e:e\ni u,\operatorname{cut}(e)=1\}
\]

\[
Q_u=X_u/D_u
\]

- \(D_u\)：degree；
- \(X_u\)：节点关联的跨分区边总数，对应绝对通信规模；
- \(Q_u\)：跨分区边比例，容易偏向低 degree 节点，不应单独作为共享收益排序。

Flight 使用 8 分区、1,349,001 条训练边，\(K=1,316\)：

| 指标 | 结果 |
| --- | ---: |
| METIS cut edges | 373,460 |
| Cut-edge ratio | 0.2768 |
| Degree 与 cross-count \(X_u\) 的 Spearman | 0.5531 |
| Degree 与 cross-ratio \(Q_u\) 的 Spearman | 0.2344 |
| 当前 SPEED 热点与 cross-count top-\(K\) 重合 | 0.5912 |
| Degree top-\(K\) 与 cross-count top-\(K\) 重合 | 0.5950 |
| 当前 SPEED 热点与 degree top-\(K\) 重合 | 0.9514 |
| 当前 SPEED 热点覆盖 cross-edge endpoint | 0.9054 |
| Degree top-\(K\) 覆盖 cross-edge endpoint | 0.9061 |
| Cross-count top-\(K\) 覆盖上限 | 0.9482 |
| 当前 SPEED 热点触及的 cut edge | 0.9836 |
| 当前 SPEED 热点与 cross-ratio top-\(K\) 重合 | 0.1353 |

解释：

1. Degree hot 与 cross-count hot 并不等价，集合重合约 59%。
2. 但跨分区流量高度集中在高 degree 节点，当前热点已覆盖约 90.5% 的 cross-edge endpoint，并触及约 98.4% 的 cut edge。
3. 直接选择 cross-count top-\(K\) 的覆盖上限为 94.8%，比 degree top-\(K\) 高约 4.2 个百分点。
4. Cross-ratio 排名与当前热点只有约 13.5% 重合，不适合作为绝对通信收益的选择标准。
5. 在全量 1,927,145 条边上，当前 SPEED 热点与 degree top-\(K\) 重合 99.92%，说明当前热点集合基本就是全图 degree top-\(K\)；训练流上的 95.14% 差异来自数据切分。

该结果支持“degree 是强但不完全准确的跨分区热点近似”，不能支持复杂动态迁移。当前 plain METIS 分区由全图生成；若论文要求严格避免 validation/test 拓扑信息，需要补充 train-only METIS 重跑。

### 4.3 Flight 的 Locality 与负载均衡诊断

这一对照与 4.2 的研究问题不同。它使用各自 top-0.1 分区的当前 `output{p}.txt` 和 `edge_output{p}.txt`，检查已经完成热点复制和 edge route 后，事件直接 src/dst 是否位于执行分区：

\[
r_u(e)=\mathbb I[u\notin L_{p(e)}]
\]

\[
\mathrm{DirectEndpointRemoteRatio}
=\frac{\sum_e(r_{src}(e)+r_{dst}(e))}{2|E|}
\]

| Flight 8 分区，top-0.1 | StarryGL/SPEED | METIS + MemShare route |
| --- | ---: | ---: |
| 训练事件 | 1,349,001 | 1,349,001 |
| 远程 src 端点 | 34,163 | 0 |
| 远程 dst 端点 | 34,995 | 4,550 |
| 总远程直接端点 | 69,158 | 4,550 |
| 直接端点 remote ratio | 0.02563 | 0.00169 |
| One-hot 边 remote ratio | 0 | 0 |
| Hot-hot 边 remote ratio | 0 | 0 |
| Cold-cold 边端点 remote ratio | 0.3031 | 0.0199 |
| 训练边负载 max/min | 1.00025 | 1.48361 |
| 训练边负载 CV | 0.00008 | 0.10987 |

当前正确解释是：

- METIS 显著降低了 cold-cold endpoint cut，但没有平衡实际训练事件；
- SPEED 将 8 个分区的训练边都控制在约 168.6K，代价是更多 cold-cold endpoint 不在执行分区；
- 这是 locality 与事件负载均衡的权衡，不是 memory sharing 方法优劣；
- 这里只统计直接 src/dst，不包含 neighbor sampling、边特征和 shared-memory synchronization，因此不能代替端到端通信计量。

旧 [Flight METIS summary](../outputs/hot_memory_motivation_metis_flight/Flight_8parts_top0.1_train.summary.json) 中的 0.9093，以及旧 [Flight hot-memory summary](../outputs/hot_memory_motivation/Flight_8parts_top0.1_train.summary.json) 中的 0.8735，均来自错误的 top0-home mismatch 口径，只保留作审计记录。

## 5. Dynamic Drift Oracle：旧结果撤销

### 5.1 旧定义

旧 profiler 将训练流划分成 50 个等事件窗口，然后：

1. 从 top-0 分区读取固定 `home(u)`；
2. 从 top-\(K\) 分区读取事件分区 \(p(e)\)；
3. 把 `home(u) != p(e)` 计为节点 \(u\) 的 counterfactual remote access；
4. 每个窗口选择该计数最大的 \(K\) 个节点作为 hindsight oracle。

形式上：

\[
R_u^{(w)}
=\sum_{e\in w,u\in e}
\mathbb I[home_{top0}(u)\ne p_{topK}(e)]
\]

\[
H_w^{oracle}=\operatorname{TopK}_u R_u^{(w)}
\]

### 5.2 为什么旧结果无效

top-0 与 top-\(K\) 是两次不同划分。特别是共享热点会改变 edge route，冷节点也可能在两次划分之间改变 owner。因此：

\[
home_{top0}(u)\ne p_{topK}(e)
\]

既可能表示真实 counterfactual remote access，也可能只是两个划分不一致。旧 profiler 无法区分这两种情况。

因此，下列旧结论全部暂停使用：

- GDELT 静态集合接近动态 oracle；
- StackOverflow top-0.02 存在约 32% 至 34% 的 oracle coverage 上界；
- 不同数据集的 oracle Jaccard/churn 差异；
- 基于这些数值否定或支持动态热点演变。

旧结果保留在 [driftshare_profiler](../outputs/driftshare_profiler/) 目录，仅用于追踪历史，不得进入论文表格。

### 5.3 正确重做方式

需要明确区分两个实验：

1. **纯结构热点实验**：使用固定 plain METIS 节点划分，按 cut-edge count 比较 degree hot 与 cross-partition hot，即第 4.2 节。
2. **固定路由的 memory oracle**：保持同一套 edge route 和节点 owner，只改变哪些候选节点复制 memory；shared node 的 counterfactual home 必须由 partitioner 明确输出，不能借用另一套 top0 划分。

如果共享集合变化会同时改变 edge route，那么真正的 dynamic oracle 必须对每个候选集合重新执行 partition/route，不能使用一次固定 trace 近似。

**阶段决策：暂停引用动态热点 oracle。动态热点路线目前是“证据未建立”，而不是“已证明没有收益”。**

## 6. Smooth Aggregation 的直接精度作用

### 6.1 同状态配对实验

这是当前最可信的精度归因实验。`raw` 与 `smooth` 来自：

- 同一个训练完成的模型；
- 同一个评估 batch；
- 同一个更新前 memory/mailbox 状态；
- shadow 分支不修改 memory 或 mailbox。

| 数据集 | 样本 | smooth 修正 | smooth 破坏 | 净修正 | AP：raw -> smooth | AUC：raw -> smooth | Accuracy：raw -> smooth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WIKI | 47,242 | 392 | 290 | +102 | 0.960674 -> 0.961822 | 0.956553 -> 0.957931 | 0.884510 -> 0.886669 |
| LASTFM | 387,932 | 3,513 | 2,522 | +991 | 0.885741 -> 0.887231 | 0.894267 -> 0.895631 | 0.770913 -> 0.773468 |

对应增量：

| 数据集 | \(\Delta\)AP | \(\Delta\)AUC | \(\Delta\)Accuracy |
| --- | ---: | ---: | ---: |
| WIKI | +0.001148 | +0.001379 | +0.002159 |
| LASTFM | +0.001490 | +0.001364 | +0.002555 |

标签方向存在不对称：

- WIKI：负样本净修正 +5，正样本净修正 +97。
- LASTFM：负样本净修正 -149，正样本净修正 +1,140。

因此，smooth 对正链接排序更有帮助，但可能增加一部分 false positive。

如果使用事后 oracle，只在 smooth 会出错时切回 raw：

- WIKI 最多比 all-smooth 再修正 290/47,242，即约 0.61 个 accuracy 百分点。
- LASTFM 最多再修正 2,522/387,932，即约 0.65 个百分点。

这是 hindsight 上界，不代表已有特征能够实现该路由。

详细报告：[memory_aggregation_findings.md](../outputs/memory_shadow_analysis/memory_aggregation_findings.md)。

### 6.2 长期训练效应与即时效应必须区分

独立训练的 `none`、`all`、SSIM/full-sync 模型会沿不同 memory 轨迹更新参数。两个模型预测不同，不能全部归因为当前样本上的 smooth 操作。

独立训练模型的 changed-sample 数量远高于配对 shadow：

| 数据集 | epoch/seed | 同步设置 | 公共样本 | `all` 相比 `none` 改善 | 退化 |
| --- | --- | --- | ---: | ---: | ---: |
| WIKI | 50/6773 | full sync | 47,242 | 1,797 | 1,899 |
| WIKI | 50/6773 | SSIM=0.3 | 47,242 | 1,818 | 1,767 |
| LASTFM | 50/6773 | full sync | 387,921 | 38,702 | 25,290 |
| LASTFM | 50/6773 | SSIM=0.3 | 387,921 | 42,111 | 22,713 |
| LASTFM | 50/6774 | full sync | 387,921 | 53,915 | 25,021 |
| LASTFM | 50/6774 | SSIM=0.3 | 387,921 | 42,504 | 26,421 |
| Flight | 20/6773 | full sync | 577,664 | 13,978 | 11,649 |
| Flight | 20/6773 | SSIM=0.3 | 577,664 | 15,737 | 11,698 |
| WikiTalk | 20/6773 | full sync | 2,925,647 | 68,546 | 57,812 |
| WikiTalk | 20/6773 | SSIM=0.3 | 2,925,647 | 77,223 | 53,675 |

这些数字反映“训练轨迹 + 同步策略 + smooth”的联合效果，不能用来训练一个即时 gate。即时机制判断应优先使用 6.1 的 same-state shadow。

在 LASTFM 中，固定 smooth 后再比较 SSIM 与 full sync，seed 6773 为 22,421 个样本改善、19,789 个退化；seed 6774 则为 20,053 个改善、24,948 个退化。这进一步说明独立训练的逐样本差异受 seed 和轨迹影响，不能只凭 changed-sample 净数判断 SSIM 的因果效果。

总体决策仍是保留 `all_ssim03` 作为当前默认方法。它已经控制同步量，并在大多数完整/较大数据集实验中保持最稳定的总体表现；现有静态 gate 尚不足以替代它。

## 7. 节点性质与坏例分析

### 7.1 固定节点标签不稳定

只统计恰好一个 shared endpoint 的样本，并按节点历史表现分类：

| 数据集 | beneficial 节点 | harmful 节点 | mixed 节点 | unchanged 节点 | one-shared 样本 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WIKI | 114 | 83 | 40 | 584 | 21,149 |
| LASTFM | 13 | 2 | 166 | 11 | 209,530 |

LASTFM 的 181 个受影响节点中有 166 个是 mixed，且 mixed 节点覆盖约 99.6% 的退化样本。同一个节点在不同事件上既可能受益，也可能受损。

### 7.2 时间外推节点策略

用测试流前半段学习“该节点用 raw 还是 smooth”，再作用到后半段：

| 数据集 | raw accuracy | all-smooth accuracy | 学得节点策略 | 相比 all-smooth | hindsight 节点 oracle | 前后半段符号一致率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WIKI | 0.873747 | 0.874693 | 0.874787 | +1 个样本 | 0.881691 | 0.5526 |
| LASTFM | 0.827347 | 0.830172 | 0.830058 | -12 个样本 | 0.831862 | 0.7523 |

固定 per-node gate 没有稳定超过 all-smooth。WIKI 的 +1 个样本没有研究意义，LASTFM 反而退化。

### 7.3 静态与运行时信号的可预测性

评估特征包括：

- burstiness；
- remote ratio；
- stale age；
- Filter support/count；
- memory 更新与预测增量的方向一致性；
- memory novelty；
- innovation/trend norm；
- raw/smooth prediction residual。

时间 holdout 的分类结果：

| 数据集 | 目标 | Logistic AUC | 小树 AUC |
| --- | --- | ---: | ---: |
| WIKI | smooth margin 方向 | 0.497 | 0.533 |
| WIKI | correctness flip | 0.510 | 0.510 |
| LASTFM | smooth margin 方向 | 0.563 | 0.567 |
| LASTFM | correctness flip | 0.512 | 0.546 |

单信号的最好结果也较弱：

- WIKI：memory novelty AUC 约 0.582，但只有 375 个 changed endpoint 样本。
- LASTFM：remote ratio AUC 约 0.569，burstiness 约 0.551。

这里的 AUC=0.5 表示模型在“改善还是退化”的二分类上与随机排序相当，不是 TGNN 链接预测 AUC 只有 50%。

结果文件：

- [WIKI gate predictability](../outputs/memory_shadow_analysis/WIKI_memory_gate_predictability_seed6773.json)
- [LASTFM gate predictability](../outputs/memory_shadow_analysis/LASTFM_memory_gate_predictability_seed6773.json)

## 8. Burst/周期/泊松分布实验

### 8.1 50 epoch，单 seed

下表中的 policy 只控制哪些节点应用 smooth，SSIM 设置保持实验脚本对应的基线。按 validation AUC 选择 epoch。

| 数据集 | policy | Test AP | Test AUC | 相比 none 的 \(\Delta\)AUC |
| --- | --- | ---: | ---: | ---: |
| WIKI | none | 0.963284 | 0.958952 | 0 |
| WIKI | all | 0.962861 | 0.958058 | -0.000894 |
| WIKI | low-burst only | 0.964819 | 0.960024 | +0.001072 |
| WIKI | mid-burst only | 0.963966 | 0.959377 | +0.000425 |
| WIKI | high-burst only | 0.961871 | 0.957772 | -0.001180 |
| LASTFM | none | 0.855381 | 0.862400 | 0 |
| LASTFM | all | 0.878139 | 0.889098 | +0.026698 |
| LASTFM | low-burst only | 0.860527 | 0.866072 | +0.003672 |
| LASTFM | mid-burst only | 0.856732 | 0.866848 | +0.004448 |
| LASTFM | high-burst only | 0.859535 | 0.865052 | +0.002652 |

WIKI 单 seed 看起来偏向 low-burst only，但 LASTFM 明显由 all-smooth 获益。结合节点外推和配对诊断，这不能推出统一的 burst gate。

原始汇总：[wiki_lastfm_50ep_smooth_summary.csv](../outputs/pattern_ablation_logs/wiki_lastfm_50ep_smooth_summary.csv)。

### 8.2 四数据集 2 epoch 初筛

| 数据集 | none AUC | all AUC | 该轮最好静态 policy | 最好 AUC |
| --- | ---: | ---: | --- | ---: |
| WIKI | 0.877016 | 0.876365 | mid-burst only | 0.883111 |
| LASTFM | 0.786404 | 0.805926 | all | 0.805926 |
| Flight | 0.942558 | 0.945814 | high-burst only | 0.945974 |
| WikiTalk | 0.962339 | 0.964205 | all | 0.964205 |

这些结果只有 2 epoch、单 seed。Flight 中 99.87% 的样本被归为 high-burst，`high-burst only` 实际上近似 `all`；WikiTalk 的高 burst 策略反而低于 all。它们只说明数据分布不同，不能证明可泛化策略。

原始汇总：[burst_strength_summary.csv](../outputs/pattern_ablation_logs/burst_strength_summary.csv)。

## 9. Gamma 与机制消融

### 9.1 Gamma 的真实作用

当前实现是：

```text
lambda_history = 1 - sigmoid(gamma)
memory_out = current + lambda_history * gate * (prediction - current)
```

Gamma 是全局历史分量 mixing logit，不是 on/off gate。

| 数据集 | raw gamma 范围 | 最终 gamma | 最终 history weight |
| --- | ---: | ---: | ---: |
| WIKI | 0.4347 - 0.4984 | 0.4856 | 0.3809 |
| LASTFM | 0.4422 - 0.5116 | 0.4910 | 约 0.3797 |

虽然 raw gamma 一直变化，但经过 sigmoid 后，有效 history weight 只变化约 0.015 至 0.016。问题不是数值不收敛，而是单个 global scalar 无法表达事件条件差异。

### 9.2 Batch min-max normalization

WIKI seed 6773：

| 模式 | Global AP | Global AUC | Accuracy |
| --- | ---: | ---: | ---: |
| 原始 batch min-max | 0.961822 | 0.957931 | 0.886669 |
| 不做 normalization | 0.961507 | 0.957241 | 0.889039 |

去掉 normalization 虽提高 accuracy，但 AP 和 AUC 都下降。对链接预测主指标不利，不应作为方法修改。

### 9.3 Filter 观测源

旧实现把已经平滑过的增量再次送入 Filter，存在自反馈：

```text
old = fused_memory - stale_memory
    = a * raw_increment + (1-a) * predicted_increment
```

消融版本改为：

```text
new = current_memory - stale_memory
    = raw_increment
```

WIKI 三 seed：

| seed | \(\Delta\)AP：current - old | \(\Delta\)AUC |
| ---: | ---: | ---: |
| 6773 | +0.001295 | +0.000496 |
| 6774 | -0.001558 | -0.001607 |
| 6775 | -0.002449 | -0.002537 |
| 配对均值 | -0.000904 | -0.001216 |

LASTFM seed 6773：

- AP +0.001004；
- AUC -0.000125；
- accuracy +0.003385。

该修改适合作为机制解释消融，但没有形成稳定改进，不应替换当前默认实现。

## 10. 双机通信与数据加载

### 10.1 测试配置

入口：[run_two_node_epoch_benchmark.sh](../tools/run_two_node_epoch_benchmark.sh)。

- 服务器：gpu07 + gpu06；
- 每台 4 GPU，共 8 rank；
- 模型：`TGN_large`；
- batch size：3,000；
- memory dim：100，GRU，mailbox size=1，async=true；
- partition：`ours`；
- sample：`boundery_recent_decay`，probability=0.1；
- memory mode：`historical`；
- SSIM threshold：0.3；
- 共享比例：StackOverflow top-0.1；
- epoch：1，跳过 validation/test；
- feature storage：当前入口显式使用 CPU；
- node memory、timestamp 和 mailbox：`SharedMailBox` 在各 rank GPU 上分配。

因此，StackOverflow 的 node feature 并不是预先全部放在 GPU；memory/mailbox 是 GPU resident。

### 10.2 StackOverflow：IB 与 Socket

| 传输 | Epoch train time | Forward/Rank | Memory commit/Rank | Batch wait/Rank | Backend memory fetch/Rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| IB，`NCCL_IB_DISABLE=0` | 97.31 s | 67.85-68.21 s | 33.65-35.65 s | 2.94-2.99 s | 2.71-2.78 s |
| Socket，`NCCL_IB_DISABLE=1` | 98.13 s | 67.83-68.16 s | 33.72-33.98 s | 2.97-3.00 s | 2.75-2.79 s |

Socket 比 IB 慢 0.82 s，即约 0.84%。该结论只适用于这次 StackOverflow top-0.1、8 rank、SSIM=0.3 的运行。

限制：

- `memory_commit`、forward 等流水阶段可能重叠，不能相加得到 epoch 时间。
- 当前 `time_memory_sync=0` 计时器没有覆盖异步 shared-memory 同步，不能据此声称同步成本为零。
- 两次运行的 shared communication counter 略有差异，不能把 counter 差异解释为传输介质效果。

日志：

- [StackOverflow IB](../outputs/two_node_epoch_cache_20260724/StackOverflow_top0.1_ibdisable0_rank0.log)
- [StackOverflow Socket](../outputs/two_node_epoch_cache_20260724/StackOverflow_top0.1_ibdisable1_rank0.log)

### 10.3 GDELT：尚无有效通信对照

GDELT 完整训练流约 128,563,062 个事件；数据准备日志中原始边规模为 191,290,882。双机运行在数据/分区准备阶段输出：

```text
tot edge 191290882 same edge 9447265
```

随后被 SIGTERM 终止，没有进入并完成一个 epoch。因此：

- 没有有效的 GDELT IB epoch 时间；
- 没有有效的 GDELT Socket epoch 时间；
- 不能从现有日志比较 GDELT 的网络开销。

当前加载路径已增加两级复用：

1. 优先读取 `edges_cache/{src,dst,time,ext_roll}.npy`，使用 mmap；
2. 其次读取 `edges_cache.pt`；
3. 两者都不存在时才回退到 pandas 读取原始 `edges.csv`。

同时移除了只用于打印统计、但会对超大边表执行全局 `torch.unique` 的步骤。`edges.csv` 的首次解析是当前数据构造所需，但每次训练重复 pandas parse 和全局 unique 不是 epoch 的必要成本，应通过预处理 cache 摊销。

## 11. 已排除或暂不推进的方向

### 11.1 在线动态热点、租约与版本化迁移

暂不推进，原因是：

- 旧 drift oracle 的 remote 定义已经撤销，尚无可信收益上界；
- 共享集合变化会改变 partition 和 edge route，不能在一次固定 trace 上公平比较；
- 即使重做 oracle，仍需计入迁移、同步、版本一致性和重路由成本；
- shared set 影响 edge route，不是局部 memory cache 更新；
- 开发成本和一致性风险明显高于当前证据。

### 11.2 固定 burst/周期/泊松 gate

暂不推进，原因是：

- 数据集之间最优规则相反；
- 节点在不同事件上经常 mixed；
- 时间外推策略没有超过 all-smooth；
- 静态/运行时信号的 holdout AUC 接近随机。

### 11.3 把 gamma 直接改成普通门

暂不推进。当前 gamma 是历史补偿强度，不是通信门。简单 sigmoid gate 仍缺少可预测信号，而且可能破坏已验证的 all-smooth 平均收益。

### 11.4 修改 Filter 观测源或移除 normalization

两者都已做消融，但没有同时稳定改善 AP 和 AUC。保留为论文机制分析，不作为新方法。

### 11.5 额外通信 coalescing 或预算调度

撤回该方向。SSIM 本身已经依据 memory 变化控制同步量，再增加周期聚合或独立通信预算会与现有机制职责重叠，也可能引入额外 stale。后续不应把它包装成新的主要贡献。

## 12. 当前候选期刊方向：Route-Hot 与 Memory-Hot 解耦

### 12.1 核心想法

固定原有路由集合：

\[
H_{\mathrm{route}} = \operatorname{TopKDegree}(V)
\]

它继续负责：

- StarryGL 图划分；
- edge route；
- feature replication/cache；
- 历史边和未来边的一致路由。

只在 \(H_{\mathrm{route}}\) 内选择真正复制 memory 的集合：

\[
H_{\mathrm{memory}} \subseteq H_{\mathrm{route}}
\]

节点有两种模式：

- `FullShare`：feature + memory/mailbox 副本，保留 SSIM 和 smooth aggregation；
- `FeatureOnly`：仍按共享热点参与 partition/route 和 feature cache，但 memory 由 home partition 管理。

这样不改变 shared-node 的分区身份，不迁移历史边，也不在线更新 edge route。

### 12.2 离线选择目标

对每个 \(u \in H_{\mathrm{route}}\) 统计：

\[
C_{\mathrm{share}}(u)
= N_{\mathrm{sig}}(u)(m-1)B_{\mathrm{memory}}
\]

\[
C_{\mathrm{home}}(u)
= N_{\mathrm{remote\_read}}(u)B_{\mathrm{read}}
+ N_{\mathrm{remote\_update}}(u)B_{\mathrm{update}}
\]

\[
U(u) = C_{\mathrm{home}}(u) - C_{\mathrm{share}}(u)
\]

- \(U(u)>0\)：复制 memory 预计更省通信；
- \(U(u)\le 0\)：保留 route/feature 热点身份，但不复制 memory。

进一步可把 accuracy 风险作为约束，而不是直接混入通信分数：

\[
\Delta\mathrm{AUC}(H_{\mathrm{memory}})
\ge -\epsilon
\]

### 12.3 为什么这个方向比动态热点更合适

- 不改变原有 top-\(K\) degree 划分证明；
- 不改变 edge route；
- 不需要版本化在线迁移；
- 不移动历史边；
- 保留 SSIM 控制同步量；
- 只增加 shared memory 的细粒度选择；
- 可以先离线 profiler，再决定是否改训练逻辑。

### 12.4 当前证据状态

该方案**尚未验证**。纯 METIS 结果只说明 degree top-\(K\) 已覆盖 Flight 约 90.6% 的 cross-edge endpoint，而 cross-count top-\(K\) 可达到约 94.8%；这表明存在有限细化空间，但不能证明 memory subset 一定有端到端收益。

必须先回答：

1. 当前 top-\(K\) 内有多少节点的 \(C_{\mathrm{share}}>C_{\mathrm{home}}\)？
2. 这些节点贡献了多少同步字节，但只避免了多少 remote memory 访问？
3. 移除其 memory replica 后，GPU memory、总通信和 epoch 时间上界是多少？
4. `FeatureOnly` 对 smooth aggregation 精度收益的影响是否可控？

## 13. 下一步最小实验与止损条件

### 13.1 第一步：只做离线 profiler

数据集优先级：

1. WIKI、LASTFM：快速验证逐节点 cost 分布；
2. Flight：验证强周期/高 burst 热点；
3. StackOverflow：验证大规模系统收益；
4. GDELT：只在 cache 加载稳定后运行。

每个 shared node 输出：

```text
node_id
degree
remote_read_count / bytes
remote_update_count / bytes
significant_update_count
SSIM-filtered sync count / bytes
partition span
burstiness / temporal class
estimated C_share
estimated C_home
estimated utility
```

这里的 `remote_read_count` 必须来自同一套 partition/route：

- non-shared node 的 home 从当前 `output{p}.txt` 唯一 owner 得到；
- shared node 的 counterfactual home 由 partitioner 显式记录；
- 不允许再从另一套 top0 划分借用 home；
- direct endpoint、sampled neighbor、memory sync 三类通信分别统计。

### 13.2 Oracle 对照

离线比较：

- `AllMemoryShare`：当前 MemShare；
- `TopDegreeMemory`：在 route-hot 内按 degree 选择 memory；
- `TopRemoteMemory`：按 remote memory access 选择；
- `CostAwareMemory`：按 \(U(u)\) 选择；
- `MemoryOracle`：按真实 trace 的最小通信成本选择。

共享 memory 预算可以先固定为 \(q|H_{\mathrm{route}}|\)，取：

\[
q \in \{0.25, 0.5, 0.75, 1.0\}
\]

### 13.3 Go/No-Go 条件

只有同时满足以下条件，才实现 `FeatureOnly`：

- 至少两个数据集上，cost-aware oracle 相比 all-memory-share 降低预计总通信 10% 以上；
- StackOverflow 上预计 epoch 关键路径收益至少 5%，或显著降低 GPU memory；
- 被移出 memory-share 的节点不是 smooth 精度收益的主要来源；
- 选择结果在 seed/窗口之间足够稳定，不需要在线迁移。

若不满足，应停止 memory 节点细分方向。届时期刊扩展更适合定位为：

- 更完整的 measurement study；
- memory smooth 的精度机制与反例分析；
- 多数据集、多模型和大规模系统评估；
- 对现有方法的工程优化与可复现性增强。

## 14. 当前论文叙事建议

目前可以安全写入论文的新增叙事是：

1. **Memory sharing 不只是效率优化，也改变 temporal memory 的统计质量。**
2. **平滑历史补偿在平均意义上提高链接预测，但存在少量上下文相关的坏例。**
3. **简单 degree、burst 或固定节点类型不能解释这些坏例。**
4. **纯 METIS 上 degree hot 与 cross-count hot 只有约 59% 重合，但 degree top-\(K\) 已覆盖约 90.6% 的跨分区端点流量。**
5. **在保持 partition/route 不变的前提下细化 memory sharing，是比在线重划分更低风险的扩展问题。**

当前不能写成论文结论的内容：

- dynamic hotspot 一定比静态 top-\(K\) 好；
- burst 节点应该关闭 smooth；
- GRU 能稳定识别长周期节点；
- gamma 不收敛导致精度问题；
- IB 对所有数据集都没有作用；
- GDELT 已完成双机通信测试；
- cost-aware memory subset 已经带来性能收益。

## 15. 主要产物索引

| 内容 | 产物 |
| --- | --- |
| Same-state raw/smooth 总报告 | [memory_aggregation_findings.md](../outputs/memory_shadow_analysis/memory_aggregation_findings.md) |
| WIKI paired summary | [WIKI_memory_shadow_50ep_seed6773.summary.json](../outputs/memory_shadow_analysis/WIKI_memory_shadow_50ep_seed6773.summary.json) |
| LASTFM paired summary | [LASTFM_memory_shadow_50ep_seed6773.summary.json](../outputs/memory_shadow_analysis/LASTFM_memory_shadow_50ep_seed6773.summary.json) |
| WIKI gate predictability | [WIKI_memory_gate_predictability_seed6773.json](../outputs/memory_shadow_analysis/WIKI_memory_gate_predictability_seed6773.json) |
| LASTFM gate predictability | [LASTFM_memory_gate_predictability_seed6773.json](../outputs/memory_shadow_analysis/LASTFM_memory_gate_predictability_seed6773.json) |
| 时序分布总表 | [summary_patterns.csv](../outputs/temporal_pattern/summary_patterns.csv) |
| 50 epoch burst policy | [wiki_lastfm_50ep_smooth_summary.csv](../outputs/pattern_ablation_logs/wiki_lastfm_50ep_smooth_summary.csv) |
| 四数据集短训 policy | [burst_strength_summary.csv](../outputs/pattern_ablation_logs/burst_strength_summary.csv) |
| 旧 Dynamic oracle，口径已撤销 | [driftshare_profiler](../outputs/driftshare_profiler/) |
| 旧 Hot-memory profiler，remote 字段已撤销 | [hot_memory_motivation](../outputs/hot_memory_motivation/) |
| 旧 Flight METIS remote 对照，口径已撤销 | [hot_memory_motivation_metis_flight](../outputs/hot_memory_motivation_metis_flight/) |
| WIKI Filter source 消融 | [WIKI_filtercurrent_vs_baseline_seed6773.summary.txt](../outputs/memory_mechanism_analysis/WIKI_filtercurrent_vs_baseline_seed6773.summary.txt) |
| LASTFM Filter source 消融 | [LASTFM_filtercurrent_vs_baseline_seed6773.summary.txt](../outputs/memory_mechanism_analysis/LASTFM_filtercurrent_vs_baseline_seed6773.summary.txt) |
| Flight/WikiTalk changed samples | [flight_wikitalk_sync_smooth_20ep_logs](../outputs/flight_wikitalk_sync_smooth_20ep_logs/) |
| StackOverflow 双机日志 | [two_node_epoch_cache_20260724](../outputs/two_node_epoch_cache_20260724/) |
| 旧离线漂移 profiler，待修复 home 定义 | [driftshare_offline_profiler.py](../tools/driftshare_offline_profiler.py) |
| 时序性质 profiler，remote 字段待修复 | [temporal_pattern_profiler.py](../tools/temporal_pattern_profiler.py) |
| 旧 Hot-memory profiler，待修复 home 定义 | [hot_memory_motivation_profiler.py](../tools/hot_memory_motivation_profiler.py) |
| 双机测试脚本 | [run_two_node_epoch_benchmark.sh](../tools/run_two_node_epoch_benchmark.sh) |
