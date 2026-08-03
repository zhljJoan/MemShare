# MemShare 期刊扩展阶段性汇报

更新时间：2026-08-03

当前分支：`core-memshare`

汇报范围：固定图划分、固定热点集合和固定 ownership 下的采样与 memory
优化。

详细依据：

- [保留方法与备选方向详述](journal_extension_effective_methods.md)
- [优化点、实验结果与推进决策](journal_extension_optimization_review.md)
- [阶段性调研发现](journal_extension_findings.md)
- [CoRe-MemShare 实验记录](core_memshare_experiments.md)

## 1. 汇报结论

当前结果支持两个层级的工作安排：

1. **优先验证对象：coordinated boundary-decay**
   - 双机八卡三 seed 完整训练中，远程节点平均下降 12.90%、远程边平均下降
     8.89%、batch 时间平均下降 3.69%，12/12 个 paired seed 均加速；
   - 四数据集各 3 次 100-batch 稳定性复测中，训练窗口网卡实测字节平均下降
     8.78%、mean batch 平均下降 3.04%，但 p95 仅平均下降 1.16%；
   - 大图单 epoch 中，StackOverflow 耗时 -1.60%，GDELT +0.95%；远程节点
     仍分别下降 11.82% 和 19.20%，说明通信对象收益尚未稳定转化为加速；
   - 每条 occurrence 仍使用自身的原采样概率，但组内联合分布和完整多层 MFG
     分布发生变化，因此不是完全无损采样；
   - 三 seed test AP 在 12 个配对中 8 次为负、4 次为正，四数据集平均
     -0.0903 个百分点；精度均值接近零来自正负抵消，收敛性也不稳定；
   - 下一步先做 group size 与 layer-restricted coordination，寻找更弱相关的配置，
     同时定位 GDELT 中 AllToAll route/launch 上升。
2. **备选方向：burst-aware smooth**
   - WIKI/WikiTalk 中观察到 high-burst 条件下的潜在劣势；
   - LASTFM 明显依赖 all-smooth，直接关闭某个 burst 分组会损失已有正收益；
   - 当前只保留为 bad-case 驱动的备选方向，不进入 coordinated 八卡对照。

默认 `all-smooth + SSIM=0.3` 是实验 baseline，不作为新的期刊技术点。

目前尚无新增方案同时证明：

- 双机八卡端到端训练加速；
- 完整多 seed 精度不退化；
- 收敛速度和训练稳定性不受影响。

## 2. 问题边界

本轮工作不修改：

- 图划分 $\pi$；
- route-hot/shared node 集合；
- node/edge ownership；
- 模型结构和 attention；
- 默认 memory 更新与 SSIM 同步链路。

动态变化的只有采样随机数之间的相关性。目标是在保持原 boundary-decay 单条
occurrence 概率的前提下，让多个 root 对同一远程对象的请求更容易复用。

```text
固定 partition / route-hot / ownership
                    │
                    ▼
原 boundary-decay 概率
                    │
                    ▼
group 内共享 keyed random
                    │
                    ▼
occurrence 按各自概率保留
                    │
                    ▼
MFG 去重并生成 dist_nid / dist_eid
                    │
                    ▼
原异步 node/edge fetch
```

## 3. 当前方向总览

| 方向 | 已观察到的结果 | 当前定位 |
| --- | --- | --- |
| Coordinated boundary-decay | 双机八卡三次系统复测中训练窗口字节平均 -8.78%、mean batch -3.04%，但 p95 仅 -1.16%；三 seed 完整训练 batch -3.69%、test AP -0.0903 个百分点且 8/12 为负；大图 StackOverflow/GDELT 分别 -1.60%/+0.95% | 继续验证 group size/分层配置并定位大图执行层开销 |
| Burst-aware smooth | WIKI 部分实验 low/mid 优于 all，high-only 有劣势；LASTFM all 最好 | 备选，不进入当前组合 |
| Residual-aware sampling | batch 时间 +50.08% | 暂停当前实现 |
| Probability-aware edge cache | 完整实现 batch 时间 +200.4% | 暂停 |
| Communication-aware hard negative | batch 时间 +238.15%，ESS 仅 3.79%–12.22% | 停止当前路线 |
| Static/Dynamic 分阶段 fetch | 尚未完成独立实现 | 多机关键路径满足条件后再测 |

## 4. Coordinated Boundary-Decay 方法

### 4.1 原采样概率

对 root $r$ 在时刻 $t$ 的远程候选邻居 $i=(v,\tau_i)$，时间衰减权重为：

$$
w_{r,i}
=
\exp
\left(
\frac{\tau_i-t}{\Delta_r}
\right).
$$

远程候选数为 $n_r^R$ 时，原 boundary-decay 概率为：

$$
p_{r,i}
=
\operatorname{clip}
\left(
\beta
\frac{w_{r,i}}
{\sum_{j\in N_r^R}w_{r,j}}
n_r^R,
1
\right).
$$

当前实验设置 $\beta=0.1$。coordinated 版本不修改 $p_{r,i}$ 的计算。

### 4.2 小组共享随机数

将采样输入位置相邻的 root 划为大小为 $G$ 的小组：

$$
g(r)
=
\left\lfloor
\frac{\operatorname{rootPosition}(r)}{G}
\right\rfloor.
$$

同一 remote node、group、batch 和 layer 使用同一个 keyed uniform：

$$
U_{v,g,b,l}
=
\operatorname{UniformHash}
\left(
v,g,b,l,\operatorname{seed}
\right).
$$

每条 occurrence 仍按自己的概率决定是否进入 MFG：

$$
z_{r,i}
=
\mathbb I
\left[
U_{v,g(r),b,l}<p_{r,i}
\right].
$$

当前实现使用 SplitMix64，key 为：

```text
(remote_node_id, root_group, batch_id, layer, seed)
```

当前主要配置：

```text
sample_type=boundery_recent_decay_coordinated
probability=0.1
core_group_size=8
core_freeze_adaptive_parameters=true
```

## 5. 为什么最大概率不会提升小概率 occurrence

需要区分两个事件：

1. 某条 occurrence 是否进入 MFG；
2. 某个远程对象是否至少被任何 occurrence 使用。

假设同组两条 occurrence 共用 $U$，但概率分别为：

$$
p_A=0.1,
\qquad
p_B=0.4.
$$

实际决策是：

$$
z_A=\mathbb I[U<0.1],
\qquad
z_B=\mathbb I[U<0.4].
$$

| $U$ 的范围 | A | B | 是否拉取对象 |
| --- | ---: | ---: | ---: |
| $0\le U<0.1$ | 选中 | 选中 | 是 |
| $0.1\le U<0.4$ | 不选 | 选中 | 是 |
| $0.4\le U<1$ | 不选 | 不选 | 否 |

因此两条 occurrence 的边际概率仍为：

$$
\Pr(z_A=1)=0.1,
\qquad
\Pr(z_B=1)=0.4.
$$

同一对象至少出现一次的概率才是：

$$
\Pr(z_A\lor z_B)
=
\max(0.1,0.4)
=
0.4.
$$

独立随机数下，该对象至少出现一次的概率为：

$$
1-(1-0.1)(1-0.4)
=
0.46.
$$

所以 $\max p$ 是对 occurrence 决策取 OR 后的**唯一对象请求概率**，不是重新
分配给每条 occurrence 的阈值。

如果错误地让所有 occurrence 都执行：

$$
z'_{r,i}
=
\mathbb I
\left[
U<\max_j p_{j,i}
\right],
$$

小概率 A 才会从 0.1 被提升到 0.4。当前 native sampler 不是这种实现：

1. 每条 occurrence 单独计算 `edge_probability`；
2. 每条 occurrence 与自身概率比较；
3. 只有通过自身阈值的 occurrence 才写入 `src_index/sample_nodes/eid`；
4. 后续只对数据拉取 ID 去重，不会把未选 occurrence 补回 MFG。

## 6. 为什么它不是完全无损

边际概率不变不等于联合分布不变。

若 $p_A\le p_B$，共享随机数下：

$$
\Pr(z_A=1,z_B=1)=p_A.
$$

两条决策的协方差为：

$$
\operatorname{Cov}(z_A,z_B)
=
p_A-p_Ap_B
=
p_A(1-p_B)>0.
$$

独立采样时协方差为 0。coordinated sampling 可能造成：

- batch 内有效独立随机性下降；
- selection variance 和 gradient covariance 改变；
- 收敛速度随 group size 变化；
- 最终 AP/AUC 出现数据集相关变化。

多层采样的影响更复杂。不同 layer 使用不同 keyed random，但上一层的相关选择会
改变下一层：

- frontier size；
- 节点和时间戳重复度；
- 候选邻居集合；
- 远程对象复用机会；
- 完整 MFG 和梯度分布。

因此当前只能表述为：

> 给定某层某条候选 occurrence 及其概率时，一阶边际概率保持不变；完整多层 MFG
> 分布和训练过程不保证不变。

## 7. 单机四卡实验结果

### 7.1 三 seed 短程系统测试

环境：

- 单机 4 × NVIDIA A40；
- 每个数据集 30 batch；
- seeds 6773/6774/6775；
- coordinated 相对原 boundary-decay。

| 数据集 | batch 时间 | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: |
| WIKI | +0.32% ± 1.62% | -22.23% ± 0.07% | -1.64% ± 0.51% |
| LASTFM | -2.18% ± 1.05% | -2.09% ± 0.04% | -7.42% ± 0.21% |
| WikiTalk | -0.62% ± 3.77% | -6.21% ± 0.34% | -9.37% ± 0.10% |
| Flight | +0.59% ± 1.31% | -12.19% ± 0.19% | -14.44% ± 0.07% |
| 四数据集非加权平均 | -0.48% | -10.68% | -8.22% |

结论：

- 远程节点和远程边下降方向稳定；
- 时间变化处于短程测量波动范围；
- 短程测试不能代替完整收敛实验。

### 7.2 完整轮数测试

实验设置：

- 单机 4 × NVIDIA A40；
- seed 6773；
- WIKI/LASTFM 各 50 epochs；
- WikiTalk/Flight 各 10 epochs。

精度变化单位为 coordinated 相对 baseline 的百分点：

| 数据集 | Test AP 变化 | Test AUC 变化 | batch 时间 | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WIKI | -0.289 | -0.266 | -0.66% | -22.37% | -2.02% |
| LASTFM | +0.737 | +0.881 | -1.22% | -2.31% | -6.26% |
| WikiTalk | -0.014 | +0.007 | +1.68% | -12.31% | -10.51% |
| Flight | -0.054 | -0.038 | +1.08% | -12.85% | -14.86% |
| 四数据集非加权平均 | — | — | +0.22% | -12.46% | -8.41% |

每个 rank、每个 batch 的 edge-feature 节省量：

| 数据集 | 节省量 |
| --- | ---: |
| WIKI | 39.7 KiB |
| LASTFM | 136.5 KiB |
| WikiTalk | 244.1 KiB |
| Flight | 458.0 KiB |

严格结论：

> 远程对象数稳定下降，但单机四卡 batch 时间持平；精度变化具有数据集相关性，
> 当前不能声称单机端到端加速或完全精度无损。该结果构成双机 8 卡、10GbE
> 验证的动机；双机结果及 layer-restricted coordination 的下一步判断见后文。


## 8. 双机八卡 10GbE 结果更新

环境为 `gpu07 + gpu06`、8 × NVIDIA A40、固定 8-way partition。NCCL
显式设置 `NCCL_IB_DISABLE=1`、`NCCL_NET=Socket` 和
`NCCL_SOCKET_IFNAME=ens4f1np1`；两端日志确认走 10GbE Socket，没有
`NET/IB`。

WIKI/LASTFM 各 50 epochs、WikiTalk/Flight 各 10 epochs，seeds
6773/6774/6775 均已完成配对。变化为 coordinated 相对同 seed baseline；精度
单位为百分点，`±` 为三个 paired seed 差值的样本标准差：

| 数据集 | batch 时间 | 远程节点 | 远程边 | test AP Δ | test AUC Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| WIKI | -5.52% ± 2.06% | -23.84% ± 0.04% | -2.33% ± 0.03% | +0.2208 ± 0.9411 | +0.3146 ± 1.0649 |
| LASTFM | -2.09% ± 1.18% | -2.49% ± 0.01% | -6.35% ± 0.02% | -0.5564 ± 1.7911 | +0.0707 ± 1.7580 |
| WikiTalk | -1.97% ± 0.48% | -13.30% ± 0.02% | -11.34% ± 0.01% | -0.0060 ± 0.2541 | +0.0498 ± 0.3592 |
| Flight | -5.18% ± 0.13% | -11.99% ± 0.01% | -15.55% ± 0.02% | -0.0195 ± 0.0740 | -0.0244 ± 0.0302 |
| 四数据集非加权平均 | -3.69% | -12.90% | -8.89% | -0.0903 | +0.1027 |

完整训练关键路径在 12/12 个 paired seed 中缩短；test AP 则 8/12 为负、4/12
为正，且每个数据集都有正负 seed。WIKI/LASTFM 的 test AP paired 标准差分别为
0.9411/1.7911 个百分点，因此不能把平均 -0.0903 个百分点解释为精度无损。
95%-AP time-to-target 的四数据集非加权平均增加 10.12%，seed 间变化很大，也未
形成一致收敛收益。

相同 seed 下，每个数据集进一步做 3 次 100-batch 配对并交替运行顺序。网卡
字节为两台机器 `ens4f1np1` 在训练窗口内的 RX+TX 合计除以 2；起止各等待
1 秒让驱动计数刷新，等待不计入训练时间。结果如下，变化为三次 paired
百分比变化的均值 ± 样本标准差：

| 数据集 | mean batch | max-rank p95 | 训练窗口 wire bytes | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WIKI | -1.42% ± 0.69% | -0.60% ± 3.10% | -10.682% ± 0.002% | -23.96% | -2.60% |
| LASTFM | -0.66% ± 1.76% | -0.47% ± 5.04% | -3.842% ± 0.264% | -2.50% | -6.69% |
| WikiTalk | -2.48% ± 2.96% | -0.55% ± 4.37% | -8.099% ± 0.028% | -13.90% | -11.26% |
| Flight | -7.59% ± 3.39% | -3.02% ± 1.33% | -12.484% ± 0.004% | -12.09% | -15.76% |
| 四数据集非加权平均 | -3.04% | -1.16% | -8.78% | -13.11% | -9.08% |

wire bytes 在 12/12 个配对中下降，mean batch 在 10/12 中下降，p95 在 8/12
中下降。因此可以确认通信量收益，但 p95 仍不足以宣称稳定尾延时收益。

大数据集不做 batch 截断，各运行 1 个完整训练 epoch，跳过评估：

| 数据集 | batches/rank | epoch baseline → coord | epoch 变化 | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: | ---: | ---: |
| StackOverflow | 1,852 | 103.303 s → 101.650 s | -1.60% | -11.82% | -7.51% |
| GDELT | 5,357 | 276.743 s → 279.363 s | +0.95% | -19.20% | -15.63% |

GDELT 中 sampler -2.40%，但 AllToAll route +19.20%、launch +17.55%，暴露
fetch wait +1.83%。整轮网卡估算字节下降 14.79%，仍未转化为 epoch
加速。所以现阶段应表述为“通信量有效，端到端收益不稳定”，而不是
“大图加速已经成立”。详细结果见
[双机八卡 10GbE 初步结果](coordinated_boundary_decay_8gpu_socket_results.md)。

## 9. 阶段延时、统计开销与收敛性

三 seed 完整轮数的 host-wall 阶段计时如下，单位为 `ms/call`。括号内是三个
paired seed 百分比变化的均值 ± 样本标准差：

| 数据集 | sampler baseline → coord | AllToAll route baseline → coord | launch baseline → coord | 暴露 fetch wait baseline → coord |
| --- | ---: | ---: | ---: | ---: |
| WIKI | 18.352 → 16.276 (-11.29% ± 1.26%) | 1.672 → 1.409 (-14.40% ± 15.20%) | 0.1594 → 0.1545 (-3.05% ± 2.96%) | 0.0127 → 0.0119 (-5.81% ± 8.96%) |
| LASTFM | 23.923 → 22.332 (-6.62% ± 2.72%) | 1.166 → 1.157 (-0.75% ± 3.00%) | 0.1548 → 0.1548 (-0.02% ± 1.56%) | 0.0159 → 0.0143 (-9.51% ± 11.88%) |
| WikiTalk | 26.854 → 25.671 (-4.41% ± 0.54%) | 1.212 → 1.203 (-0.74% ± 0.26%) | 0.1539 → 0.1542 (+0.19% ± 0.50%) | 0.0211 → 0.0196 (-7.15% ± 7.01%) |
| Flight | 36.421 → 33.631 (-7.66% ± 0.39%) | 1.147 → 1.191 (+3.81% ± 4.39%) | 0.1552 → 0.1568 (+1.06% ± 2.92%) | 0.0216 → 0.0208 (-3.85% ± 6.37%) |

四数据集非加权平均 paired 变化为 sampler -7.49%、route -3.02%、launch
-0.46%、暴露 fetch wait -6.58%。需要注意：这些阶段存在 pipeline 重叠，不能
直接相加成 batch latency；fetch wait 的绝对值只有约 0.01–0.02 ms/call，说明
大部分通信原本已被计算覆盖。远程对象和字节减少后，固定 collective 次数、route、
launch、forward/backward 与 memory 更新仍在，因此 batch 时间不会按字节同比下降。

阶段计时只在显式开启 `--core_collect_phase_timing` 时收集。WIKI 30-batch 的一次
计时开/关对照为 52.258/53.487 ms/batch，开启计时反而快 2.30%；单次配对不能把
负开销解释为收益，但说明当前未检测到高于短跑波动的可分辨统计开销。

统计效率方面，以每个 seed 的 baseline 最佳 validation AP 的 95% 为 target：

| 数据集 | 三 seed time-to-target 变化均值 ± 标准差 |
| --- | ---: |
| WIKI | +16.38% ± 31.07% |
| LASTFM | -0.29% ± 9.01% |
| WikiTalk | -1.97% ± 0.48% |
| Flight | +26.38% ± 54.56% |
| 四数据集非加权平均 | +10.12% |

该 target 对 10-epoch 数据集的一次 epoch 跳变很敏感，不能作为精确加速比；但它
足以否定“batch 更短必然带来更快收敛”。后续应优先降低 group/layer 相关性，再
重复统计效率测试。

## 10. Burst-Aware Smooth 备选方向

### 10.1 已尝试的分层

根据节点训练时间流的 inter-arrival burstiness $B_v$ 离线分组：

$$
\operatorname{group}(v)
=
\begin{cases}
\mathrm{low}, & 0\le B_v<0.55,\\
\mathrm{mid}, & 0.55\le B_v<0.75,\\
\mathrm{high}, & B_v\ge0.75.
\end{cases}
$$

已测试：

- `all`；
- `none`；
- `low_burst_only`；
- `mid_burst_only`；
- `high_burst_only`。

这些 policy 是静态 hard selection，只决定哪些 shared node event 使用 smooth；
不会改变 fetch、SSIM、partition 或 ownership。

### 10.2 WIKI/LASTFM 50-epoch 结果

实验为单机四卡、seed 6773，按 validation AUC 选择 epoch。变化相对默认 `all`，
单位为百分点：

| 数据集 | policy | Test AP 变化 | Test AUC 变化 |
| --- | --- | ---: | ---: |
| WIKI | low-only | +0.1958 | +0.1966 |
| WIKI | mid-only | +0.1105 | +0.1319 |
| WIKI | high-only | -0.0990 | -0.0286 |
| LASTFM | low-only | -1.7612 | -2.3026 |
| LASTFM | mid-only | -2.1407 | -2.2250 |
| LASTFM | high-only | -1.8604 | -2.4046 |

WIKI seed 6773 中 low/mid-only 优于 all，high-only 最差；LASTFM 中 all-smooth
明显最好。

WIKI seed 6774 的部分重复结果为：

| policy | 相对 all 的 Test AP 变化 | 相对 all 的 Test AUC 变化 |
| --- | ---: | ---: |
| none | -0.4650 | -0.5290 |
| low-only | -0.3127 | -0.3800 |

该 seed 中 low-only 高于 none，但没有复现“low-only 超过 all”。

### 10.3 四数据集短筛

2-epoch、单 seed 的 Test AUC：

| 数据集 | all | high-only | 该轮最好 policy |
| --- | ---: | ---: | --- |
| WIKI | 0.876365 | 0.873330 | mid-only |
| LASTFM | 0.805926 | 0.786812 | all |
| Flight | 0.945814 | 0.945974 | high-only |
| WikiTalk | 0.964205 | 0.959503 | all |

Flight 中 99.87% 的 event occurrence 位于 high-burst 分组，因此 high-only 几乎
等价于 all，不能证明 high-burst 独立有效。

### 10.4 当前判断

现有证据只支持：

- smooth 平均有益，但存在 event-level 坏例；
- WIKI/WikiTalk 中 high-burst 条件存在进一步分析空间；
- 静态 hard cutoff 不能跨数据集泛化；
- 尚无方法保证在抑制 high-burst 坏例时保留其他 event 的 smooth 正收益。

若以后重启，优先尝试以 all-smooth 为锚点的保守 high-tail attenuation：

$$
m_v^{\mathrm{out}}
=
m_v^{\mathrm{cur}}
+
\lambda_{\mathrm{hist}}g_v
\left(
\bar m_v-m_v^{\mathrm{cur}}
\right),
$$

其中 high-tail 的 $g_v$ 有严格下界，不直接设为 0。该设计目前只是备选设想，
不是已验证方法。

## 11. 当前取舍

### 11.1 进入下一步验证

- coordinated boundary-decay；
- GDELT AllToAll route/launch 上升归因；
- group-size 与 layer-restricted coordination；
- 在候选 group/layer 配置选定后补充精度与收敛性统计。

### 11.2 只作备选

- burst-aware conservative smooth；
- static feature prefetch + dynamic memory JIT。

### 11.3 暂停或停止

- 当前 residual-aware sampling；
- `core-static`、top-1 和提高采样概率；
- 当前 Python edge cache；
- hard-negative/cost-only negative；
- full CoRe 组合调参；
- 固定 per-node 和现有 event-conditioned smooth gate。


## 12. 其他角度
目前考虑的几个角度基本上是围绕：
（1）采样侧：微调采样概率（做成可学习的开销大、偏好方案一的处理，同时有去重支持）、去重（这个做了很久也很多次尝试了效果上不来，偏好性很低）
（2）kernel侧：不熟练，之前做了一些尝试，有提升但不明显，而且不是在这个代码上改的，也不太好讲故事。之前chunk预处理采样来减少prefetch的处理已经被别人发出来了，也做的比我们成熟。
（3）图划分侧（改动空间不大，和hot nodes是强耦合的，现有方法是没有做批次的迁移和图划分，改造代价大，可能代码成本和代码开销可能会恶化）
（4）图推理侧（简单做一个支持推理阶段场景的Implementation，而不是提前选择热点，只在推理阶段做演化不影响训练效率，而且能增加场景模式覆盖，这个我偏好性高一点，而且拓展期刊有场景优势）

## 13. 汇报结语

本阶段最重要的发现不是“已经获得训练加速”，而是：

> Coordinated randomness 在双机八卡三次系统复测中将四数据集训练窗口网卡
> 字节平均降低 8.78%，但 p95 仅平均降低 1.16%；三 seed 完整训练的关键路径
> 平均缩短 3.69%、12/12 配对加速，但 test AP 8/12 为负、四数据集平均
> -0.0903 个百分点，time-to-target 也不稳定；StackOverflow/GDELT 单 epoch
> 时间分别为 -1.60%/+0.95%。当前证据支持“通信量和中小图关键路径稳定有效，
> 尾延时、大图和统计效率尚未同时稳定”；下一步先筛选 group size/分层配置，
> 并定位 GDELT route/launch 开销。

burst-aware smooth 揭示了 high-burst 坏例分析的可能空间，但现有 hard 分层缺少
跨数据集稳定性，因此只作为后续备选。

## 14. 结果与复现入口

| 内容 | 位置 |
| --- | --- |
| Coordinated 单机四卡完整轮数结果 | `outputs/core_memshare_screen/coordinated-decay-full-epochs-seed6773/matrix_summary.json` |
| 双机八卡 Socket、三 seed 完整训练、三次系统复测和大图单 epoch | [coordinated_boundary_decay_8gpu_socket_results.md](coordinated_boundary_decay_8gpu_socket_results.md) |
| CoRe 实验汇总 | [core_memshare_experiments.md](core_memshare_experiments.md) |
| WIKI/LASTFM burst 50-epoch | [wiki_lastfm_50ep_smooth_summary.csv](../outputs/pattern_ablation_logs/wiki_lastfm_50ep_smooth_summary.csv) |
| 四数据集 burst 初筛 | [burst_strength_summary.csv](../outputs/pattern_ablation_logs/burst_strength_summary.csv) |
| 方法与理论边界 | [journal_extension_effective_methods.md](journal_extension_effective_methods.md) |
| 全部优化决策 | [journal_extension_optimization_review.md](journal_extension_optimization_review.md) |
