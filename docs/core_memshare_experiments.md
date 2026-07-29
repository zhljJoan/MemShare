# CoRe-MemShare 四数据集实验记录

更新时间：2026-07-29

代码分支：`core-memshare`

固定数据集：WIKI、LASTFM、WikiTalk、Flight

## 1. 当前结论

本轮最终只保留一个扩展技术点：

```text
boundery_recent_decay_coordinated
```

下文简称 **coordinated-decay**。它完全复用原 MemShare
`boundery_recent_decay` 的候选集、时间衰减概率、回填规则和 attention 路径，
只让同一小组 root 对同一远程对象共享随机数。原 baseline 仍是兼容性默认值，
coordinated-decay 是本分支唯一推荐继续做长训练和跨机验证的期刊扩展。

选择它的原因是：30-batch、3 seeds 下，四数据集平均远程节点减少 10.68%，
远程边减少 8.22%，batch 时间平均变化 -0.48%。随后完成 seed 6773 的完整轮数
对照：WIKI/LASTFM 各 50 epochs，WikiTalk/Flight 各 10 epochs。按最佳
validation AP 所在 epoch 选择 test 指标，四个数据集的 test AP 变化为：

| WIKI | LASTFM | WikiTalk | Flight |
| ---: | ---: | ---: | ---: |
| -0.00289 | +0.00737 | -0.00014 | -0.00054 |

因此它仍是唯一继续推进的扩展，但结论从“早期实验无系统回退”修正为“通信下降
稳定，完整训练存在最多约 0.0029 AP 的小幅数据集相关权衡”。完整轮数目前只有
一个 seed；跨机器链路收益和完整轮数多随机种子仍需后续确认。

保留的是可复用且已经验证的实验基础设施与正确性修复：

- 强制覆盖四个数据集的矩阵运行器和结构化指标；
- 对比实验冻结 adaptive sampling/synchronization 参数，避免算法与预算互相
  改写；
- 默认关闭会触发 GPU 同步的逐 batch 概率诊断，只在
  `--core_collect_diagnostics` 下采集；
- request planner 只在显式启用时运行，不再污染 sampler 的计时；
- Flight 空 edge batch 保护；
- predictor 只观测真实远程获取节点；
- edge cache 批量写入与 O(1) 空槽管理；
- hard-negative 候选恢复原 localism proposal，并加入两阶段 importance
  correction；
- 18 个 sampling、cache、negative、planner、MFG 和 native sampler 回归测试。

下列模块保留为 **默认关闭的实验代码**，只用于复现和下一轮重构，不属于当前
推荐组合：

- 在线 last-fetched residual feedback；
- probability-aware edge cache；
- communication-aware hard negative；
- static communication-aware sampler；
- newest top-1 deterministic protection；
- 上述模块的完整组合。

原因不是这些模块没有局部通信收益，而是它们没有同时通过四数据集的端到端
时间与训练质量门槛。特别地，`core-static` 不是最终保留项；它改变采样概率并
在 WIKI 上产生可重复的早期质量回退。coordinated-decay 则保留原概率，只改变
跨 root 的随机相关性。

## 2. 强制实验规则

任何一次方法判断必须同时包含：

```text
WIKI + LASTFM + WikiTalk + Flight
```

`scripts/run_core_memshare_matrix.py` 将数据集集合写死，4-way partition 下也不
允许改用非 4 卡运行。只有所有 dataset/variant/repeat cell 都生成有限 loss
和结构化 JSON，矩阵才标记为 complete。

本轮固定设置如下：

| 项目 | 设置 |
| --- | --- |
| GPU | 4 × NVIDIA A40 |
| model | `TGN_large` |
| partition | 固定 4-way、`top0.1` |
| sampling budget | `probability=0.1` |
| memory | `historical` |
| shared-memory threshold | `0.3` |
| coordinated group | 8 |
| adaptive parameters | 对比期间冻结 |
| 默认 seed | 6773；多 seed 时使用 6773、6774、6775 |

筛选门槛按以下顺序执行：

1. 四数据集完整、loss 有限、collective 正常；
2. 短跑检查每 batch 时间、远程节点、远程边、cache/negative 诊断；
3. 只有系统指标有希望的变体进入 train/validation/test screen；
4. 任一数据集出现明确质量回退，就不因其他数据集的吞吐收益而保留；
5. 短 screen 可以用于淘汰，不能替代期刊最终长训练和多随机种子结果。

## 3. 尝试顺序与有效性

| 运行名 | 范围 | 结果用途 |
| --- | --- | --- |
| `validation-2b` | 4 数据集，2 batch | Flight 暴露空 edge reshape，矩阵不完整，不能用于方法判断 |
| `validation-2b-fixed` | 4/4 | 验证 Flight 修复和矩阵执行链 |
| `screen-30b-seed6773` | 20/20 cells | adaptive 参数未冻结，cache 时延改变后续采样预算；结果判为无效 |
| `screen-30b-fixed-seed6773` | 20/20 | 冻结参数后的第一轮筛选，定位 Python 热路径和 hard-negative proposal 问题 |
| `optimize-30b-fixed-seed6773` | 16/16 | predictor/cache 优化后的模块筛选 |
| `isolate-30b-fixed-seed6773` | 12/12 | 隔离 static sampler 与修正后的 hard negative |
| `confirm-core-static-30b-3seed` | 24/24 | static sampler 三随机种子系统确认 |
| `quality-1epoch-seed6773` | 24/24 | static sampler 三随机种子早期质量筛选 |
| `quality-3epoch-seed6773` | 8/8 | static sampler 三 epoch 质量确认 |
| `quality-top1-3epoch-seed6773` | 8/8 | newest top-1 保护的最终针对性尝试 |
| `final-candidates-30b-seed6773` | 36/36 | 去除统计同步污染后，复核 static、校正、概率、cache 和 negative 候选 |
| `final-quality-3epoch-seed6773` | 20/20 | 对仍可能保留的 static/校正变体做质量归因 |
| `coordinated-decay-30b-3seed` | 24/24 | 保持 baseline 概率的协同采样，三 seed 系统确认 |
| `coordinated-decay-quality-3epoch-3seed` | 24/24 | 最终候选三 seed、三 epoch 精度确认 |
| `coordinated-decay-full-epochs-seed6773` | 8/8 | WIKI/LASTFM 50 epochs、WikiTalk/Flight 10 epochs 的完整轮数对照 |

第一次 30-batch matrix 的动态 adaptive 参数会让慢模块在后续 batch 得到不同
的 `alpha/beta`，看上去像是模块节省了大量通信。加入
`--core_freeze_adaptive_parameters` 后才消除这一混杂，因此冻结前的数据只
记录为失败尝试。

第二个计时混杂来自诊断代码。早期 `sample_prob.sum/min/max().item()` 在每个
batch 都执行，会强制 GPU/CPU 同步；同时 metrics-only request planner 也曾被
所有变体默认启用。最终候选轮次已将二者改为显式 opt-in。旧轮次仍可说明
correctness 和通信方向，但最终时间判断只使用清理后的轮次。

## 4. 30-batch 系统筛选

以下为 predictor 仅观测远程节点、cache 批量写入后的配对汇总；变化均相对同轮
baseline：

| 变体 | batch 时间 | 远程节点 | 远程边 | 诊断 | 决定 |
| --- | ---: | ---: | ---: | --- | --- |
| online residual | +50.08% | -10.64% | -6.69% | residual 更新仍在 Python/CPU 热路径 | 暂缓 |
| edge cache | +200.4% | 约 0% | -5.54% | 四数据集合计节省 59.27 MiB；hit rate 1.81%–8.97% | 暂缓 |
| residual + cache | +154.7% | -10.70% | -12.35% | 通信收益不足以覆盖 lookup/admission 开销 | 暂缓 |
| hard negative | +238.15% | -10.70% | -0.29% | ESS 很低，见下表 | 暂缓 |
| core-static，3 seeds | -5.46% ± 0.45% | -10.82% ± 0.09% | -6.45% ± 0.18% | 唯一进入质量 screen 的变体 | 进入质量验证 |

edge cache 的 object hit rate：

| WIKI | LASTFM | WikiTalk | Flight |
| ---: | ---: | ---: | ---: |
| 4.15% | 1.81% | 6.59% | 8.97% |

这说明缓存确实减少静态边特征字节，但当前逐 batch 查询、评分和替换实现的成本
显著高于节省的通信时间。下一轮必须先做 GPU/vectorized lookup 和离线 admission
更新，不能继续在现实现上增加策略复杂度。

修正 localism proposal 后，hard negative 的诊断为：

| 数据集 | mean hardness | ESS / samples |
| --- | ---: | ---: |
| WIKI | 0.5067 | 12.22% |
| LASTFM | 0.5089 | 6.03% |
| WikiTalk | 0.5006 | 5.14% |
| Flight | 0.5715 | 3.79% |

除 Flight 外，hardness 基本接近未知 embedding 的 0.5 默认值，同时 importance
weight 使 ESS 降至 3.79%–12.22%。因此当前有限候选池没有提供足够训练信息增益，
却带来 238% 的 batch 时间开销。

## 5. static sampler 的质量门槛

`core-static` 关闭在线 predictor 和 edge cache，只保留 native
communication-aware probability 与 grouped randomness。30-batch、三随机种子
下，它的系统收益相对稳定：

| 数据集 | batch 时间 | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: |
| WIKI | +1.55% ± 4.90% | -20.25% | +0.52% ± 0.56% |
| LASTFM | -8.12% | -1.58% | -5.69% |
| WikiTalk | -7.24% | -4.99% | -4.13% |
| Flight | -7.66% | -11.86% | -14.53% |

一 epoch、三随机种子的 AP 差值噪声较大，尤其是 WIKI，因此只作为继续验证的
依据：

| 数据集 | validation AP Δ | test AP Δ |
| --- | ---: | ---: |
| WIKI | +0.02261 ± 0.03552 | +0.02018 ± 0.02934 |
| LASTFM | +0.00800 ± 0.00829 | +0.01380 ± 0.02463 |
| WikiTalk | -0.00095 ± 0.00164 | -0.00011 ± 0.00254 |
| Flight | -0.00130 ± 0.00148 | -0.00160 ± 0.00177 |

三 epoch、seed 6773 的同轮 baseline 对照给出：

| 数据集 | batch 时间 | 远程节点 | 远程边 | validation AP Δ | test AP Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| WIKI | -9.34% | -20.32% | +0.16% | -0.00884 | -0.01012 |
| LASTFM | -17.64% | -1.67% | -4.24% | +0.00230 | +0.00344 |
| WikiTalk | -13.84% | -9.50% | -3.99% | -0.00105 | -0.00347 |
| Flight | -17.42% | -12.54% | -14.97% | +0.00072 | +0.00115 |

虽然四个数据集的 batch 时间都改善，但 WIKI test AP 下降 0.01012，因而
`core-static` 未通过质量门槛。

## 6. newest top-1 保护尝试

为验证 WIKI 回退是否来自最新邻居被随机丢弃，新增实验参数：

```text
--core_deterministic_recent 1
```

最新一个 candidate 的 inclusion probability 固定为 1，原通信目标扣除其成本后
只分配给 tail。三 epoch、seed 6773 的完整结果如下：

| 数据集 | batch 时间 | 远程节点 | 远程边 | validation AP Δ | test AP Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| WIKI | -7.69% | -5.49% | -6.31% | -0.02263 | -0.02214 |
| LASTFM | -18.41% | -0.16% | +1.85% | -0.00463 | -0.00238 |
| WikiTalk | -9.41% | +4.29% | +10.09% | -0.00229 | -0.00235 |
| Flight | -13.40% | -9.74% | -26.11% | +0.00155 | +0.00042 |

该保护没有修复 WIKI，反而在同轮对照中扩大了质量回退；WikiTalk 的远程节点和
边也增加。因此不继续尝试 top-2/top-L 扫描，top-1 仅作为默认关闭的复现实验
保留。

## 7. 其他候选的最终复核

清除逐 batch GPU 同步和默认 planner 后，用相同 seed 6773、30 batch
重新比较候选。下表是四数据集非加权平均，相对同轮 baseline：

| 变体 | batch 时间 | 远程节点 | 远程边 | 判断 |
| --- | ---: | ---: | ---: | --- |
| core-static | -6.40% | -9.76% | -5.99% | 有系统收益，进入质量归因 |
| core-independent | -6.74% | -5.83% | +2.05% | 去掉协同后边请求反而增加 |
| core-clipped | -7.08% | -9.76% | -5.99% | 只减弱 probability correction |
| core-uncorrected | -6.81% | -9.76% | -5.99% | 完全关闭 probability correction |
| core-p15 | -4.92% | +16.49% | +29.20% | 通信明显增加，放弃 |
| core-p20 | -4.67% | +39.11% | +59.96% | 通信明显增加，放弃 |
| edge-cache-frozen | +68.67% | 0% | -2.50% | 无替换仍太慢，放弃当前实现 |
| cost-negative | +7.22% | -12.60% | -1.29% | 改变梯度但边收益很小，放弃 |

`edge-cache-frozen` 已去掉复杂评分和替换，仍有 +68.67% 的时间开销，说明当前
瓶颈主要是动态 Python dictionary lookup、admission 和 miss routing，而不是
评分公式本身。它仍有算法空间，但在完成 GPU/vectorized cache index 前不应继续
扫缓存策略。

`cost-negative` 已去掉 embedding hard score，只保留通信代价候选重排；它仍使
batch 时间增加 7.22%，远程边只下降 1.29%，且负样本分布与梯度方差发生变化。
因此 hard negative 也不进入最终方案。

对 core-static 及其 attention 校正变体做三 epoch、seed 6773 质量归因后，
test AP 变化如下：

| 变体 | WIKI | LASTFM | WikiTalk | Flight |
| --- | ---: | ---: | ---: | ---: |
| core-static | -0.02188 | -0.00017 | -0.00213 | +0.00168 |
| core-independent | -0.02452 | -0.01631 | +0.00019 | +0.00158 |
| core-clipped | -0.02068 | -0.00531 | -0.00150 | +0.00145 |
| core-uncorrected | -0.01581 | -0.00146 | +0.00079 | +0.00199 |

完全关闭 correction 只能缩小 WIKI 回退，不能消除它；group=1 也没有修复。
因此问题不只是 HT/self-normalized attention 校正，而是 core-static 本身改变了
原始 boundary-decay 的采样分布。继续扫 correction floor、\(p=0.15/0.20\)
没有形成可信的保留路径。

## 8. 最终保留：coordinated boundary-decay

最终策略不再计算 residual/cost-aware 新概率。对 baseline 给出的每个
root-object 概率 \(p_{r,i}\)，group 内共享：

\[
h_b(i)\sim U(0,1),\qquad
z_{r,i}=\mathbb I[h_b(i)<p_{r,i}].
\]

所以每个 occurrence 仍满足：

\[
\Pr(z_{r,i}=1)=p_{r,i}.
\]

独立采样时，同一对象在一组 root 中至少被请求一次的概率为：

\[
1-\prod_r(1-p_{r,i}),
\]

协同后变为：

\[
\max_r p_{r,i}.
\]

实现使用 `(remote_node_id, root_group, batch, layer, seed)` 生成 keyed
uniform；group 大小固定为 8。它不导出 `sample_prob`，不执行
probability-corrected attention，也不启用 predictor、cache、hard negative 或
request-planner 热路径。

### 8.1 三 seed 系统结果

`coordinated-decay-30b-3seed` 的配对均值和 seed 间标准差如下：

| 数据集 | batch 时间 | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: |
| WIKI | +0.32% ± 1.62% | -22.23% ± 0.07% | -1.64% ± 0.51% |
| LASTFM | -2.18% ± 1.05% | -2.09% ± 0.04% | -7.42% ± 0.21% |
| WikiTalk | -0.62% ± 3.77% | -6.21% ± 0.34% | -9.37% ± 0.10% |
| Flight | +0.59% ± 1.31% | -12.19% ± 0.19% | -14.44% ± 0.07% |
| 四数据集非加权平均 | **-0.48%** | **-10.68%** | **-8.22%** |

当前单机 4 GPU 上，hash 计算开销与通信节省基本抵消，不能声称已有显著训练
加速；有效证据是远程对象数量稳定下降。跨机器环境中通信代价更高，预期更容易
转化为时间收益，但这是待验证推断。

### 8.2 早期三 epoch、三 seed 质量结果

`coordinated-decay-quality-3epoch-3seed` 报告每个 seed 的第 3 个 epoch
validation/test 指标；下表为相对同 seed baseline 的配对均值 ± 标准差：

| 数据集 | validation AP Δ | test AP Δ | validation AUC Δ | test AUC Δ |
| --- | ---: | ---: | ---: | ---: |
| WIKI | +0.00076 ± 0.00344 | +0.00210 ± 0.00382 | +0.00066 ± 0.00573 | +0.00101 ± 0.00675 |
| LASTFM | -0.00247 ± 0.00308 | +0.00631 ± 0.00718 | -0.00387 ± 0.00304 | +0.00058 ± 0.00759 |
| WikiTalk | -0.00017 ± 0.00106 | -0.00035 ± 0.00187 | -0.00003 ± 0.00125 | -0.00050 ± 0.00240 |
| Flight | +0.00100 ± 0.00264 | +0.00059 ± 0.00299 | +0.00074 ± 0.00262 | +0.00052 ± 0.00326 |

没有数据集出现跨三个 seed 同方向的 test AP 下降。该实验说明方法值得进入长
训练，但只有三 epoch，不能作为最终精度结论。

### 8.3 完整轮数、seed 6773 结果

最终补充的完整轮数为：

| 数据集 | epochs |
| --- | ---: |
| WIKI | 50 |
| LASTFM | 50 |
| WikiTalk | 10 |
| Flight | 10 |

每个变体都在完成全部 epochs 后，选择 validation AP 最高的 epoch，并报告该
epoch 已同步计算的 test AP/AUC。结果如下：

| 数据集 | baseline / coordinated 最佳 epoch | validation AP Δ | test AP Δ | validation AUC Δ | test AUC Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| WIKI | 46 / 49 | -0.00347 | -0.00289 | -0.00297 | -0.00266 |
| LASTFM | 48 / 49 | +0.00857 | +0.00737 | +0.00899 | +0.00881 |
| WikiTalk | 9 / 9 | -0.00008 | -0.00014 | +0.00022 | +0.00007 |
| Flight | 9 / 9 | -0.00105 | -0.00054 | -0.00077 | -0.00038 |

全训练阶段的累计通信和平均训练 batch 时间变化为：

| 数据集 | batch 时间 | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: |
| WIKI | -0.66% | -22.37% | -2.02% |
| LASTFM | -1.22% | -2.31% | -6.26% |
| WikiTalk | +1.68% | -12.31% | -10.51% |
| Flight | +1.08% | -12.85% | -14.86% |
| 四数据集非加权平均 | +0.22% | -12.46% | -8.41% |

该轮修正了早期三 epoch 的乐观表述：WIKI 的完整训练 test AP 下降 0.00289，
Flight 和 WikiTalk 也有更小的 AP 负值。由于 LASTFM 明显改善、其余三个数据集
下降均不超过 0.0029，当前仍允许继续推进，但必须把它表述为通信—精度权衡，并
补充完整轮数多 seed。单机训练 batch 时间平均 +0.22%，仍不能宣称端到端加速。

## 9. 统计效率与运行开销

该策略不是“零代价去重”，存在两类开销：

1. **运行开销**：每个远程候选需要一次 keyed hash/uniform。30-batch、3-seed
   实测四数据集平均 batch 时间为 -0.48%，单数据集变化落在约
   -2.18% 到 +0.59%，说明当前看不到系统性额外时间开销，也不能把 -0.48%
   解释成稳定加速。
2. **统计效率开销**：单个 root 的边际采样分布不变，但同组 root 的选择产生
   正相关；同一 root 内若多个 temporal edge 指向同一 remote node，也会共享
   keyed random。因此 batch gradient 的协方差可能增大，有效独立样本数可能
   下降。group=8 是通信复用与相关性的限幅器，持续变化的 batch/layer key
   会为后续 batch 生成新的随机数。

早期 3-epoch、3-seed AP/AUC 没有显示系统性训练损失；完整轮数的 seed 6773
则在 WIKI 上得到 test AP -0.00289。这与“相关性可能增加统计方差”的风险一致，
但一个 seed 仍不能区分方法偏差和训练随机波动。后续必须同时报告：

- 多 seed 的 time-to-target-AP 和收敛 epoch；
- batch gradient variance 或相邻 root 的 selection correlation；
- group size 1/4/8/16 的通信—方差曲线；
- 多机链路下 unique request、字节数和 wall-clock。

因此最终表述应是“保持边际采样概率并以有限相关性换取请求复用”，而不是“完全
保持 baseline 的联合采样分布”。

## 10. 保留与暂缓清单

| 项目 | 状态 | 依据 |
| --- | --- | --- |
| 原 MemShare baseline | **保留为默认/对照** | 保持 master 兼容性 |
| coordinated-decay，group=8 | **唯一继续推进的扩展** | 完整训练节点 -12.46%、边 -8.41%；WIKI test AP -0.00289，需多 seed |
| 四数据集 matrix + JSON metrics | **保留** | 防止单数据集和动态预算混杂 |
| correctness/performance 修复 | **保留** | 18/18 tests；避免诊断同步和默认 planner 污染 |
| online residual predictor | 暂缓 | 通信下降但 batch 时间 +50.08% |
| probability-aware edge cache | 暂缓 | 完整版 +200.4%；冻结简化版仍 +68.67% |
| hard negative | 暂缓 | batch 时间 +238.15%，ESS 仅 3.79%–12.22% |
| core-static/correction 变体 | 暂缓 | 清理计时后仍有收益，但 WIKI test AP -0.01581 至 -0.02452 |
| core-top1 | 暂缓 | WIKI test AP -0.02214，WikiTalk 通信增加 |
| full CoRe combination | 暂缓 | 组成模块尚未分别通过门槛 |

“暂缓”表示不进入推荐配置和后续长训练，不表示删除复现实验入口。下一轮只有在
完成 vectorized predictor/cache、真正的非零预测残差聚合、以及更稳定的
attention tail 校正后，才重新进入同样的四数据集门槛。

## 11. 复现实验

先构建 native sampler：

```bash
PYTHON_BIN=/home/zlj/.miniconda3/envs/tgnn_3.10/bin/python \
BUILD_DIR=/tmp/core_memshare_native_build \
bash scripts/build_native_sampler.sh
```

运行 30-batch 四数据集筛选：

```bash
/home/zlj/.miniconda3/envs/tgnn_3.10/bin/python \
scripts/run_core_memshare_matrix.py \
  --variants baseline coordinated-decay \
  --batches 30 \
  --repeats 3 \
  --native-lib /tmp/core_memshare_native_build/libstarrygl_sampler.so \
  --run-name coordinated-decay-system
```

运行完整轮数的 validation/test 对照：

```bash
/home/zlj/.miniconda3/envs/tgnn_3.10/bin/python \
scripts/run_core_memshare_matrix.py \
  --variants baseline coordinated-decay \
  --dataset-epochs WIKI=50 LASTFM=50 WikiTalk=10 Flight=10 \
  --repeats 1 \
  --native-lib /tmp/core_memshare_native_build/libstarrygl_sampler.so \
  --run-name coordinated-decay-full-epochs
```

该命令保存逐 epoch history，以 validation AP 最佳的 epoch 作为主结果，并同时
保存最后一个 epoch 的指标。当前先完成 seed 6773；期刊最终表需要把
`--repeats` 提高到至少 3。

暂停方案仍可用对应 variant 名复现，但不应与最终推荐命令混跑后取最优值。若要
重新评估 residual/cache/negative，应先单独建立新的假设和四数据集门槛。

原始日志和 metrics 默认写入被 git 忽略的
`outputs/core_memshare_screen/<run-name>/`；`matrix_summary.json` 中
`complete=true` 才表示本次四数据集实验有效。
