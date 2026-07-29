# MemShare 期刊扩展优化点、实验结果与推进决策

更新时间：2026-07-30

当前分支：`core-memshare`

本文综合以下两份记录：

- [阶段性调研发现](journal_extension_findings.md)：smooth aggregation、热点、
  分区、双机通信与数据加载；
- [CoRe-MemShare 四数据集实验](core_memshare_experiments.md)：正邻居采样、
  edge cache、negative sampling 与请求执行链。

目标是按优化点回答四个问题：

1. 实际尝试了什么；
2. 效率发生了什么变化；
3. 精度发生了什么变化；
4. 是否值得继续推进。

## 1. 结果口径

- AP/AUC 的 `Δ` 均为绝对差值，不是百分比。
- CoRe 系统变化均相对同轮 baseline。最终系统筛选使用单机
  4 × NVIDIA A40、WIKI/LASTFM/WikiTalk/Flight、固定 4-way partition。
- `30-batch、3 seeds` 用于判断通信和运行开销；`3-epoch、3 seeds` 只用于
  早期质量筛选，不能替代完整收敛实验。
- 完整轮数 seed 6773 已完成：WIKI/LASTFM 50 epochs，WikiTalk/Flight
  10 epochs；主指标取 validation AP 最佳 epoch 对应的 test AP/AUC。
- `journal_extension_findings.md` 中已撤销的旧 remote ratio、remote coverage
  和 dynamic drift oracle 数值不进入本表。
- “继续”表示进入下一轮完整训练或多机验证；“条件推进”表示必须先满足列出的
  profiler/重构门槛；“暂停”表示保留复现入口，但不继续组合调参。

## 2. Memory、smooth 与热点方向

| 优化点 | 已尝试方案 | 效率变化 | 精度变化 | 推进结论 |
| --- | --- | --- | --- | --- |
| Smooth aggregation | 同一模型、同一 memory 状态下做 raw/smooth shadow 配对 | 本实验用于精度归因，未单独给出可靠时间收益；现有 SSIM=0.3 已控制同步量 | WIKI：AP +0.001148、AUC +0.001379；LASTFM：AP +0.001490、AUC +0.001364 | **保留为 baseline 机制**。平均收益真实，但不是新的期刊优化点 |
| 全量 smooth + SSIM | `all_ssim03` 与 none/full-sync 等独立训练对照 | SSIM 能减少无显著变化的同步；现有记录没有可跨数据集归一的独立加速数值 | 大多数完整/较大实验中总体最稳定；独立训练差异同时包含训练轨迹影响 | **继续作为默认对照**，不重新包装成新贡献 |
| 固定 burst gate | low/mid/high-burst-only 与 all/none | 没有证明稳定通信或时间收益；policy 主要改变 smooth 使用范围 | WIKI 单 seed low-burst AUC 比 none +0.001072；LASTFM 则 all-smooth 比 none +0.026698；四数据集最优规则不一致 | **暂停**。不能形成跨数据集统一规则 |
| 固定 per-node smooth gate | 用测试流前半段学习节点 raw/smooth 策略，外推到后半段 | 未形成可用效率收益 | WIKI 仅比 all-smooth 多修正 1 个样本；LASTFM 反而少 12 个正确样本 | **停止推进**。节点行为以 mixed 为主 |
| 运行时 smooth gate | burstiness、remote ratio、staleness、memory novelty、方向一致性等特征，使用 logistic/tree | gate 本身会增加统计和预测开销，尚无通信收益 | 时间 holdout AUC 多在 0.497–0.567；接近随机排序 | **停止推进当前特征集** |
| Global gamma 调节 | 检查 gamma 轨迹，并考虑改成普通 gate | 无直接效率收益；有效 history weight 只变化约 0.015–0.016 | 单一全局标量无法解释事件级好坏例，尚无替代方案收益 | **暂停**。保留机制解释，不作为优化 |
| 移除 batch min-max normalization | WIKI 配对消融 | 计算差异很小，没有端到端效率证据 | Accuracy +0.002370，但 AP -0.000315、AUC -0.000690 | **不推进**，主指标下降 |
| 修改 Filter 观测源 | 从 fused increment 改成 raw increment | 无独立效率收益 | WIKI 三 seed：AP -0.000904、AUC -0.001216；LASTFM seed 6773：AP +0.001004、AUC -0.000125 | **不替换默认实现**，仅保留机制消融 |
| 在线动态热点/迁移 | 旧 drift oracle、窗口热点、租约与版本化迁移设想 | 旧 profiler 混用 top-0 home 和 top-K route，收益数字无效；迁移成本未计入 | 未得到有效精度结果 | **停止当前路线**。先修正定义也不足以授权在线重划分 |
| Cross-count 静态热点 | plain METIS 下比较 degree top-K 与跨分区边计数 top-K | Flight：degree/current 热点覆盖约 90.5%–90.6% cross-edge endpoint；cross-count 上限约 94.8%，只有约 4.2 个百分点空间 | 未进行训练精度验证 | **只做离线候选分析**，不支持动态迁移 |
| Route-hot / memory-hot 解耦 | 固定 partition/edge route，只在 route-hot 内选择 `FullShare` 或 `FeatureOnly` | **尚未实测**。理论上可减少低价值 memory replica、同步字节和 GPU memory | `FeatureOnly` 会失去部分 smooth 收益，当前没有质量结果 | **条件推进**：先做同一路由下的逐节点 cost profiler；至少两个数据集预计总通信下降 >10%，且 StackOverflow 关键路径预计 >5% 或显著省显存，才实现 |

## 3. 正邻居采样方向

| 优化点 | 已尝试方案 | 效率变化 | 精度变化 | 推进结论 |
| --- | --- | --- | --- | --- |
| 在线 residual-aware sampling | last-fetched residual、age、edge error 和通信代价共同调整 \(p_i\) | batch 时间 +50.08%；远程节点 -10.64%；远程边 -6.69% | 当前实现仍是 zero-baseline probability correction，没有完整非零预测残差质量证据 | **暂停当前实现**。通信方向有效，但 Python/CPU 在线更新不划算 |
| Static communication-aware sampling (`core-static`) | 关闭 predictor/cache，只保留 native cost-aware probability 和 group randomness | 清理统计开销后：batch 时间 -6.40%、远程节点 -9.76%、远程边 -5.99% | seed 6773 三 epoch test AP：WIKI -0.02188、LASTFM -0.00017、WikiTalk -0.00213、Flight +0.00168 | **不继续作为主方案**。WIKI 回退不可接受 |
| 独立随机数 | `core-independent`，group size=1 | batch 时间 -6.74%；远程节点 -5.83%；远程边 +2.05% | test AP：WIKI -0.02452、LASTFM -0.01631、WikiTalk +0.00019、Flight +0.00158 | **停止**。既未修复精度，也失去边请求复用 |
| Attention correction 裁剪/关闭 | correction floor=0.1；或完全关闭 correction | 通信与 core-static 基本相同；batch 时间约 -7% | WIKI test AP：clipped -0.02068；uncorrected -0.01581。只能缩小、不能消除回退 | **停止扫 correction floor** |
| 提高采样概率 | \(p=0.15\) 和 \(p=0.20\) | p=0.15：远程节点 +16.49%、边 +29.20%；p=0.20：节点 +39.11%、边 +59.96% | 未显示值得用大量额外通信换取的质量证据 | **停止** |
| Newest top-1 保护 | 最新候选确定性保留，只采样 tail | WIKI 时间 -7.69%、节点 -5.49%、边 -6.31%；但 WikiTalk 节点 +4.29%、边 +10.09% | test AP：WIKI -0.02214、LASTFM -0.00238、WikiTalk -0.00235、Flight +0.00042 | **停止**。没有修复 WIKI，且部分数据集通信增加 |
| 协同 boundary-decay | 保留原 boundary-decay 概率；每 8 个 root 对相同 remote node 使用 keyed shared random | 完整训练四数据集平均：batch 时间 +0.22%、远程节点 -12.46%、远程边 -8.41% | 完整轮数 seed 6773 test AP Δ：WIKI -0.00289、LASTFM +0.00737、WikiTalk -0.00014、Flight -0.00054 | **继续推进，但按通信—精度权衡表述并补完整多 seed** |
| 真正 prediction-corrected residual aggregation | 未采样对象使用非零 \(\bar y_i\)，只拉取并校正残差 | **尚未实现和实测**；需要改变 MFG/fetch 数据布局 | **尚未实测** | **低优先级条件推进**：只有 predictor/cache 向量化后再单独立项，不能直接叠加到当前方案 |

### 3.1 协同采样的保留边界

30-batch、3-seed 的四数据集系统结果：

| 数据集 | batch 时间 | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: |
| WIKI | +0.32% ± 1.62% | -22.23% ± 0.07% | -1.64% ± 0.51% |
| LASTFM | -2.18% ± 1.05% | -2.09% ± 0.04% | -7.42% ± 0.21% |
| WikiTalk | -0.62% ± 3.77% | -6.21% ± 0.34% | -9.37% ± 0.10% |
| Flight | +0.59% ± 1.31% | -12.19% ± 0.19% | -14.44% ± 0.07% |

早期 3-epoch 质量配对均值 ± seed 间标准差：

| 数据集 | validation AP Δ | test AP Δ | validation AUC Δ | test AUC Δ |
| --- | ---: | ---: | ---: | ---: |
| WIKI | +0.00076 ± 0.00344 | +0.00210 ± 0.00382 | +0.00066 ± 0.00573 | +0.00101 ± 0.00675 |
| LASTFM | -0.00247 ± 0.00308 | +0.00631 ± 0.00718 | -0.00387 ± 0.00304 | +0.00058 ± 0.00759 |
| WikiTalk | -0.00017 ± 0.00106 | -0.00035 ± 0.00187 | -0.00003 ± 0.00125 | -0.00050 ± 0.00240 |
| Flight | +0.00100 ± 0.00264 | +0.00059 ± 0.00299 | +0.00074 ± 0.00262 | +0.00052 ± 0.00326 |

完整轮数 seed 6773 的主结果：

| 数据集 | epochs | test AP Δ | test AUC Δ | batch 时间 | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WIKI | 50 | -0.00289 | -0.00266 | -0.66% | -22.37% | -2.02% |
| LASTFM | 50 | +0.00737 | +0.00881 | -1.22% | -2.31% | -6.26% |
| WikiTalk | 10 | -0.00014 | +0.00007 | +1.68% | -12.31% | -10.51% |
| Flight | 10 | -0.00054 | -0.00038 | +1.08% | -12.85% | -14.86% |

它保持单个 occurrence 的边际采样概率，但改变组内联合分布，可能增加 gradient
covariance。完整训练显示 WIKI 存在 0.00289 的小幅 AP 下降，因此当前应表述为：

> 以受控相关性和小幅、数据集相关的精度变化，换取稳定的批内远程请求复用。

下一轮需要补充完整轮数多 seed、group size 1/4/8/16、
gradient/selection correlation 以及真正跨机环境的 time-to-target-AP。

## 4. Edge cache、负样本与请求执行

| 优化点 | 已尝试方案 | 效率变化 | 精度变化 | 推进结论 |
| --- | --- | --- | --- | --- |
| Probability-aware edge cache | 概率、通信字节、reuse 和方差收益联合评分，机会式插入 | 完整版 batch 时间 +200.4%、远程边 -5.54%，四数据集合计节省 59.27 MiB；object hit rate 1.81%–8.97% | 没有形成可抵消系统开销的质量收益 | **暂停当前实现** |
| Frozen/simple edge cache | 关闭动态替换和复杂 signal，隔离 lookup/admission 成本 | batch 时间仍 +68.67%；远程节点约 0%、远程边 -2.50% | 未进入质量验证 | **仅允许重写后再测**：必须使用 GPU/vectorized index 和批量离线 admission |
| Communication-aware hard negative | 原 localism proposal 生成候选，再按 hardness/通信代价重排并 importance correction | batch 时间 +238.15%；远程边仅 -0.29% | hardness 多接近 0.5；ESS 仅 3.79%–12.22%，梯度方差风险大 | **停止当前路线** |
| Cost-only negative | 去掉 embedding hardness，只按通信代价重排 | batch 时间 +7.22%；远程节点 -12.60%；远程边 -1.29% | 改变负样本梯度，未得到训练质量收益 | **停止**。节点数下降不足以补偿目标扰动 |
| Unified request planner | node/edge/version 分类、owner 分桶和请求去重统计 | 当前主要是 metrics-only；默认启用曾污染计时，尚无独立 wall-clock 收益 | 不改变模型精度 | **保留为测量基础设施**，不作为独立贡献 |
| Static/Dynamic 分阶段 fetch | static feature 提前拉取，memory updater 后 JIT 获取 memory/version | 尚未完整实现和独立实测 | 未实测 | **条件推进为系统优化**，优先在多机上证明关键路径收益 |
| 额外 memory-sync coalescing/预算 | 在 SSIM 外再增加周期聚合或独立同步预算 | 未证明收益，且可能增加 stale；与 SSIM 职责重叠 | 可能破坏现有 smooth 状态 | **不作为主要贡献** |
| Residual + cache/full CoRe | residual sampler、cache、hard negative 和 planner 组合 | residual+cache：batch 时间 +154.7%、节点 -10.70%、边 -12.35%；完整组合的组成模块均未通过门槛 | 没有可靠的组合质量收益 | **停止组合调参**，必须先让单模块独立过门槛 |

## 5. 分区、网络与数据加载

| 优化点 | 已尝试方案 | 效率变化 | 精度变化 | 推进结论 |
| --- | --- | --- | --- | --- |
| SPEED 与 METIS locality/负载权衡 | Flight 8 分区、固定 top-0.1，对比 direct endpoint locality 和训练边负载 | METIS direct endpoint remote ratio 0.00169、负载 CV 0.10987；SPEED remote ratio 0.02563、负载 CV 0.00008 | 未进行模型精度归因 | **作为 measurement 结果保留**，不能据此替换 partitioner |
| IB 与 Socket | StackOverflow、双机 8 rank、1 epoch | Socket 98.13s，IB 97.31s；Socket 仅慢 0.82s（0.84%） | 跳过 validation/test | **不优先优化网络传输**。结论仅适用于该配置 |
| GDELT 双机运行 | IB/Socket 对照尝试 | 数据准备阶段被终止，没有有效 epoch 时间 | 无 | **无结论**，不能写入收益表 |
| 数据加载 cache | mmap `edges_cache/*.npy`、`edges_cache.pt` 回退，并移除仅打印用的全局 `torch.unique` | 避免重复 pandas parse/global unique；尚无完整 GDELT epoch 加速数字 | 不改变训练语义 | **保留工程优化**，补齐准备时间和峰值内存基准后再写论文 |

## 6. 最终推进优先级

### P0：继续

1. **Coordinated boundary-decay，group=8**
   - 当前唯一具有稳定通信下降、且完整训练精度变化仍在小幅范围内的新增技术点；
   - 下一步补完整轮数多 seed、group-size sweep 和双机/多机验证；
   - 若 WIKI 多 seed 平均 AP 下降超过预设容忍值，则降级为系统消融而非主方法。
2. **原 all-smooth + SSIM=0.3**
   - 作为 baseline 机制保留；
   - 不再尝试用静态 burst/per-node gate 替换。

### P1：满足门槛后再决定

1. **Route-hot / memory-hot 离线 cost profiler**
   - 只做 profiler，不先实现在线切换；
   - 未达到两个数据集预计通信下降 10% 和 StackOverflow 关键路径 5% 门槛即停止。
2. **Static feature prefetch + dynamic memory JIT**
   - 只在多机关键路径中验证；
   - 必须报告真实字节、overlap 和 p95 batch latency。

### P2：需要重写，暂不投入主线

1. GPU/vectorized immutable edge cache；
2. 真正的非零 prediction-corrected residual aggregation。

二者只能在新实现消除当前 Python 热路径后重新进入四数据集门槛，不应继续在现有
代码上调 admission score、correction floor 或组合权重。

### 停止推进

- 动态热点迁移、租约和在线重划分；
- 固定 burst/per-node/runtime smooth gate；
- gamma 普通门、移除 normalization、修改 Filter source；
- core-static、top-1、提高 \(p\)、attention correction sweep；
- 当前 hard-negative、cost-only negative；
- 当前 Python edge cache；
- full CoRe 组合调参；
- 在 SSIM 外增加独立 memory-sync coalescing。

## 7. 一句话结论

当前最可信的期刊扩展不是同时叠加 residual、cache 和 hard negative，而是：

> 在固定 partition、固定 route-hot 和原始 boundary-decay 概率下，通过小组级
> 协同随机数减少批内唯一远程请求；其余方向必须先证明独立收益，不能与该方案
> 捆绑。
