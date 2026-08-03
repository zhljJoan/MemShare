# MemShare 期刊扩展优化点、实验结果与推进决策

更新时间：2026-08-03

当前分支：`core-memshare`

本文综合以下两份记录：

- [阶段性调研发现](journal_extension_findings.md)：smooth aggregation、热点、
  分区、双机通信与数据加载；
- [CoRe-MemShare 四数据集实验](core_memshare_experiments.md)：正邻居采样、
  edge cache、negative sampling 与请求执行链。

优先验证对象和备选方向另见
[保留方法与备选方向详述](journal_extension_effective_methods.md)。该文件给出实际
计算流程、公式、实现边界和可复现实验结果。

目标是按优化点回答四个问题：

1. 实际尝试了什么；
2. 效率发生了什么变化；
3. 精度发生了什么变化；
4. 是否值得继续推进。

## 1. 结果口径

- 方法对比中的 AP、AUC 和 Accuracy 变化统一使用“百分点”，计算方式为
  \((\text{方法指标}-\text{baseline 指标})\times 100\)。单个模型的原始
  AP/AUC、gate 分类 AUC、比例和时间仍保留各自原始单位。
- CoRe 系统变化均相对同轮 baseline。第 3.1 节是历史单机 4 × NVIDIA A40、
  固定 4-way partition 结果；第 3.2 节是 2026-08-03 新增的双机 8 × A40、
  固定 8-way partition 结果，两者不能混算。
- coordinated boundary-decay 已完成 seeds 6773/6774/6775 的双机八卡完整轮数配对，
  并完成 WIKI/LASTFM/WikiTalk/Flight 各 3 次 100-batch 系统稳定性复测，
  并完成 StackOverflow/GDELT 各 1 个完整训练 epoch 的大数据集耗时
  筛查；NCCL 显式关闭 IB，并固定走两端 10GbE `ens4f1np1`。该结果
  仍缺 group-size sweep、collective device/overlap 和层级归因，不能作为最终
  方法结论。
- `30-batch、3 seeds` 用于判断通信和运行开销；`3-epoch、3 seeds` 只用于
  早期质量筛选，不能替代完整收敛实验。
- 三个完整训练 seed 均已完成：WIKI/LASTFM 50 epochs，WikiTalk/Flight
  10 epochs；主指标取 validation AP 最佳 epoch 对应的 test AP/AUC。
- `journal_extension_findings.md` 中已撤销的旧 remote ratio、remote coverage
  和 dynamic drift oracle 数值不进入本表。
- “继续”表示进入下一轮完整训练或多机验证；“条件推进”表示必须先满足列出的
  profiler/重构门槛；“备选”表示观察到研究空间但尚无稳定改法；“暂停”表示保留
  复现入口，但不继续组合调参。

## 2. Memory、smooth 与热点方向

| 优化点 | 已尝试方案 | 效率变化 | 精度变化 | 推进结论 |
| --- | --- | --- | --- | --- |
| 同状态 raw/smooth 机制诊断（非部署方案） | 同一模型、同一 event、同一更新前 memory/mailbox 状态分别计算 raw 和 smooth；shadow 分支不写回状态 | 只用于精度归因；额外 shadow forward 不属于部署路径，没有效率收益结论 | WIKI：AP +0.1148 个百分点、AUC +0.1379 个百分点；LASTFM：AP +0.1490 个百分点、AUC +0.1364 个百分点 | **证明现有 smooth 平均略有益**，但它不是独立训练方案或新增方法 |
| 全量 smooth + SSIM=0.3 | 所有 eligible shared node 使用 smooth；以 cosine distance 阈值 0.3 过滤 shared-memory 同步，即当前 `all_ssim03` 完整方案 | SSIM 实际控制进入同步 buffer 的对象；现有记录没有相对 full-sync 的四数据集独立字节和加速数值 | 大多数完整/较大实验中总体最稳定；上行 shadow 结果只证明 smooth 的即时作用，独立训练差异还包含训练轨迹 | **继续作为默认 baseline**，不重新包装成新贡献 |
| 固定 burst 分层 smooth | 按离线 burstiness 使用 low-only（<0.55）、mid-only（0.55–0.75）、high-only（≥0.75）hard policy，并与 all/none 比较 | 不改变 fetch、SSIM 同步或图划分；没有稳定通信或时间收益 | WIKI 50-epoch seed 6773：low/mid 相对 all 的 AUC 分别 +0.1966/+0.1319 个百分点，high-only -0.0286；但 seed 6774 low-only 相对 all -0.3800。LASTFM 的 all 比 low/mid/high 高 2.23–2.40 个 AUC 百分点；WikiTalk 2-epoch high-only 比 all -0.4702 | **保留为备选，不替换默认方法**。high-burst 有劣势和坏例研究动机，但现有互斥 hard policy 无法保证保留其他 event 的正收益；若重启，只尝试以 all-smooth 为锚点的保守 high-tail 衰减 |
| 固定 node-ID smooth gate（离线诊断） | 将时间流前半段的 event-level raw/smooth 胜负按唯一 shared endpoint 聚合，为每个节点固定选择 raw 或 smooth，再在后半段 event 上验证 | 不改变远程获取和同步；未实现独立运行时方案 | WIKI 比 all-smooth 多正确 1/10,574 个 event；LASTFM 少正确 12/104,765 个 event | **停止推进**。LASTFM 受影响节点主要为 mixed，WIKI 节点效果前后不稳定 |
| Event-level smooth-benefit gate 可预测性诊断 | 用 burstiness、remote ratio、staleness、memory novelty、方向一致性等 event 特征训练 logistic/tree，并按时间前后划分验证 | 只是离线分类诊断，尚未实现训练时 gate，也没有通信收益 | gate 分类 AUC 为 0.497–0.567，接近随机排序；这不是 TGNN 链接预测 AUC | **停止推进当前特征集** |
| Global gamma 轨迹 | 检查全模型共享的 memory 混合参数；event-conditioned adaptive soft gate 仅作为未实现设想 | 无直接效率收益；有效 history weight 只变化约 0.015–0.016 | 当前只有 gamma 轨迹和离线 gate 可预测性结果，没有实际 adaptive gate 的训练收益 | **暂停**。保留机制解释，不把未实现 gate 写成已尝试方法 |
| 在线动态热点/迁移 | 无，仅完成理论分析 | 在线改变热点集合会联动图划分、edge route 和 node ownership；还需要迁移 feature、memory、mailbox 与路由元数据，并承担版本一致性、同步屏障和迁移通信开销 | 无训练精度实验 | **仅保留理论判断**，暂时没有明确后续路线 |
| <span style="color:#8a8f98">Cross-count 静态热点（仅分析）</span> | <span style="color:#8a8f98">plain METIS 下比较 degree top-K 与跨分区边计数 top-K</span> | <span style="color:#8a8f98">Flight：degree/current 热点覆盖约 90.5%–90.6% cross-edge endpoint；cross-count 上限约 94.8%，只有约 4.2 个百分点空间</span> | <span style="color:#8a8f98">未进行训练精度验证</span> | <span style="color:#8a8f98">仅保留分析记录，暂时没有明确后续路线</span> |
| <span style="color:#8a8f98">Route-hot / memory-hot 解耦（仅分析）</span> | <span style="color:#8a8f98">理论上保持 partition/edge route 不变，只在 route-hot 内区分 FullShare 与 FeatureOnly</span> | <span style="color:#8a8f98">尚未实测；可能减少 memory replica、同步字节和 GPU memory，但没有 profiler 证据</span> | <span style="color:#8a8f98">没有训练质量结果</span> | <span style="color:#8a8f98">仅保留分析记录，暂时没有明确后续路线</span> |

<span style="color:#8a8f98">灰色条目只表示已有离线或理论分析，不代表候选方法、推进承诺或明确的后续实验路线。</span>

## 3. 正邻居采样方向

| 优化点 | 已尝试方案 | 效率变化 | 精度变化 | 推进结论 |
| --- | --- | --- | --- | --- |
| 在线 residual-aware sampling | last-fetched residual、age、edge error 和通信代价共同调整 \(p_i\) | batch 时间 +50.08%；远程节点 -10.64%；远程边 -6.69% | 当前实现仍是 zero-baseline probability correction，没有完整非零预测残差质量证据 | **暂停当前实现**。通信方向有效，但 Python/CPU 在线更新不划算 |
| Static communication-aware sampling (`core-static`) | 关闭 predictor/cache，只保留 native cost-aware probability 和 group randomness | 清理统计开销后：batch 时间 -6.40%、远程节点 -9.76%、远程边 -5.99% | seed 6773 三 epoch test AP：WIKI -2.188 个百分点、LASTFM -0.017 个百分点、WikiTalk -0.213 个百分点、Flight +0.168 个百分点 | **不继续作为主方案**。WIKI 回退不可接受 |
| 独立随机数 | `core-independent`，group size=1 | batch 时间 -6.74%；远程节点 -5.83%；远程边 +2.05% | test AP：WIKI -2.452 个百分点、LASTFM -1.631 个百分点、WikiTalk +0.019 个百分点、Flight +0.158 个百分点 | **停止**。既未修复精度，也失去边请求复用 |
| Attention correction 裁剪/关闭 | correction floor=0.1；或完全关闭 correction | 通信与 core-static 基本相同；batch 时间约 -7% | WIKI test AP：clipped -2.068 个百分点；uncorrected -1.581 个百分点。只能缩小、不能消除回退 | **停止扫 correction floor** |
| 提高采样概率 | \(p=0.15\) 和 \(p=0.20\) | p=0.15：远程节点 +16.49%、边 +29.20%；p=0.20：节点 +39.11%、边 +59.96% | 未显示值得用大量额外通信换取的质量证据 | **停止** |
| Newest top-1 保护 | 最新候选确定性保留，只采样 tail | WIKI 时间 -7.69%、节点 -5.49%、边 -6.31%；但 WikiTalk 节点 +4.29%、边 +10.09% | test AP：WIKI -2.214 个百分点、LASTFM -0.238 个百分点、WikiTalk -0.235 个百分点、Flight +0.042 个百分点 | **停止**。没有修复 WIKI，且部分数据集通信增加 |
| 协同 boundary-decay | 保留每条 occurrence 原 boundary-decay 概率；每 8 个 root 对相同 remote node 使用 keyed shared random，但仍各自与自身概率比较 | 中小四数据集双机八卡三 seed 完整训练平均：batch -3.69%、远程节点 -12.90%、远程边 -8.89%，12/12 配对加速。大图单 epoch：StackOverflow 时间 -1.60%、GDELT +0.95%，远程节点分别 -11.82%/-19.20% | 三 seed test AP 四数据集平均 -0.0903 个百分点，12 个配对中 8 次为负、4 次为正，且每个数据集都有正负 seed；大图单 epoch 跳过评估；联合及多层 MFG 分布并非无损 | **继续验证，不定为最终方案**。系统收益成立，但统计效率不稳定、大图未形成一致加速；先做 group size/分层配置，再定位 route/launch 与补充统计验证 |
| 真正 prediction-corrected residual aggregation | 未采样对象使用非零 \(\bar y_i\)，只拉取并校正残差 | **尚未实现和实测**；需要改变 MFG/fetch 数据布局 | **尚未实测** | **低优先级条件推进**：只有 predictor/cache 向量化后再单独立项，不能直接叠加到当前方案 |

### 3.1 协同采样的单机四卡历史结果与验证设计

本节以下旧结果均来自**单机四卡**；新的双机八卡结果单列在第 3.2 节。
30-batch、3-seed 的四数据集系统结果：

| 数据集 | batch 时间 | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: |
| WIKI | +0.32% ± 1.62% | -22.23% ± 0.07% | -1.64% ± 0.51% |
| LASTFM | -2.18% ± 1.05% | -2.09% ± 0.04% | -7.42% ± 0.21% |
| WikiTalk | -0.62% ± 3.77% | -6.21% ± 0.34% | -9.37% ± 0.10% |
| Flight | +0.59% ± 1.31% | -12.19% ± 0.19% | -14.44% ± 0.07% |

早期 3-epoch 质量配对均值 ± seed 间标准差：

| 数据集 | validation AP Δ（百分点） | test AP Δ（百分点） | validation AUC Δ（百分点） | test AUC Δ（百分点） |
| --- | ---: | ---: | ---: | ---: |
| WIKI | +0.076 ± 0.344 | +0.210 ± 0.382 | +0.066 ± 0.573 | +0.101 ± 0.675 |
| LASTFM | -0.247 ± 0.308 | +0.631 ± 0.718 | -0.387 ± 0.304 | +0.058 ± 0.759 |
| WikiTalk | -0.017 ± 0.106 | -0.035 ± 0.187 | -0.003 ± 0.125 | -0.050 ± 0.240 |
| Flight | +0.100 ± 0.264 | +0.059 ± 0.299 | +0.074 ± 0.262 | +0.052 ± 0.326 |

完整轮数 seed 6773 的主结果：

| 数据集 | epochs | test AP Δ（百分点） | test AUC Δ（百分点） | batch 时间 | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WIKI | 50 | -0.289 | -0.266 | -0.66% | -22.37% | -2.02% |
| LASTFM | 50 | +0.737 | +0.881 | -1.22% | -2.31% | -6.26% |
| WikiTalk | 10 | -0.014 | +0.007 | +1.68% | -12.31% | -10.51% |
| Flight | 10 | -0.054 | -0.038 | +1.08% | -12.85% | -14.86% |

四数据集平均 batch 时间为 +0.22%，各数据集变化范围为 -1.22% 至 +1.68%；
相对于短跑实验中 1.05%–3.77% 的 seed 间时间波动，该变化应判断为**时间持平**，
不能解释为稳定加速或稳定减速。

按完整轮数计数折算，每个 rank、每个 batch 省下的 edge-feature 数据为：

| 数据集 | edge-feature 节省量 / rank / batch |
| --- | ---: |
| WIKI | 39.7 KiB |
| LASTFM | 136.5 KiB |
| WikiTalk | 244.1 KiB |
| Flight | 458.0 KiB |

这些数字只统计 edge-feature bytes，不包含 node feature/memory。当前训练时间没有
同步下降，主要原因是：

1. 对固定候选 occurrence，代码仍使用其自身的 \(p_{r,i}\) 与共享随机数比较；
   \(\max_r p_{r,i}\) 只是所有 occurrence 决策取 OR 后的唯一对象请求概率，不是
   回填给小概率 occurrence 的阈值。每个 batch 的 node/edge AllToAll 调用次数
   没有减少。
2. feature fetch 使用异步 pipeline。传输只要在真正的 wait 点前完成，减少 payload
   只会增加 overlap 余量，不会缩短关键路径。
3. 单机四卡下每批省下的数据量较小，集合通信启动、路由和同步等固定延迟不变。
4. keyed shared random 为每个远程候选增加 hash 计算，该采样开销可能抵消一部分
   机内传输收益。
5. 当前 batch 时间包含采样、forward/backward、optimizer 和 memory/mailbox 更新；
   其中大部分没有被该策略优化。

这种边际保持只对固定候选及其概率成立。方法改变组内联合分布，可能增加
gradient covariance；上一层相关选择还会改变下一层 frontier，因此完整多层 MFG
分布并不保持。完整训练显示 WIKI 存在 0.289 个百分点的小幅 AP 下降，所以该方法
不能表述为完全无损。

> 以受控相关性和小幅、数据集相关的精度变化，换取稳定的批内远程请求量下降；
> 单机四卡下尚未得到端到端训练加速。

双机 8 卡验证清单及当前状态如下，不预设为期刊主方法：

1. 四数据集完整轮数 paired baseline/coordinated：seeds 6773/6774/6775 均已
   完成；三 seed test AP 的方向不稳定，后续统计重复应在 group/layer 配置筛选后做；
2. group=8 的 3 次系统重复已完成；independent baseline 和 group size
   1/4/16 尚未完成；
3. sampler、MFG/route、AllToAll launch、暴露 fetch wait、训练窗口网卡字节、
   batch/epoch wall time 和 p95 latency 已记录；collective device duration 与
   overlap 尚未完成；
4. 记录完整 AP/AUC/loss 曲线、多 seed 精度变化、time-to-target-AP/AUC、最佳
   epoch 和固定 wall-time 下的质量，判断收敛速度及稳定性变化；
5. 增加分层采样分析：逐层记录 occurrence、重复度、unique object、远程字节、
   经验边际概率、selection correlation、frontier size 和 fetch wait，并比较
   layer 0 only、deeper layer only、all layers 及 layer-specific group size。

完整概率解释和八卡测量表见
[保留方法与备选方向详述](journal_extension_effective_methods.md)。

### 3.2 双机八卡 10GbE 结果

环境为 `gpu07 + gpu06`、8 × NVIDIA A40、固定 8-way partition；NCCL 设置
`NCCL_IB_DISABLE=1`、`NCCL_NET=Socket`，并固定
`NCCL_SOCKET_IFNAME=ens4f1np1`。8-rank 自检和训练日志均确认
`NET/Socket`，未出现 `NET/IB`。

seeds 6773/6774/6775 均完成规定完整轮数。变化为 coordinated 相对同 seed
baseline；精度变化单位为百分点，`±` 是三个 paired seed 差值的样本标准差：

| 数据集 | epochs | batch 时间 | 远程节点 | 远程边 | test AP Δ | test AUC Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WIKI | 50 | -5.52% ± 2.06% | -23.84% ± 0.04% | -2.33% ± 0.03% | +0.2208 ± 0.9411 | +0.3146 ± 1.0649 |
| LASTFM | 50 | -2.09% ± 1.18% | -2.49% ± 0.01% | -6.35% ± 0.02% | -0.5564 ± 1.7911 | +0.0707 ± 1.7580 |
| WikiTalk | 10 | -1.97% ± 0.48% | -13.30% ± 0.02% | -11.34% ± 0.01% | -0.0060 ± 0.2541 | +0.0498 ± 0.3592 |
| Flight | 10 | -5.18% ± 0.13% | -11.99% ± 0.01% | -15.55% ± 0.02% | -0.0195 ± 0.0740 | -0.0244 ± 0.0302 |
| 四数据集非加权平均 | — | -3.69% | -12.90% | -8.89% | -0.0903 | +0.1027 |

三 seed 阶段计时平均变化为 sampler -7.49%、AllToAll route -3.02%、launch
-0.46%、暴露 fetch wait -6.58%。fetch wait 绝对值仅约 `0.01–0.03 ms/wait`，因此主要
收益并不是等待时间按远程对象数同比下降。

相同 seed 下进一步完成每数据集 3 次 100-batch 配对，并交替运行顺序。训练窗口
网卡计数在起止各静默刷新 1 秒，等待不计入训练时间。wire bytes 为两台机器
`ens4f1np1` 的 RX+TX 合计除以 2；它包含 collective 和协议开销，不是 NCCL
payload。结果为三次 paired 变化的均值 ± 样本标准差：

| 数据集 | mean batch | max-rank p95 | 训练窗口 wire bytes | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WIKI | -1.42% ± 0.69% | -0.60% ± 3.10% | -10.682% ± 0.002% | -23.96% | -2.60% |
| LASTFM | -0.66% ± 1.76% | -0.47% ± 5.04% | -3.842% ± 0.264% | -2.50% | -6.69% |
| WikiTalk | -2.48% ± 2.96% | -0.55% ± 4.37% | -8.099% ± 0.028% | -13.90% | -11.26% |
| Flight | -7.59% ± 3.39% | -3.02% ± 1.33% | -12.484% ± 0.004% | -12.09% | -15.76% |
| 四数据集非加权平均 | -3.04% | -1.16% | -8.78% | -13.11% | -9.08% |

wire bytes 在 12/12 个配对中下降，mean batch 在 10/12 中下降，p95 仅在 8/12
中下降。通信量证据稳定，但尾延时仍不是稳定收益。

完整训练 mean batch 在 12/12 个配对中下降，但 test AP 在 12 个配对中 8 次为
负、4 次为正，而且四个数据集均出现正负 seed 混合。四数据集 test AP 平均变化
为 -0.0903 个百分点，是正负结果抵消；WIKI/LASTFM 的 paired 标准差分别达到
0.9411/1.7911 个百分点。95%-AP time-to-target 的四数据集非加权平均增加
10.12%，且 seed 间变化很大。所以当前证据支持“跨机训练关键路径加速”，不支持
“完全无损”“收敛稳定改善”或“已经确定为主方法”。详细结果见
[双机八卡 10GbE 初步结果](coordinated_boundary_decay_8gpu_socket_results.md)。

大数据集进一步运行 1 个完整训练 epoch，跳过 validation/test：

| 数据集 | batches/rank | epoch baseline → coord | epoch 变化 | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: | ---: | ---: |
| StackOverflow | 1,852 | 103.303 s → 101.650 s | -1.60% | -11.82% | -7.51% |
| GDELT | 5,357 | 276.743 s → 279.363 s | +0.95% | -19.20% | -15.63% |

StackOverflow 的 sampler/route/launch/fetch wait 关键 rank 时间分别变化
-2.06%/-4.81%/-10.66%/-13.38%。GDELT 分别变化
-2.40%/+19.20%/+17.55%/+1.83%。GDELT 的整轮网卡估算 wire bytes 下降
14.79%，但 route/launch 上升与 epoch 时间未下降同时出现。这进一步
限定了当前结论：远程对象和字节减少是真实信号，但尚不是大图稳定
端到端加速。

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
| GDELT 双机运行 | 8 rank、10GbE Socket-only、baseline/coordinated 各 1 个完整训练 epoch | baseline 276.743s，coordinated 279.363s (+0.95%)；远程节点 -19.20%、远程边 -15.63%、整轮网卡估算字节 -14.79% | 跳过 validation/test | **有通信量收益，无稳定耗时收益**。定位 route/launch 上升后再决定是否继续 |
| 数据加载 cache | mmap `edges_cache/*.npy`、`edges_cache.pt` 回退，并移除仅打印用的全局 `torch.unique` | 已支持 GDELT 1.91 亿边在双机 8 rank 上完成整个 epoch；本轮没有 cache 开/关配对，不宣称加速幅度 | 不改变训练语义 | **保留为必要工程支持**，若写入论文仍需补 cache 开/关准备时间和峰值内存基准 |

## 6. 最终推进优先级

`all-smooth + SSIM=0.3` 继续作为默认 baseline，不作为独立优化点参与优先级排序。

### P0：下一步验证

1. **Coordinated boundary-decay，group=8**
   - 双机八卡三 seed 完整训练的 mean batch 在 12/12 配对中下降，三次系统复测
     的 wire bytes 在 12/12 配对中下降；
   - p95 只在 8/12 系统配对中下降，GDELT epoch 时间 +0.95%；test AP 则在
     12 个完整配对中 8 次为负、4 次为正，因此不能声称统计效率无损；
   - 下一步先做 group-size sweep 和 layer-restricted coordination，寻找更弱相关
     且保留通信收益的配置，同时定位 GDELT route/launch 上升，再决定最终定位。

### P1：满足门槛后再决定

1. **Burst-aware conservative smooth（备选）**
   - 现有 low/mid/high-only 结果只证明分组异质性，不证明统一 hard gate；
   - 若重启，从 all-smooth 出发，只对数据集内 high-burst tail 做有下界的小幅
     衰减，不能直接关闭；
   - 必须独立完成 WIKI/LASTFM 50 epochs、WikiTalk/Flight 10 epochs 和至少三个
     seed；不少于两个数据集稳定改善且其余无显著回退，才进入主方案。
2. **Static feature prefetch + dynamic memory JIT**
   - 只在多机关键路径中验证；
   - 必须报告真实字节、overlap 和 p95 batch latency。

### 仅分析，暂无明确后续路线

- <span style="color:#8a8f98">在线动态热点/迁移；</span>
- <span style="color:#8a8f98">Cross-count 静态热点；</span>
- <span style="color:#8a8f98">Route-hot / memory-hot 解耦。</span>

### P2：需要重写，暂不投入主线

1. GPU/vectorized immutable edge cache；
2. 真正的非零 prediction-corrected residual aggregation。

二者只能在新实现消除当前 Python 热路径后重新进入四数据集门槛，不应继续在现有
代码上调 admission score、correction floor 或组合权重。

### 停止推进

- 固定 per-node smooth gate；
- event-conditioned adaptive smooth gate；
- core-static、top-1、提高 \(p\)、attention correction sweep；
- 当前 hard-negative、cost-only negative；
- 当前 Python edge cache；
- full CoRe 组合调参；
- 在 SSIM 外增加独立 memory-sync coalescing。

## 7. 一句话结论

下一步优先工作不是继续叠加 residual、cache 和 hard negative，而是：

> coordinated sampling 在双机八卡三次系统复测中将四数据集训练窗口网卡字节
> 平均降低 8.78%、mean batch 平均降低 3.04%，但 p95 仅平均降低 1.16% 且
> 存在反向重复；StackOverflow/GDELT 单 epoch 时间分别为 -1.60%/+0.95%，且
> 三 seed 完整训练的关键路径平均缩短 3.69%、12/12 配对加速，但 test AP
> 8/12 为负，四数据集平均 -0.0903 个百分点，time-to-target 也不稳定。当前
> 定位是“通信量和中小图关键路径稳定有效，尾延时、大图和统计效率尚未同时
> 稳定”；先做 group size 和逐层相关性筛选，并定位 GDELT route/launch 开销，
> 再决定方法定位。
> burst 分层继续只保留为独立备选。
