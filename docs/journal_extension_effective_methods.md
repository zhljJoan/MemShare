# MemShare 期刊扩展：保留方法与备选方向

更新时间：2026-08-03

当前分支：`core-memshare`

本文只详细记录两类仍值得保留的期刊扩展方向：

1. 已经稳定降低远程对象数、并完成双机八卡三 seed 完整训练的 coordinated
   boundary-decay；
2. 在部分实验中显示出优化空间、但尚未形成稳定规则的 burst-aware smooth。

`all-smooth + SSIM=0.3` 是当前默认方法和所有实验的 memory baseline，不再作为
单独优化点展开，也不计入期刊新增贡献。除非特别说明，本文中的 smooth 策略实验
均保持 SSIM 设置不变。

| 方向 | 当前定位 | 已确认结果 | 尚未确认 |
| --- | --- | --- | --- |
| Coordinated boundary-decay，group=8 | 继续验证对象 | 双机八卡三次系统复测中训练窗口字节平均 -8.78%、mean batch -3.04%、p95 -1.16%；三 seed 完整训练关键路径平均 -3.69%，12/12 配对加速；test AP 平均 -0.0903 个百分点且 8/12 配对为负；大图 StackOverflow/GDELT 单 epoch -1.60%/+0.95% | GDELT route/launch 开销、group size、分层归因与更稳健的统计效率配置 |
| Burst-aware smooth | 备选方向 | WIKI 部分完整实验中 low/mid 分层优于 all，high-only 存在劣势 | 跨数据集统一规则，以及在保留 all-smooth 正收益时消除 high-burst 坏例 |

目前没有任何新增方法同时证明了“跨机训练显著加速”和“完整多 seed 精度不
退化”。下文严格区分已测结果、现象解释和待验证设计。

## 1. 优先验证：Coordinated Boundary-Decay Sampling

### 1.1 方法目标

对应采样策略：

```text
boundery_recent_decay_coordinated
```

它保持以下内容不变：

- 图划分、热点集合和 node/edge ownership；
- boundary-decay 的候选集合；
- 每个固定 root-neighbor occurrence 使用的原始概率 \(p_{r,i}\)；
- attention、memory 和模型结构。

它只改变不同 root 的采样随机数：

> 同一小组 root 在同一 batch/layer 中遇到相同 remote node 时，共享同一个
> keyed random number，使重复对象更倾向于同时被选中或同时不被选中，从而减少远程访问的节点数量。

目标不是直接降低 occurrence 级期望采样量，而是改变 occurrence 之间的相关性，
减少 MFG 去重后的唯一远程 node/edge 对象数。

### 1.2 原 boundary-decay 概率

对 root $r$ 在时刻 $t$ 的远程候选邻居 $i=(v,\tau_i)$，先计算时间衰减
权重：

$$
w_{r,i}
=
\exp\left(\frac{\tau_i-t}{\Delta_r}\right).
$$

若远程候选数为 $n_r^R$，当前实现使用：

$$
p_{r,i}
=
\operatorname{clip}
\left(
\beta
\frac{w_{r,i}}{\sum_{j\in N_r^R}w_{r,j}}
n_r^R,
1
\right).
$$

当前实验中 $\beta=0.1$。coordinated 版本完全复用 $p_{r,i}$

//还测了一些大模型给的方案包括residual、cache cost 或 hard-negative score但没测出来有效结果，这个是一个自方案。

### 1.3 小组共享随机数

按 root 在 batch 中的位置划分小组：

$$
g(r)
=
\left\lfloor
\frac{\operatorname{rootPosition}(r)}{G}
\right\rfloor,
\qquad G=8.
$$

对 remote node \(v\)，由以下 key 生成均匀随机数：

$$
U_{v,g,b,l}
=
\operatorname{UniformHash}
\left(
v,g,b,l,\operatorname{seed}
\right)
\in[0,1).
$$

当前 native 实现使用 SplitMix64，key 包含：

```text
(remote_node_id, root_group, batch_id, layer, seed)
```

每个 occurrence 的决策为：

$$
z_{r,i}
=
\mathbb I
\left[
U_{v,g(r),b,l}<p_{r,i}
\right].
$$

同一 remote node、同一 group、同一 batch 和同一 layer 共享 \(U\)；不同
group、batch 或 layer 使用不同随机数。

实现位置：

- `csrc/sampler/include/sampler.h`：native SplitMix64 和 coordinated
  sampling；
- `starrygl/sample/sample_core/neighbor_sampler.py`：采样策略入口；
- `examples/train_boundery.py`：`core_group_size` 配置。

### 1.4 为什么小概率 occurrence 没有被提升到最大概率

这里最容易混淆的是：

- **单条 occurrence 是否进入 MFG**：每条 occurrence 仍与自己的
  \(p_{r,i}\) 比较；
- **远程对象是否至少拉取一次**：对同组所有 occurrence 的选择结果取 OR。

算法并没有先计算 \(\max_r p_{r,i}\)，再让所有 occurrence 都与这个最大值比较。
对任意一条固定 occurrence：

$$
z_{r,i}
=
\mathbb I[U_{v,g,b,l}<p_{r,i}],
$$

所以在把 keyed hash 视为均匀随机数的条件下：

$$
\Pr(z_{r,i}=1)
=
\Pr(U_{v,g,b,l}<p_{r,i})
=
p_{r,i}.
$$

例如同一组中两条 occurrence 使用同一个 \(U\)，但各自概率为
\(p_A=0.1\) 和 \(p_B=0.4\)：

$$
z_A=\mathbb I[U<0.1],
\qquad
z_B=\mathbb I[U<0.4].
$$

| \(U\) 的范围 | occurrence A | occurrence B | 对象是否需要拉取 |
| --- | ---: | ---: | ---: |
| \(0\le U<0.1\) | 选中 | 选中 | 是 |
| \(0.1\le U<0.4\) | 不选 | 选中 | 是 |
| \(0.4\le U<1\) | 不选 | 不选 | 否 |

因此：

$$
\Pr(z_A=1)=0.1,
\qquad
\Pr(z_B=1)=0.4.
$$

小概率的 A 在 \(0.1\le U<0.4\) 时仍然不进入 MFG，并没有被 B 的较大概率
“带着选中”。只有“对象是否至少被任何 occurrence 使用”的概率变为：

$$
\Pr(z_A\lor z_B)
=
\max(0.1,0.4)
=
0.4.
$$

独立随机数下，同一对象至少出现一次的概率则是：

$$
1-(1-0.1)(1-0.4)
=
0.46.
$$

一般情况下，\(\max_r p_{r,i}\) 是对所有 occurrence 决策取 OR 后得到的**唯一
对象请求概率**，不是分配给每条 occurrence 的新采样阈值。如果错误地实现为：

$$
z'_{r,i}
=
\mathbb I
\left[
U<\max_j p_{j,i}
\right],
$$

那么上例中 A 的边际概率确实会从 0.1 提升到 0.4，采样分布就会改变；当前代码
不是这种实现。

实际代码链路为：

1. `csrc/sampler/include/sampler.h` 先为每条 occurrence 计算自己的
   `edge_probability`；
2. 每条 occurrence 独立执行
   `if (uniform >= edge_probability) continue`；
3. 只有通过自身阈值的 occurrence 才写入 `src_index/sample_nodes/eid`；
4. `starrygl/sample/batch_data.py` 后续只对 `dist_nid/dist_eid` 做数据拉取去重，
   DGL block 仍按已保留的 `src_index` 建边，不会把未选中的 occurrence 补回。

在这个条件下，occurrence 级期望采样量保持为：

$$
\mathbb E\left[\sum_{r,i}z_{r,i}\right]
=
\sum_{r,i}p_{r,i}.
$$

但这并不等于“采样完全无损”。若 \(p_A\le p_B\)，共享随机数会产生：

$$
\Pr(z_A=1,z_B=1)=p_A,
$$

$$
\operatorname{Cov}(z_A,z_B)
=
p_A-p_Ap_B
=
p_A(1-p_B)>0.
$$

独立采样时协方差为 0。也就是说，单条 occurrence 的一阶边际不变，但联合分布、
每批方差、完整 MFG 分布和训练梯度分布都会变化。

### 1.5 与执行路径的关系

```text
boundary-decay 计算原概率
        │
        ▼
root group 共享 keyed uniform
        │
        ▼
产生 occurrence 级邻居选择
        │
        ▼
non-redundant MFG 去重 node/edge ID
        │
        ▼
生成 dist_nid / dist_eid
        │
        ▼
node/edge AllToAll fetch
```

协同随机数使重复对象的选择更集中，因此去重后的 `dist_nid/dist_eid` 更少。但
当前执行层仍保持相同的 collective 调用次数、route construction、异步 prefetch
和 wait 点，所以对象数下降不一定直接转化为 wall-clock 加速。

### 1.6 三 seed 短程系统结果

环境为单机 4 × NVIDIA A40；每个数据集测 30 batch；seeds 为
6773/6774/6775。变化均为 coordinated 相对原 boundary-decay：

| 数据集 | batch 时间 | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: |
| WIKI | +0.32% ± 1.62% | -22.23% ± 0.07% | -1.64% ± 0.51% |
| LASTFM | -2.18% ± 1.05% | -2.09% ± 0.04% | -7.42% ± 0.21% |
| WikiTalk | -0.62% ± 3.77% | -6.21% ± 0.34% | -9.37% ± 0.10% |
| Flight | +0.59% ± 1.31% | -12.19% ± 0.19% | -14.44% ± 0.07% |
| 四数据集非加权平均 | -0.48% | -10.68% | -8.22% |

远程对象数的方向在四个数据集和三个 seed 上一致；时间变化落在短程波动范围内。

### 1.7 完整轮数结果

完整实验使用单机四卡、seed 6773；WIKI/LASTFM 为 50 epochs，
WikiTalk/Flight 为 10 epochs。主结果如下，精度变化单位均为百分点：

| 数据集 | test AP 变化 | test AUC 变化 | batch 时间 | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WIKI | -0.289 | -0.266 | -0.66% | -22.37% | -2.02% |
| LASTFM | +0.737 | +0.881 | -1.22% | -2.31% | -6.26% |
| WikiTalk | -0.014 | +0.007 | +1.68% | -12.31% | -10.51% |
| Flight | -0.054 | -0.038 | +1.08% | -12.85% | -14.86% |
| 四数据集非加权平均 | — | — | +0.22% | -12.46% | -8.41% |

每个 rank、每个 batch 的 edge-feature 节省量为：

| 数据集 | 节省量 |
| --- | ---: |
| WIKI | 39.7 KiB |
| LASTFM | 136.5 KiB |
| WikiTalk | 244.1 KiB |
| Flight | 458.0 KiB |

原始结果：

- `outputs/core_memshare_screen/coordinated-decay-full-epochs-seed6773/`
  `matrix_summary.json`；
- [CoRe-MemShare 实验记录](core_memshare_experiments.md)。

### 1.8 时间没有下降的原因

当前结果只能证明 payload/object count 下降，不能证明训练加速：

1. 每个 batch 的 node/edge AllToAll 次数没有减少；
2. fetch 使用异步 pipeline，减少的传输如果原本已被计算覆盖，就不会缩短关键
   路径；
3. 单机四卡下每批节省量较小，collective launch、route 和同步固定延迟不变；
4. keyed hash 为每个远程候选增加少量采样计算；
5. batch 时间还包含 forward/backward、optimizer、memory/mailbox 更新等未优化
   部分。

所以单机四卡平均 +0.22% 应判断为**时间持平**。

### 1.9 统计效率开销

在固定候选及其概率的条件下，每个 occurrence 的一阶边际概率不变；但同组决策
正相关，可能导致：

- batch 内有效独立随机性下降；
- gradient covariance 增加；
- 收敛速度和最终质量随 group size 改变。

双机八卡三 seed 完整训练中，test AP 在 12 个配对中 8 次为负、4 次为正；四数据集
非加权平均为 -0.0903 个百分点，但 WIKI/LASTFM 的 paired 标准差达到
0.9411/1.7911 个百分点。这说明结果既不是稳定精度回退，也不能解释成无损；
group=8 目前只是通信复用与相关性之间的经验折中。

多层采样还会放大这个边界。当前 key 包含 `cur_layer`，所以不同层不会直接复用
同一个随机数；但上一层的相关选择会改变下一层 frontier、节点重复度和候选集合。
因此只能说“给定某层某条候选 occurrence 时，其条件边际概率保持不变”，不能说
完整多层 MFG 的路径分布、子图分布或梯度分布保持不变。

### 1.10 双机八卡稳定性验证方案

现阶段观测应表述为：

> 单机四卡实验中，coordinated boundary-decay 的四数据集远程节点平均下降
> 12.46%、远程边平均下降 8.41%，但 batch 时间持平；同时存在数据集相关的精度
> 变化。该结果只说明它值得进入双机八卡稳定性验证，不代表该方法已经无损，也不
> 预设其为期刊主方法。

验证分为五部分；其中 group=8 的三 seed 完整训练，以及 seed 6773 的三次系统
重复均已完成；group-size、分层归因和 collective device/overlap 门槛继续保留。

#### A. 八卡实验矩阵

- 环境：双机 8 GPU、8 ranks，保持固定 partition、route-hot、ownership 和默认
  memory 链路；
- 数据集：WIKI/LASTFM 各 50 epochs，WikiTalk/Flight 各 10 epochs；
- 对照：原 independent boundary-decay 与 coordinated 严格配对，使用相同数据
  顺序、训练 seed 和配置；
- 稳定性：至少 3 个完整训练 seed；系统延时对每个配置在 warm-up 后至少重复
  3 次；
- group size：独立 baseline，以及 coordinated 的 1/4/8/16。注意
  `group_size=1` 仍可能让同一 root 内重复 remote node occurrence 共享随机数，
  因此不能直接把它当作 independent baseline。

#### B. 通信与关键路径延时

必须分别记录平均值、p50 和 p95，不能只报告总 batch 时间：

| 阶段 | 需要记录的指标 |
| --- | --- |
| sampler | sampling kernel/CPU wall time、候选 occurrence 数、选中 occurrence 数 |
| MFG/route | 去重、block 构建和 route construction 延时 |
| AllToAll launch | node/edge collective 次数、CPU enqueue/launch 延时 |
| AllToAll device | 实际 collective device duration、真实发送/接收字节 |
| fetch wait | node/edge feature 或 memory 在消费点暴露出的阻塞时间 |
| overlap | collective duration 中被计算覆盖的比例 |
| end-to-end | train-step、batch 和 epoch wall time，以及 p95 batch latency |

需要特别区分：

- `launch time`：CPU 发起 collective 的固定开销；
- `collective duration`：通信流上的实际执行时间；
- `fetch wait`：训练真正因数据未返回而阻塞的时间。

只有 `fetch wait` 或未被 overlap 覆盖的 collective 时间下降，才可能稳定缩短
epoch wall time。

#### C. 精度与收敛性

该方法不是完全无损采样，必须同时比较：

- test AP/AUC 的多 seed 均值、标准差和每个 seed 的配对变化；
- 每个 epoch 的 train loss、validation AP/AUC 曲线；
- 最佳 validation epoch、达到最佳值所需 wall time；
- time-to-target-AP/AUC；
- 固定 epoch 和固定 wall-time 两种预算下的质量；
- early-stop epoch、训练曲线方差和是否出现更慢或不稳定收敛。

不能只比较最终单点 AP，也不能把一个数据集上的正向波动抵消另一个数据集上的
回退。

#### D. 分层采样分析

为判断相关性来自哪一层，按 layer 分别统计：

- candidate/selected occurrence 数；
- remote node/edge 的重复度与 unique count；
- 真实远程字节、sampler 延时和 fetch wait；
- 按目标概率 \(p\) 分桶后的经验边际采样率；
- 同组 pairwise selection correlation/covariance；
- frontier size、节点时间戳重复度和下一层候选分布。

进一步做以下层级消融：

1. 所有层都 coordinated；
2. 只在 `layer 0` coordinated；
3. 只在 deeper layer coordinated；
4. 每层使用不同 group size \(G_l\)；
5. 只在重复度或远程字节较高的层启用 coordinated。

这类“层采”分析可以判断是否能把通信复用限制在收益最大的层，同时减少对完整
MFG 联合分布和收敛性的扰动。

#### E. 验证后的结果分类

八卡实验结束后再根据证据决定定位：

- unique object/byte 下降，但 fetch wait 和 epoch time 不变：仅作为通信量
  measurement；
- fetch wait 下降，但精度或收敛稳定回退：作为通信—统计效率 trade-off；
- 延时、精度和收敛结果随 seed 或层配置不稳定：继续做层级归因，不宣称稳定收益；
- 只有跨机关键路径和训练质量都形成稳定结果后，再讨论最终论文定位。

### 1.11 双机八卡结果

2026-08-03 在 `gpu07 + gpu06`、8 × NVIDIA A40 上完成 Socket-only 配对。NCCL
固定使用两端 10GbE `ens4f1np1`，显式关闭 IB；8-rank all-reduce 和训练日志均
显示 `Using network Socket`，未出现 `NET/IB`。

完整轮数为 WIKI/LASTFM 50 epochs、WikiTalk/Flight 10 epochs，seeds 为
6773/6774/6775。变化均为 coordinated 相对同 seed baseline；表中精度单位为
百分点，`±` 是三个 paired seed 差值的样本标准差：

| 数据集 | batch 时间 | 远程节点 | 远程边 | test AP Δ | test AUC Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| WIKI | -5.52% ± 2.06% | -23.84% ± 0.04% | -2.33% ± 0.03% | +0.2208 ± 0.9411 | +0.3146 ± 1.0649 |
| LASTFM | -2.09% ± 1.18% | -2.49% ± 0.01% | -6.35% ± 0.02% | -0.5564 ± 1.7911 | +0.0707 ± 1.7580 |
| WikiTalk | -1.97% ± 0.48% | -13.30% ± 0.02% | -11.34% ± 0.01% | -0.0060 ± 0.2541 | +0.0498 ± 0.3592 |
| Flight | -5.18% ± 0.13% | -11.99% ± 0.01% | -15.55% ± 0.02% | -0.0195 ± 0.0740 | -0.0244 ± 0.0302 |
| 四数据集非加权平均 | -3.69% | -12.90% | -8.89% | -0.0903 | +0.1027 |

三 seed 阶段计时的四数据集非加权平均变化为 sampler -7.49%、AllToAll route
-3.02%、launch -0.46%、暴露 fetch wait -6.58%。但 fetch wait 的绝对值仅约
`0.01–0.03 ms/wait`，说明通信大部分已被 pipeline 覆盖。

随后在相同 seed 下对四个数据集各做 3 次 100-batch 系统复测，并交替运行顺序。
网卡计数在训练窗口起止各等待 1 秒刷新；等待不计入训练时间。wire bytes 是两台
机器 `ens4f1np1` 的 RX+TX 合计除以 2，包含协议和 collective，不是 NCCL
payload。下表为三次 paired 百分比变化的均值 ± 样本标准差：

| 数据集 | mean batch | max-rank p95 | 训练窗口 wire bytes | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WIKI | -1.42% ± 0.69% | -0.60% ± 3.10% | -10.682% ± 0.002% | -23.96% | -2.60% |
| LASTFM | -0.66% ± 1.76% | -0.47% ± 5.04% | -3.842% ± 0.264% | -2.50% | -6.69% |
| WikiTalk | -2.48% ± 2.96% | -0.55% ± 4.37% | -8.099% ± 0.028% | -13.90% | -11.26% |
| Flight | -7.59% ± 3.39% | -3.02% ± 1.33% | -12.484% ± 0.004% | -12.09% | -15.76% |
| 四数据集非加权平均 | -3.04% | -1.16% | -8.78% | -13.11% | -9.08% |

12 个 paired comparison 中，wire bytes 为 12/12 下降，mean batch 为 10/12
下降，p95 为 8/12 下降。因此“通信字节稳定下降”已经得到比“尾延时稳定下降”
更强的证据。sampler 四数据集平均下降 6.58%，但 route/launch 分别变化
+0.07%/+0.43%，仍没有随对象数同比下降。

完整训练的 mean batch 在 12/12 个 paired seed 中下降；但 test AP 在 12 个
配对中 8 次为负、4 次为正，四个数据集均出现正负 seed 混合。以 baseline 最佳
validation AP 的 95% 为 target，四数据集 time-to-target 变化的非加权平均为
增加 10.12%，其中 WIKI/Flight 的 seed 间标准差分别为 31.07%/54.56%。该指标会被
10-epoch 数据集的一次 epoch 跳变放大，但足以说明收敛并未一致改善。因此当前能
确认跨机系统效率收益，不能说统计效率无损。

详细环境、绝对时间、逐阶段延时、收敛表和原始结果位置见
[双机八卡 10GbE 初步结果](coordinated_boundary_decay_8gpu_socket_results.md)。当前
仍缺 group-size sweep、collective device 时间、overlap 和分层采样分析；若继续
增加精度 seed，应在先选出更弱相关的 group/layer 配置后进行。

#### 大数据集单 epoch 耗时

StackOverflow 和 GDELT 各运行 1 个完整训练 epoch，不截断 batch；
为只测训练耗时，跳过 validation/test。两个 variant 均使用 CPU feature、
8-way partition、seed 6773 和 Socket-only 10GbE。

| 数据集 | batches/rank | epoch baseline → coord | epoch 变化 | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: | ---: | ---: |
| StackOverflow | 1,852 | 103.303 s → 101.650 s | -1.60% | -11.82% | -7.51% |
| GDELT | 5,357 | 276.743 s → 279.363 s | +0.95% | -19.20% | -15.63% |

阶段计时表明，StackOverflow 的 sampler、AllToAll route、launch 和暴露
fetch wait 分别变化 -2.06%、-4.81%、-10.66% 和 -13.38%；GDELT 则为
-2.40%、+19.20%、+17.55% 和 +1.83%。GDELT 的整轮网卡估算字节下降
14.79%，但 route/launch 增长，完整 epoch 没有加速。

因此大数据集将当前结论进一步收紧为：

> 通信对象减少已经跨数据集复现，但尚不能声称大图稳定加速。
> StackOverflow 只有 1.60% 的单次收益，GDELT 反而慢 0.95%。下一步
> 应先定位 GDELT 的 route/launch 上升，而不是继续只追求 unique object 数。

StackOverflow baseline 未记录 p95/NIC 计数，因此不用该组做 p95/真实字节
配对结论。GDELT 的 max-rank p95 从 68.784 ms 变为 68.481 ms
(-0.44%)。网卡字节为整轮接口计数，不是精确 NCCL payload。

## 2. 备选：Burst-Aware Smooth

### 2.1 实际尝试的方案

该方向不是按节点预测链接，而是先用训练时间流中每个节点的 inter-arrival
statistics 离线计算 burstiness \(B_v\)，再用节点所属分组决定其 event 是否使用
smooth memory。最终报告的 AP/AUC 仍是**事件级链接预测指标**。

历史实验实现位于 commit `c19f972`。固定阈值为：

$$
\operatorname{group}(v)
=
\begin{cases}
\mathrm{low}, & 0\le B_v<0.55,\\
\mathrm{mid}, & 0.55\le B_v<0.75,\\
\mathrm{high}, & B_v\ge 0.75.
\end{cases}
$$

缺少有效 burstiness 的节点不进入 low/mid/high-only 分组。对 eligible shared
node，先按默认链路得到：

$$
m_v^{\mathrm{smooth}}
=
m_v^{\mathrm{cur}}
+
\lambda_{\mathrm{hist}}
\left(
\bar m_v-m_v^{\mathrm{cur}}
\right).
$$

每个 hard policy 再使用固定掩码 \(h_v\)：

$$
m_v^{\mathrm{out}}
=
h_v m_v^{\mathrm{smooth}}
+
(1-h_v)m_v^{\mathrm{cur}}.
$$

| policy | \(h_v=1\) 的节点 |
| --- | --- |
| `all` | 所有 eligible shared node |
| `none` | 无 |
| `low_burst_only` | \(B_v<0.55\) |
| `mid_burst_only` | \(0.55\le B_v<0.75\) |
| `high_burst_only` | \(B_v\ge0.75\) |

因此，之前尝试的是**互斥分组 hard selection**，不是连续 gate，也不是在线预测
器。它只控制 smooth 的应用范围；SSIM 同步过滤、图划分、热点和 ownership 均不
改变。

### 2.2 WIKI/LASTFM 50-epoch 结果

实验为单机四卡、seed 6773；两个数据集均运行 50 epochs，并按 validation AUC
选择 epoch。变化均相对默认 `all`，单位为百分点。

#### WIKI

| policy | 最后 epoch smooth 应用比例 | Test AP | AP 变化 | Test AUC | AUC 变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all` | 100.00% | 0.962861 | 0 | 0.958058 | 0 |
| `none` | 0% | 0.963284 | +0.0423 | 0.958952 | +0.0894 |
| `low_burst_only` | 56.25% | 0.964819 | +0.1958 | 0.960024 | +0.1966 |
| `mid_burst_only` | 37.28% | 0.963966 | +0.1105 | 0.959377 | +0.1319 |
| `high_burst_only` | 4.42% | 0.961871 | -0.0990 | 0.957772 | -0.0286 |

这一次实验中，low/mid-only 同时高于 all 和 none，而只给 high-burst tail 使用
smooth 最差。它提供了“high-burst 区间可能包含不适合当前 smooth 的 event”这一
研究动机，但收益只有约 0.11–0.20 个 AP/AUC 百分点，且目前只是单 seed。

#### LASTFM

| policy | 最后 epoch smooth 应用比例 | Test AP | AP 变化 | Test AUC | AUC 变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `all` | 100.00% | 0.878139 | 0 | 0.889098 | 0 |
| `none` | 0% | 0.855381 | -2.2758 | 0.862400 | -2.6698 |
| `low_burst_only` | 29.98% | 0.860527 | -1.7612 | 0.866072 | -2.3026 |
| `mid_burst_only` | 39.76% | 0.856732 | -2.1407 | 0.866848 | -2.2250 |
| `high_burst_only` | 26.96% | 0.859535 | -1.8604 | 0.865052 | -2.4046 |

LASTFM 的结论相反：all-smooth 明显最好，任何互斥分层都会丢失其他分组中的
正收益。它直接否定了“统一关闭某一 burst 分组”的泛化结论。

原始汇总：

- [WIKI/LASTFM 50-epoch smooth policy](../outputs/pattern_ablation_logs/wiki_lastfm_50ep_smooth_summary.csv)。

### 2.3 WIKI 部分重复结果

seed 6774 已完成 WIKI 的 `all`、`none` 和 `low_burst_only` 50-epoch 对照；
该轮训练代码按 validation AP 选择 checkpoint：

| policy | Test AP | Test AUC | 相对 all 的 AP 变化 | 相对 all 的 AUC 变化 |
| --- | ---: | ---: | ---: | ---: |
| `all` | 0.965909 | 0.961300 | 0 | 0 |
| `none` | 0.961259 | 0.956010 | -0.4650 | -0.5290 |
| `low_burst_only` | 0.962782 | 0.957500 | -0.3127 | -0.3800 |

low-only 在该 seed 中仍高于 none（AP +0.1523、AUC +0.1490 个百分点），但低于
all。这说明 WIKI seed 6773 的 “low-only 超过 all” 尚未复现。seed 6774 的完整
low/mid/high 矩阵以及 LASTFM 重复尚未完成，不能将这部分结果写成多 seed 结论。

原始日志：

- `outputs/pattern_ablation_logs/repeat_50ep/`。

### 2.4 四数据集 2-epoch 初筛

下表给出各 hard policy 的 Test AUC；括号内为最后一轮 smooth 应用比例。实验只有
2 epochs、单 seed，只能用于筛选。

| 数据集 | none | all | low-only | mid-only | high-only | 该轮最好 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| WIKI | 0.877016 | 0.876365 | 0.880223 (55.97%) | 0.883111 (37.76%) | 0.873330 (4.52%) | mid-only |
| LASTFM | 0.786404 | 0.805926 | 0.782022 (30.20%) | 0.787079 (40.27%) | 0.786812 (26.86%) | all |
| Flight | 0.942558 | 0.945814 | 0.942095 (0%) | 0.942338 (0.12%) | 0.945974 (99.87%) | high-only |
| WikiTalk | 0.962339 | 0.964205 | 0.963111 (7.63%) | 0.962307 (12.81%) | 0.959503 (76.04%) | all |

需要特别注意：

- Flight 的 99.87% event occurrence 落在 high-burst 分组，`high-only` 几乎
  等价于 `all`，不能证明 high-burst 策略独立有效；
- WikiTalk 的 high-only 比 all 低 0.4702 个 AUC 百分点；
- WIKI 的 high-only 比 all 低 0.3035 个 AUC 百分点；
- 四个数据集的最优规则不一致。

原始汇总：

- [四数据集 burst 初筛](../outputs/pattern_ablation_logs/burst_strength_summary.csv)。

### 2.5 “high-burst 有坏例”的准确含义

现有证据应拆成两层：

1. same-state shadow 诊断确认 smooth 平均为正，但确实会破坏一部分原本预测正确的
   event：WIKI 为 290 个，LASTFM 为 2,522 个；
2. WIKI 的分层训练中 high-only 相对 all/low/mid 有劣势，WikiTalk 短训也出现
   high-only 劣势。

这两层结果提供了继续研究 high-burst 条件化 smooth 的动机，但**尚未证明 shadow
中的坏例就是 high-burst event 导致的**。已有 event-level 特征预测 smooth
收益/损害的时间 holdout AUC 只有 0.497–0.567，burstiness 单信号也较弱，不能把
静态相关性解释为可靠的 event gate。

固定阈值还会产生明显的数据集分布偏移：Flight 几乎全部落入 high 组，而 WIKI
只有约 4%–5% 的应用 occurrence 落入 high 组。同一组名在不同数据集上并不代表
相似的覆盖范围。

### 2.6 效率与统计开销

该 policy 的主要目标是精度，不是通信：

- 图划分、远程 fetch 和 SSIM 同步条件不变；
- hard mask 只改变已获取 memory 的融合方式；
- 虽然跳过部分 smooth 算术，但没有证据显示 epoch time 或通信字节下降；
- 离线 burstiness 表需要额外存储一个 per-node scalar，运行时只需一次索引和
  mask，计算开销预计很小，但尚无独立 profiler 数字。

统计风险比运行开销更重要：hard selection 会完整丢弃未选分组中已经确认存在的
smooth 正收益。LASTFM 的 2.23–2.40 个 AUC 百分点回退就是直接反例。

### 2.7 如果重启，应该尝试什么

目前没有已验证方案能够在消除 high-burst 坏例的同时保证保留 all-smooth 的全部
正收益。因此该方向只作为备选，不进入当前默认组合。

若后续重启，优先尝试“从 all-smooth 保守衰减 high tail”，而不是再次做互斥
low/mid/high-only：

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

$$
g_v=
\begin{cases}
1, & v\notin\text{high tail},\\
g_{\mathrm{high}}, & v\in\text{high tail},
\end{cases}
\qquad
g_{\min}\le g_{\mathrm{high}}\le1.
$$

这个设计只是一条低风险候选路线，不是已证明解法。建议约束为：

1. 用数据集内 quantile 定义 high tail，避免固定 0.55/0.75 在 Flight 上退化；
2. 保留 \(g_{\min}>0\)，不直接关闭 high-burst smooth；
3. 以 \(g=1\) 的 all-smooth 为正则化锚点，只允许小幅衰减；
4. 同时报告各 burst stratum 的 event AP/AUC、corrected/harmed event 和覆盖率；
5. 先独立验证，不与 coordinated sampling 组合调参。

完整验证仍需 WIKI/LASTFM 50 epochs、WikiTalk/Flight 10 epochs，并至少三个
seed。只有在不少于两个数据集上稳定优于 all-smooth、其余数据集无显著回退时，
才升级为正式候选；否则保留为 bad-case analysis。

### 2.8 当前结论

> Burst 分层已经观察到可研究的异质性，尤其是 WIKI/WikiTalk 的 high-burst
> 劣势；但互斥 hard policy 不具备跨数据集泛化性，也没有现成方法保证在抑制坏例
> 时保留其他 event 的 smooth 正收益。因此保留为备选方向，不替换默认
> all-smooth，不进入下一步 coordinated 八卡对照。

## 3. 八卡对照组合

本轮稳定性验证使用的系统组合为：

```text
固定 partition / 固定 route-hot / 固定 ownership
                    │
                    ▼
原 boundary-decay 概率
                    │
                    ▼
group=8 coordinated random
                    │
                    ▼
non-redundant MFG + 原异步 fetch
                    │
                    ▼
默认 all-smooth + SSIM=0.3
```

关键配置：

```text
model=TGN_large
memory_type=historical
shared_memory_ssim=0.3
sample_type=boundery_recent_decay_coordinated
probability=0.1
core_group_size=8
core_freeze_adaptive_parameters=true
```

burst-aware smooth 暂不加入该组合。应先单独完成保守 high-tail attenuation 的
多 seed、完整轮数验证，再决定是否组合。

## 4. 其他未保留方法

| 项目 | 当前结论 |
| --- | --- |
| 固定 per-node smooth gate | WIKI 后半段仅多正确 1 个 event，LASTFM 少正确 12 个，停止 |
| event-conditioned smooth gate | 离线 gate AUC 0.497–0.567，当前特征集停止 |
| online residual-aware sampling | batch 时间 +50.08%，暂停当前实现 |
| `core-static` | WIKI 三 epoch test AP 下降 2.188 个百分点，不作主方案 |
| probability-aware edge cache | 完整实现 batch 时间 +200.4%，暂停 |
| communication-aware hard negative | batch 时间 +238.15%，ESS 仅 3.79%–12.22%，停止 |
| unified request planner | 主要是 metrics-only，保留为测量基础设施 |
| static/dynamic 分阶段 fetch | 未完整实现，满足多机关键路径门槛后再考虑 |
| route-hot / memory-hot 解耦 | 仅有理论分析，暂无明确路线 |

## 5. 最终结论

当前工作安排分为两个层级：

1. **继续验证对象**：coordinated boundary-decay。三 seed 双机八卡完整训练中，
   四数据集关键路径平均缩短 3.69%，12/12 个配对均加速；三次系统复测的实测
   链路字节平均下降 8.78%。但 p95 仅平均下降 1.16%，GDELT 耗时增加 0.95%，
   test AP 8/12 为负且 time-to-target 不稳定。下一步先做 group size 与分层配置
   筛选，同时定位 route/launch 开销，再决定定位。
2. **备选方法**：burst-aware smooth。它揭示了 high-burst 条件下的潜在劣势，
   但现有 hard 分层不能保留其他分组的正收益；后续若继续，只尝试以 all-smooth
   为锚点的保守衰减，并先独立过完整实验门槛。

默认 `all-smooth + SSIM=0.3` 只作为实验基线保留，不再作为本文的独立技术点。
