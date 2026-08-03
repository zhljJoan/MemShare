# Coordinated Boundary-Decay 双机八卡 10GbE 初步结果

更新时间：2026-08-03

状态：**四个规定数据集的三 seed 完整训练、三次系统稳定性复测和大数据集
单 epoch 性能筛查均已完成；这是下一步验证结果，不是最终方法结论。**

## 1. 测试目的与边界

本轮只比较：

- baseline：`boundery_recent_decay`；
- coordinated：`boundery_recent_decay_coordinated`，`group_size=8`。

两者保持相同的固定 partition、固定 route-hot、固定 ownership、候选集合、
occurrence 级概率、模型与训练顺序。默认 memory 路径均为：

```text
model=TGN_large
memory_type=historical
shared_memory_ssim=0.3
probability=0.1
core_freeze_adaptive_parameters=true
seed=6773/6774/6775（完整训练）；6773（系统复测与大图筛查）
```

本轮不组合 residual sampler、edge cache、hard negative、burst-aware smooth 或动态
热点迁移。

## 2. 双机网络配置确认

| 项目 | gpu07 | gpu06 |
| --- | --- | --- |
| GPU | 4 × NVIDIA A40 | 4 × NVIDIA A40 |
| 10GbE 训练网卡 | `ens4f1np1` | `ens4f1np1` |
| 训练 IP | `192.168.1.107` | `192.168.1.106` |
| 网卡速率 | 10,000 Mb/s | 10,000 Mb/s |

显式设置：

```text
NCCL_IB_DISABLE=1
NCCL_NET=Socket
NCCL_SOCKET_IFNAME=ens4f1np1
GLOO_SOCKET_IFNAME=ens4f1np1
MASTER_ADDR=192.168.1.107
```

8-rank all-reduce 自检中，每个 rank 都得到期望和值 36。两端 NCCL 日志逐 rank
显示：

```text
Bootstrap : Using ens4f1np1:192.168.1.*
Using network Socket
... via NET/Socket/0
```

日志中未出现 `NET/IB`。因此下表是双机 8 卡、10GbE Socket-only 结果，不是单机
四卡，也不是 IB 结果。

## 3. 30-batch 系统筛选

变化均为 coordinated 相对 baseline：

| 数据集 | batch 时间 | 远程节点 | 远程边 | sampler | AllToAll route | launch | 暴露 fetch wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| WIKI | -1.09% | -23.93% | -2.78% | -10.73% | +34.01% | -2.92% | +27.54% |
| LASTFM | +0.34% | -2.50% | -6.70% | +1.88% | -0.75% | -3.77% | +20.74% |
| WikiTalk | -1.01% | -9.03% | -10.16% | -4.84% | -6.38% | -4.24% | -14.02% |
| Flight | -8.21% | -12.08% | -16.54% | -9.42% | -19.77% | -1.87% | -21.11% |
| 四数据集非加权平均 | -2.49% | -11.89% | -9.04% | -5.78% | +1.78% | -3.20% | +3.28% |

这里的 fetch wait 相对百分比容易误导：绝对值只有约
`0.01–0.03 ms/wait`，大部分 feature 通信已经被 pipeline 覆盖。远程对象下降
不会按比例映射到 batch 时间。

阶段计时只在显式开启 `--core_collect_phase_timing` 时运行。WIKI 30-batch 单次
对照中，开启计时为 52.258 ms/batch，关闭计时为 53.487 ms/batch。开启版本反而
快 2.30%，说明当前没有检测到可分辨的计时开销，但只有一次配对，不能把负开销
解释成收益；更合理的结论是开销小于该短跑的运行波动。

## 4. 100-batch 三次稳定性复测

为了补齐 p95 和网卡实测字节，四个规定数据集各执行 3 次 100-batch
baseline/coordinated 配对。3 次均使用 seed 6773，以隔离系统波动；这不是 3 个
训练 seed。运行顺序为：第 1、3 次 baseline → coordinated，第 2 次
coordinated → baseline。

字节统计在两台机器上各由 local-rank 0 读取 `ens4f1np1` 的 RX/TX 计数。开始与
结束快照前各保留 1 秒静默刷新，且这 2 秒位于训练计时窗口之外。表中 wire bytes
按以下方式折算，避免把同一链路流量在发送端和接收端重复计算：

$$
C_{\mathrm{wire}}
=
\frac{
\sum_{h\in\{\mathrm{gpu06},\mathrm{gpu07}\}}
\left(C_h^{\mathrm{RX}}+C_h^{\mathrm{TX}}\right)
}{2}.
$$

管理 SSH 走 `10.130.141.33`，不经过训练网卡。这里得到的是训练窗口内
`ens4f1np1` 的实测链路字节，包含 NCCL/Gloo、协议开销和同网卡其他流量，不是
NCCL payload 估计值。测试期间没有并发 GPU 作业。

下表绝对值为三次均值；变化为三次 paired 百分比变化的均值 ± 样本标准差：

| 数据集 | mean batch ms baseline → coord | mean batch Δ | max-rank p95 ms baseline → coord | p95 Δ | wire MiB/100 batches baseline → coord | wire Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WIKI | 40.543 → 39.968 | -1.42% ± 0.69% | 54.127 → 53.823 | -0.60% ± 3.10% | 2890.95 → 2582.13 | -10.682% ± 0.002% |
| LASTFM | 39.146 → 38.886 | -0.66% ± 1.76% | 44.459 → 44.263 | -0.47% ± 5.04% | 2830.33 → 2721.58 | -3.842% ± 0.264% |
| WikiTalk | 39.114 → 38.126 | -2.48% ± 2.96% | 43.942 → 43.678 | -0.55% ± 4.37% | 3117.24 → 2864.79 | -8.099% ± 0.028% |
| Flight | 50.202 → 46.347 | -7.59% ± 3.39% | 58.129 → 56.362 | -3.02% ± 1.33% | 4323.01 → 3783.34 | -12.484% ± 0.004% |
| 四数据集非加权平均 | — | -3.04% | — | -1.16% | — | -8.78% |

远程对象变化在相同 seed 的三次系统复测中完全一致：

| 数据集 | 远程节点 | 远程边 |
| --- | ---: | ---: |
| WIKI | -23.96% | -2.60% |
| LASTFM | -2.50% | -6.69% |
| WikiTalk | -13.90% | -11.26% |
| Flight | -12.09% | -15.76% |
| 四数据集非加权平均 | -13.11% | -9.08% |

四数据集共 12 个 paired comparison 中，wire bytes 为 12/12 下降，mean batch
为 10/12 下降，max-rank p95 为 8/12 下降。由此可以区分三个强度不同的结论：

1. **通信字节下降最稳定**：四个数据集三次复测都为正向；
2. **mean batch 有初步收益**：四数据集平均 -3.04%，但 LASTFM/WikiTalk 各有
   一次反向；
3. **p95 证据较弱**：仅 Flight 三次均下降，其余数据集存在反向重复。

阶段计时绝对值如下，单位为 `ms/call`，均为三次均值：

| 数据集 | sampler baseline → coord | route baseline → coord | launch baseline → coord | fetch wait baseline → coord |
| --- | ---: | ---: | ---: | ---: |
| WIKI | 18.672 → 17.591 | 1.757 → 1.765 | 0.1641 → 0.1636 | 0.0127 → 0.0130 |
| LASTFM | 25.456 → 24.272 | 1.188 → 1.218 | 0.1593 → 0.1643 | 0.0171 → 0.0150 |
| WikiTalk | 26.185 → 24.781 | 1.180 → 1.178 | 0.1604 → 0.1616 | 0.0193 → 0.0183 |
| Flight | 37.881 → 33.843 | 1.194 → 1.159 | 0.1619 → 0.1587 | 0.0213 → 0.0198 |

三次 paired 百分比变化的四数据集非加权平均为：sampler -6.58%、route
+0.07%、launch +0.43%、暴露 fetch wait -5.52%。route、launch 和 wait 的绝对值
很小且重复间方向不稳定；它们与 pipeline 互相重叠，不能相加成 batch 时间。

## 5. 规定完整轮数三 seed 结果

seeds 6773/6774/6775 均完成规定轮数：WIKI/LASTFM 各 50 epochs，
WikiTalk/Flight 各 10 epochs。精度取每次运行中 validation AP 最佳 epoch 对应的
test 指标；精度变化单位为百分点。下表的 `±` 是三个 paired seed 差值的样本
标准差。batch 时间与远程对象均取 `run_summary` 中覆盖全部已完成 epoch 的累计
结果，而不是 `result` 中最后一个 epoch 的快照。

| 数据集 | batch 时间 | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: |
| WIKI | -5.52% ± 2.06% | -23.84% ± 0.04% | -2.33% ± 0.03% |
| LASTFM | -2.09% ± 1.18% | -2.49% ± 0.01% | -6.35% ± 0.02% |
| WikiTalk | -1.97% ± 0.48% | -13.30% ± 0.02% | -11.34% ± 0.01% |
| Flight | -5.18% ± 0.13% | -11.99% ± 0.01% | -15.55% ± 0.02% |
| 四数据集非加权平均 | -3.69% | -12.90% | -8.89% |

完整训练的 12/12 个配对均缩短训练关键路径，远程对象变化也跨 seed 稳定。

精度汇总如下：

| 数据集 | validation AP Δ | validation AUC Δ | test AP Δ | test AUC Δ |
| --- | ---: | ---: | ---: | ---: |
| WIKI | +0.3321 ± 0.7495 | +0.4011 ± 0.8730 | +0.2208 ± 0.9411 | +0.3146 ± 1.0649 |
| LASTFM | +0.7184 ± 1.0043 | +0.6385 ± 1.0025 | -0.5564 ± 1.7911 | +0.0707 ± 1.7580 |
| WikiTalk | -0.0666 ± 0.1285 | -0.0459 ± 0.2004 | -0.0060 ± 0.2541 | +0.0498 ± 0.3592 |
| Flight | +0.0150 ± 0.0359 | -0.0048 ± 0.0203 | -0.0195 ± 0.0740 | -0.0244 ± 0.0302 |
| 四数据集非加权平均 | +0.2497 | +0.2472 | -0.0903 | +0.1027 |

四数据集均出现 test AP 正负 seed 混合，具体如下：

| 数据集 | seed | test AP Δ | test AUC Δ | baseline → coordinated 最佳 epoch |
| --- | ---: | ---: | ---: | ---: |
| WIKI | 6773 | -0.2883 | -0.1746 | 49 → 49 |
| WIKI | 6774 | -0.3562 | -0.4177 | 47 → 44 |
| WIKI | 6775 | +1.3068 | +1.5362 | 49 → 49 |
| LASTFM | 6773 | -0.5735 | +0.2815 | 42 → 47 |
| LASTFM | 6774 | +1.2432 | +1.7138 | 44 → 47 |
| LASTFM | 6775 | -2.3390 | -1.7832 | 43 → 46 |
| WikiTalk | 6773 | +0.2787 | +0.4507 | 9 → 9 |
| WikiTalk | 6774 | -0.2096 | -0.2430 | 9 → 9 |
| WikiTalk | 6775 | -0.0873 | -0.0582 | 9 → 9 |
| Flight | 6773 | -0.0346 | +0.0012 | 9 → 9 |
| Flight | 6774 | -0.0847 | -0.0577 | 9 → 9 |
| Flight | 6775 | +0.0610 | -0.0165 | 9 → 9 |

test AP 在 12 个配对中 8 次为负、4 次为正。四数据集均值接近 0 是正负结果抵消，
不能解释为无损；WIKI/LASTFM 尤其存在较大的 seed 间方差。

三 seed 的完整轮数阶段计时均值如下，单位为 host wall `ms/call`：

| 数据集 | sampler baseline → coord | route baseline → coord | launch baseline → coord | fetch wait baseline → coord |
| --- | ---: | ---: | ---: | ---: |
| WIKI | 18.352 → 16.276 | 1.672 → 1.409 | 0.1594 → 0.1545 | 0.0127 → 0.0119 |
| LASTFM | 23.923 → 22.332 | 1.166 → 1.157 | 0.1548 → 0.1548 | 0.0159 → 0.0143 |
| WikiTalk | 26.854 → 25.671 | 1.212 → 1.203 | 0.1539 → 0.1542 | 0.0211 → 0.0196 |
| Flight | 36.421 → 33.631 | 1.147 → 1.191 | 0.1552 → 0.1568 | 0.0216 → 0.0208 |

四数据集非加权平均 paired 变化为：sampler -7.49%、route -3.02%、launch
-0.46%、暴露 fetch wait -6.58%。这些 pipeline 阶段互相重叠，不能把各阶段时间
相加后当作 batch latency。

## 6. 大数据集单 epoch 耗时筛查

按后续验证要求，StackOverflow 和 GDELT 不做小 batch 截断，各运行
1 个完整训练 epoch。两组均跳过 validation/test，因此只用于系统耗时
和通信量筛查，不提供 AP/AUC 结论。大图特征固定放在 CPU，两个
variant 使用相同 seed、8-way partition 和阶段计时配置。

| 数据集 | batches/rank | epoch 时间 baseline → coord | ms/batch baseline → coord | epoch 变化 | 远程节点 | 远程边 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| StackOverflow | 1,852 | 103.303 s → 101.650 s | 55.779 → 54.887 | -1.60% | -11.82% | -7.51% |
| GDELT | 5,357 | 276.743 s → 279.363 s | 51.660 → 52.149 | +0.95% | -19.20% | -15.63% |

关键 rank 的 host-wall 阶段时间如下。各阶段有 pipeline 重叠，不能相加为
epoch 时间。

| 数据集 | sampler baseline → coord | AllToAll route baseline → coord | launch baseline → coord | fetch wait baseline → coord |
| --- | ---: | ---: | ---: | ---: |
| StackOverflow | 74.932 s → 73.392 s (-2.06%) | 16.258 s → 15.476 s (-4.81%) | 0.527 s → 0.470 s (-10.66%) | 0.139 s → 0.120 s (-13.38%) |
| GDELT | 218.875 s → 213.628 s (-2.40%) | 26.405 s → 31.474 s (+19.20%) | 1.220 s → 1.434 s (+17.55%) | 0.243 s → 0.247 s (+1.83%) |

GDELT 的 max-rank p95 batch latency 从 68.784 ms 变为 68.481 ms
(-0.44%)，但均值和完整 epoch 分别上升 0.95%。这说明 tail 没有恶化，
但也没有形成稳定的端到端加速。GDELT 两次运行在 `ens4f1np1` 上的
整轮估算 wire bytes 从 210.60 GiB 下降到 179.46 GiB (-14.79%)。该数值
来自整张网卡计数器，包含启动、协议开销和同期其他流量，不能当作精确的
NCCL payload。

StackOverflow baseline 在 p95/NIC 采集参数接入前已经启动，所以其
epoch、batch、远程对象和阶段计时可用，但不对缺失的 baseline p95/网卡
字节做配对推断。整个 launcher wall time 包含数分钟数据预处理且受页缓存
影响，也不用于训练加速结论。

大数据集结果对当前定位的影响是：

> coordinated boundary-decay 在 StackOverflow 和 GDELT 上仍能明显减少
> 远程对象，GDELT 还观测到网卡字节下降；但 epoch 时间分别为
> -1.60% 和 +0.95%。因此当前证据只支持“通信对象/字节减少”，不支持
> “在大数据集上稳定加速”。GDELT 中 route/launch 增长是下一步需要
> 定位的执行层开销。

## 7. 精度与收敛性

以每个 seed 的 baseline 最佳 validation AP 的 95% 作为 target，并用训练关键路径
估算 time-to-target。表中 epoch 是从 0 开始的记录编号；耗时计算使用
`epoch + 1` 个已完成 epoch。

| 数据集 | seed | target AP | baseline → coordinated 达标 epoch | 估算训练 time-to-target 变化 |
| --- | ---: | ---: | ---: | ---: |
| WIKI | 6773 | 0.89385 | 12 → 13 | +0.11% |
| WIKI | 6774 | 0.89107 | 10 → 10 | -3.18% |
| WIKI | 6775 | 0.89501 | 7 → 12 | +52.20% |
| LASTFM | 6773 | 0.63522 | 22 → 24 | +7.76% |
| LASTFM | 6774 | 0.62209 | 20 → 21 | +1.39% |
| LASTFM | 6775 | 0.64289 | 24 → 22 | -10.02% |
| WikiTalk | 6773 | 0.91545 | 1 → 1 | -2.08% |
| WikiTalk | 6774 | 0.91659 | 1 → 1 | -2.38% |
| WikiTalk | 6775 | 0.91618 | 1 → 1 | -1.45% |
| Flight | 6773 | 0.87510 | 1 → 1 | -5.18% |
| Flight | 6774 | 0.87396 | 0 → 0 | -5.05% |
| Flight | 6775 | 0.87500 | 0 → 1 | +89.38% |

| 数据集 | 三 seed time-to-target 变化均值 ± 标准差 |
| --- | ---: |
| WIKI | +16.38% ± 31.07% |
| LASTFM | -0.29% ± 9.01% |
| WikiTalk | -1.97% ± 0.48% |
| Flight | +26.38% ± 54.56% |
| 四数据集非加权平均 | +10.12% |

WIKI seed 6775 和 Flight seed 6775 的一批次/数个 epoch 差异会被低阈值放大，
因此 95%-target 本身是较粗的诊断指标；但它足以否定“更短 batch 必然带来更快
收敛”。该方法改变组内联合分布，统计效率仍需与系统效率一起报告。

## 8. 当前结论

当前允许的结论是：

> 在双机 8 卡、10GbE Socket-only、seed 6773 的三次 100-batch 系统配对中，
> group=8 coordinated boundary-decay 将四个规定数据集的训练窗口实测链路字节
> 平均降低 8.78%、mean batch 平均缩短 3.04%，但 p95 仅平均缩短 1.16% 且存在
> 反向重复。三 seed 完整训练中，远程节点平均降低 12.90%、远程边平均降低
> 8.89%，训练关键路径平均缩短 3.69%，且 12/12 个配对均加速。大数据集
> StackOverflow/GDELT 的远程节点分别下降 11.82%/19.20%，但 epoch
> 时间分别为 -1.60%/+0.95%。三 seed test AP 在 12 个配对中 8 次为负、4 次为
> 正，WIKI/LASTFM 的 paired 标准差分别为 0.94/1.79 个百分点，time-to-target
> 也未跨数据集一致改善。因此不能表述为完全无损、大图稳定加速或已经确定的
> 期刊主方法。

结论定位仍是“继续验证”，不是“保留为最终方案”。下一步门槛：

1. 完成 group size 1/4/8/16 与 independent baseline，判断统计方差是否随相关
   强度单调变化；
2. 优先尝试 layer-restricted coordination，检查能否保留通信收益并降低完整 MFG
   的相关扰动；
3. 补齐 StackOverflow baseline 的 p95/NIC 计数，并重复大图系统测试；
4. 定位 GDELT 中 AllToAll route/launch 上升，补齐 collective device
   duration 和 overlap；
5. 做 layer 级 candidate/selected occurrence、unique object、经验边际概率、
   selection correlation 和 frontier 变化分析。

## 9. 结果文件

有效结果位于：

```text
outputs/core_memshare_8gpu_socket/20260803/metrics/
outputs/core_memshare_8gpu_socket/20260803/logs/
outputs/core_memshare_8gpu_socket/20260803/launcher/
```

四数据集三次稳定性复测的有效标签统一为：

```text
baseline__<dataset>__100b__seed6773__settled_wire_r{1,2,3}
coordinated-decay__<dataset>__100b__seed6773__settled_wire_r{1,2,3}
```

早期 `wire_p95_r*` 使用未静默刷新的训练窗口 NIC 快照，窗口字节存在驱动刷新
抖动，不参与最终 wire bytes 表；其中 batch/p95/阶段计时本身未失效，但为了避免
混用，本文统一使用 `settled_wire_r*` 的全部系统指标。

大数据集本轮标签为：

```text
baseline__StackOverflow__1ep__seed6773__large1ep
coordinated-decay__StackOverflow__1ep__seed6773__large1ep
baseline__GDELT__1ep__seed6773__large1ep
coordinated-decay__GDELT__1ep__seed6773__large1ep
```

seed 6773 的有效完整轮数标签为：

```text
baseline__{WIKI,LASTFM}__50ep__seed6773__full
coordinated-decay__{WIKI,LASTFM}__50ep__seed6773__full
baseline__WikiTalk__10ep__seed6773__full
coordinated-decay__WikiTalk__10ep__seed6773__full
```

Flight 第一次 10-epoch baseline 在 epoch 3 收到外层 `SIGINT`，标签
`baseline__Flight__10ep__seed6773__full` 只保留为中断诊断，不参与任何表格。
有效 Flight 配对标签统一为 `full_retry1`。

seeds 6774/6775 的完整轮数标签统一为：

```text
baseline__<dataset>__<epochs>ep__seed{6774,6775}__full_multiseed
coordinated-decay__<dataset>__<epochs>ep__seed{6774,6775}__full_multiseed
```
