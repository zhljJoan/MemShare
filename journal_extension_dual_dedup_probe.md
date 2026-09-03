# Sampler-Native Block-Unique Compact Execution 调研记录

## 当前目标

评估训练侧是否存在“相同 UID、不同 timestamp/occurrence”带来的重复计算空间，并收敛出能够在双机八卡稳定生效的统一执行方案。

本文按时间记录统计探针、局部原型和端到端 A/B。所有实验后端均由默认关闭的开关控制；文末结论代表当前判断，前面的阶段性结论只保留为实验过程。只包含最终保留链路的干净方案说明见 [最终方案设计](journal_extension_block_unique_final_method.md)。

最终保留的实验组合是：

```text
Sampler-native hierarchical compaction（仅训练）
                    ↓
direct COO Block + 缓存 CSC segment plan
                    ↓
Fused Segment Attention（materialized occurrence K/V）
                    +
gather-pointwise Compact GRU
```

对应开关为：

```text
--hierarchical_compaction on
--fused_segment_attention on
--compact_gru_update on
--compact_kv_projection off
--triton_edge_projection off
```

P0 compaction 只对训练 sampler 开启；evaluation sampler 继续使用 legacy compaction。Fused Segment 与 Compact GRU 在训练和评估中都保持开启。Projected K/V、Triton edge projection、8-warps 和 K/V 公共输入物化均因端到端门槛未通过，不进入最终组合。

## 早期探针实现（历史）

- `starrygl/sample/dual_dedup_view.py`
  - 记录 block 内 state occurrence 行数；
  - 记录 compact 后 object node 行数；
  - 记录 edge occurrence 行数；
  - 记录 compact 后 object edge 行数。
- `examples/train_boundery.py`
  - 新增 `--dual_dedup_view on/off`；
  - 新增 `--max_epoch`；
  - 新增 `--skip_eval`，用于短跑 profile。
- `csrc/fused_edge_projection.cu`
  - 新增独立 fused forward 原型；
  - 融合 `state[edge_src]`、`edge[edge_to_eid]`、`time` 的 gather/concat/linear；
  - 当前只做数值验证，尚未接入训练主链路。

## WIKI 4 卡 1 epoch 探针结果

命令配置：

```text
dataset: WIKI
GPU: 4
epoch: 1
sample_type: boundery_recent_decay
probability: 0.1
memory_type: historical
shared_memory_ssim: 0.3
dual_dedup_view: on
skip_eval: on
```

结果：

| 指标 | 数值 |
| --- | ---: |
| state_rows | 580153 |
| object_node_rows | 66287 |
| edge_occurrence_rows | 4284775 |
| object_edge_rows | 325717 |
| state_per_object_node | 8.75 |
| edge_occurrence_per_object_edge | 13.15 |

结论：

- uid 级复用空间存在，WIKI 上每个 compact node 平均对应约 8.75 个 state occurrence。
- edge feature 级复用空间更明显，每个 compact edge 平均对应约 13.15 个 edge occurrence。
- 这说明“先按 uid/edge object 计算，再 scatter 到 occurrence”有理论空间。

## 当前 fused 原型状态

已确认使用：

```text
conda env: tgnn_3.10
torch: 2.1.1+cu118
CUDA: 11.8
cmake: 3.26.4
```

`fused_edge_temporal_linear_forward` 已能构建并通过单元测试。

注意：之前失败来自系统 CMake 3.16，不是 CUDA 11.8 本身的问题。

## Fused forward microbenchmark

脚本：

```text
tools/bench_fused_edge_projection.py
```

对比对象：

- baseline：`torch.cat([state[edge_src], edge[edge_to_eid], time], dim=1)` + `linear`
- fused：单 CUDA kernel 内完成 gather + concat + linear

中等规模、真实维度测试：

```text
rows: 65536
state_dim: 100
edge_dim: 172
time_dim: 100
out_dim: 100
repeat: 20
```

结果：

| 方法 | median ms |
| --- | ---: |
| baseline | 1.256 |
| naive fused | 6.740 |
| Triton fused 初始参数 | 1.052 |
| Triton fused 最优扫参 | 0.486 |

结论：

- 当前 naive fused kernel 慢约 5.37 倍。
- 原因是 fused kernel 用 scalar dot，丢掉了 `linear` 的 cuBLAS/Tensor Core 路径。
- 该版本不接入训练主链路。
- Triton fused 初始参数在同配置下比 baseline 快约 16.26%，刚超过 15% 的局部门槛。
- 扫参后最佳配置为 `BLOCK_M=32, BLOCK_N=64, BLOCK_K=32`，同配置下降约 61.27%。
- Triton 版本可以作为下一步训练链路 A/B 的候选，但还不能证明端到端收益。

更大 rows 规模复测：

```text
rows: 262144
state_nodes: 66287
edges: 325717
state_dim: 100
edge_dim: 172
time_dim: 100
out_dim: 100
repeat: 10
BLOCK_M: 32
BLOCK_N: 64
BLOCK_K: 32
```

| 方法 | median ms |
| --- | ---: |
| baseline | 4.914 |
| naive fused | 26.839 |
| Triton fused | 1.930 |

结论：

- rows 放大后 Triton 仍下降约 60.72%。
- 当前收益来自减少 `gather + cat + materialize`，同时保留 `tl.dot` 的矩阵乘路径。
- 当前结果只覆盖 forward microbenchmark，未覆盖 backward 和训练端到端。

## 下一步门槛

下一步只推进一个方向：

1. 先补更多数据集/更多卡数的复用率统计；
2. 如果 state/object 复用率稳定高于 4x，再考虑保留 GEMM/Tensor Core 的融合方案；
3. 新方案必须直接替换当前 Python `to_block`/attention 前的重复 gather + concat + linear 链路；
4. 只有端到端 epoch time 明确下降时保留。

不做 PyTorch split-kernel 版本，因为 kernel 数量不降，预期只会转移开销。

## 训练链路 A/B 初测

接入方式：

- 新增 `--triton_edge_projection on/off`；
- 只替换非 combined attention 中同时包含 node、edge、time 的 K/V projection；
- 不改变 `prepare_input`；
- 不延迟 `b.edata['f'] = edge_feat[e_idx]`；
- backward 使用等价 PyTorch 公式，保证梯度正确。

实现注意：

- 初版把 `ROWS` 作为 Triton `constexpr`，batch 边数变化会导致多次 JIT 编译；
- 已改为 runtime `ROWS`，避免按 batch shape 反复编译。

WIKI 4 卡 2 epoch smoke：

```text
dataset: WIKI
GPU: 4
epoch: 2
skip_eval: on
sample_type: boundery_recent_decay
probability: 0.1
memory_type: historical
shared_memory_ssim: 0.3
```

| 配置 | epoch 1 | epoch 2 | avg |
| --- | ---: | ---: | ---: |
| baseline/off | 0.78s | 0.29s | 0.537s |
| Triton/on，ROWS runtime | 2.01s | 0.29s | 1.152s |

结论：

- Triton 首轮仍有 JIT 编译开销；
- 第二轮稳定 epoch 与 baseline 基本持平，没有观察到端到端收益；
- 当前接入只替换 attention K/V projection，未覆盖提前物化的 `b.edata['f']`；
- 该方向暂时不能作为已有效训练加速点，只能作为可选候选继续在更大数据集或更长 epoch 上验证。

WikiTalk 4 卡 2 epoch 复测：

```text
dataset: WikiTalk
GPU: 4
epoch: 2
skip_eval: on
sample_type: boundery_recent_decay
probability: 0.1
memory_type: historical
shared_memory_ssim: 0.3
```

| 配置 | epoch 1 | epoch 2 | avg |
| --- | ---: | ---: | ---: |
| baseline/off | 12.20s | 12.35s | 12.277s |
| Triton/on | 13.52s | 12.80s | 13.158s |

结论：

- WikiTalk 上 Triton/on 平均慢约 7.18%；
- 第二轮仍慢约 3.64%；
- 当前 Triton attention projection 接入不保留为有效优化点。

运行前修复了 memory updater 的空 shared transition guard：

```text
transition_dense.numel() > 0
```

该修复只影响空 tensor 分支，baseline/off 和 Triton/on 使用同一修复。

## Triton forward 覆盖扩大尝试

进一步尝试：

- `prepare_input` 在 Triton 开关打开时不再提前物化 `b.edata['f'] = edge_feat[e_idx]`；
- block 只保留：
  - compact edge feature object；
  - occurrence 到 edge object 的 `_edge_to_eid`；
- attention 中 Triton kernel 直接按 `_edge_to_eid` 读取 edge object；
- backward 仍使用等价 PyTorch 实现，保证梯度正确。

WikiTalk 4 卡 2 epoch：

| 配置 | epoch 1 | epoch 2 | avg |
| --- | ---: | ---: | ---: |
| baseline/off | 12.20s | 12.35s | 12.277s |
| Triton/on，已物化 edge f | 13.52s | 12.80s | 13.158s |
| Triton/on，object edge load | 13.98s | 12.82s | 13.401s |

结论：

- 延迟 edge feature 物化后，端到端仍没有收益；
- 平均比 baseline 慢约 9.16%；
- 第二轮稳定后仍慢约 3.81%；
- 当前 forward 覆盖扩大不能抵消 backward materialization、DGL softmax/aggregation、memory updater 和 block 构造等全链路开销；
- Triton projection 方向暂时停止推进，除非后续做完整 fused backward 或发现 projection 在 profile 中占比显著升高。

## WikiTalk 当前端到端阶段占比

新增 `--epoch_breakdown on`，用同步 host timing 统计每个 epoch 的主要阶段。

注意：该 profile 会引入同步开销，只用于判断阶段占比，不用于绝对性能对比。

配置：

```text
dataset: WikiTalk
GPU: 4
epoch: 1
skip_eval: on
triton_edge_projection: off
epoch_breakdown: on
```

结果：

| 阶段 | avg seconds | 占 epoch |
| --- | ---: | ---: |
| next_batch / sampling-build 可见等待 | 0.650 | 5.24% |
| prepare_mail / batch mail 准备 | 0.088 | 0.71% |
| forward + loss | 5.917 | 47.68% |
| async fetch wait | 0.563 | 4.53% |
| backward + optimizer step | 4.997 | 40.27% |
| mailbox update | 0.097 | 0.78% |
| uncovered | 0.099 | 0.79% |
| epoch | 12.410 | 100.00% |

结论：

- 当前 WikiTalk 上最大开销是 forward/loss 与 backward/step；
- visible next_batch / sampling-build 等待约 5.24%，不是同步 profile 下最大项；
- fetch wait 约 4.53%，不是当前最大项；
- 如果继续挖全链路收益，下一步应拆 forward 内部：
  - memory updater；
  - GNN attention；
  - DGL edge softmax / aggregation；
  - predictor/loss；
- 只优化 projection kernel 很难改变端到端。

## Compact-first joint K/V 分层投影

### 尝试内容

当前 attention 会先把 temporal state 和 edge feature 展开到每条 edge occurrence，再分别执行 K/V Linear：

$$
K_i=W_K[h_{s(i)},e_{id(i)},\phi(\Delta t_i)]+b_K.
$$

本次利用 Linear 的可加性，将其等价拆成：

$$
K_i=
W_K^h h_{s(i)}+
W_K^e e_{id(i)}+
W_K^t\phi(\Delta t_i)+b_K,
$$

V 同理。实现中：

- state 分量在已经去重的 `(UID, timestamp)` state 上计算一次；
- edge 分量在 compact edge object 上计算一次；
- time 分量仍按 edge occurrence 计算；
- K/V 沿输出维联合计算，再按 `edge_src` 和 `_edge_to_eid` gather-add；
- 不新增 `unique`，不修改 Q、softmax、dropout 和 aggregation；
- 第一版为兼容其他未覆盖的 attention 分支，仍保留 occurrence 级 `b.edata['f']` 物化，只验证投影计算去冗余；
- `--compact_kv_projection on/off` 用于 A/B，默认关闭，并与旧 Triton 开关互斥。

前向输出以及 state、edge、time、K/V 参数的反向梯度已通过 double-precision 等价测试。该变换数学等价，但浮点加法顺序改变，因此不保证逐 bit 一致。

### 局部真实维度测试

测试规模取自 WIKI 复用统计的近似单 batch：

```text
edge occurrences: 107120
temporal states: 14504
edge objects: 8143
state/edge/time dim: 100/172/100
K/V output dim: 100/100
```

| 范围 | baseline | compact-first | 变化 |
| --- | ---: | ---: | ---: |
| K/V forward | 2.816 ms | 2.316 ms | -17.76% |
| K/V forward + backward | 8.695 ms | 6.200 ms | -28.69% |

说明分层投影确实减少了 K/V 局部计算，而不是单纯转移到 backward。

### WikiTalk 四卡端到端结果

配置：`pipeline on`、`skip_eval`、同一 seed、baseline/compact 顺序运行。

2 epoch 初测中，第二个 epoch 从约 13.81 s 降至 13.54 s，约下降 1.9%。该差异随后未在 10 epoch 测试中稳定复现。

| 10 epoch 统计 | baseline/off | compact-first/on | 变化 |
| --- | ---: | ---: | ---: |
| 全轮 rank 平均 | 13.551 s | 13.582 s | +0.23% |
| 去首轮 epoch 中位数 | 13.510 s | 13.685 s | +1.30% |

日志：

- `outputs/compact_kv_projection_ab/wikitalk_4gpu_off_10epoch.log`
- `outputs/compact_kv_projection_ab/wikitalk_4gpu_on_10epoch.log`

### 当前结论

- 分层 K/V 在局部 forward/backward 上真实有效；
- 单独接入完整训练后，WikiTalk epoch 时间基本持平，不能视为端到端有效方法；
- 当前保持默认关闭，不进入有效方法列表，也不继续扩展到全部 attention 分支；
- 由于效率门槛未通过，本轮没有继续执行完整精度/收敛实验；
- 如果后续推进 fused segment attention，可把 compact state/edge projection 作为其内部组件复用，并融合 gather-add、attention score、segment softmax 和 aggregation，避免局部收益再次被短 kernel 与中间张量抵消。

## Memory updater 去冗余评估

### 细粒度 profile

在 `AsyncMemeoryUpdater` 中加入默认关闭的分阶段 profiler。host wall 统计全部 batch；CUDA event 在 warm-up 32 个 batch 后每 8 个 batch抽样一次。WikiTalk 四卡、1 epoch、457 batch/rank 的 CUDA 抽样均值如下。

| 阶段 | ms/batch | updater 抽样阶段合计占比 |
| --- | ---: | ---: |
| next_batch_fetch_launch | 2.261 | 23.73% |
| raw_memory_update | 1.747 | 18.33% |
| shared_update_submit | 1.702 | 17.86% |
| mail_memory_prepare | 0.912 | 9.57% |
| smooth_prediction | 0.763 | 8.00% |
| previous_shared_sync | 0.658 | 6.91% |
| feature_attach | 0.426 | 4.47% |
| smooth_filter_update | 0.416 | 4.36% |
| smooth_blend | 0.316 | 3.32% |
| local_mail_memory_write | 0.250 | 2.62% |
| state_snapshot | 0.064 | 0.67% |
| shared_mask_prepare | 0.013 | 0.14% |

这些数值用于阶段排序，不能直接相加为严格的 updater wall time，因为 CUDA 工作与通信存在重叠。结果表明，smooth/Filter 主体约 1.495 ms/batch；通信与编排相关阶段合计约 5.783 ms/batch，是更大的优化对象。

日志：

- `outputs/memory_updater_ab/wikitalk_4gpu_skip_local_reduce_profile_1epoch.log`

### 保留：删除本地写入前的第二轮 reduce

原链路中，`get_update_memory()` 与 `get_update_mail()` 已经分别执行：

1. 按 distributed node ID 做 `unique(return_inverse=True)`；
2. 按 timestamp 做 `scatter_max`；
3. 返回每个 node 唯一的一条最新 memory/mail。

随后只筛选本 rank 的元素。布尔筛选不会重新制造重复 ID，而同一 partition 内 `DistIndex.loc` 对这些 ID 仍是一一对应。旧实现仍向 `set_memory_local()` 和 `set_mailbox_local()` 传入 `Reduce_Op='max'`，导致两套输入已唯一的 `unique + scatter_max` 再执行一次。

当前保留的改动只是移除这两个 `Reduce_Op='max'` 参数，让唯一索引直接写入本地 memory/mailbox。它不改变远程通信、采样、模型参数或 timestamp 选择结果；`mailbox_size > 1` 时，每个唯一节点仍只推进一次 mailbox position。

WikiTalk 四卡、10 epoch、相同 seed 的 profiler-off 结果：

| 当前代码复测指标 | 恢复二次 reduce | 删除二次 reduce | 变化 |
| --- | ---: | ---: | ---: |
| 四 rank 全轮平均 epoch | 13.8447 s | 13.6153 s | -1.66% |
| 关键 rank 全轮平均 epoch | 13.8521 s | 13.6251 s | -1.64% |
| 去首轮稳态均值 | 13.7942 s | 13.5442 s | -1.81% |
| 去首轮稳态中位数 | 13.7275 s | 13.4950 s | -1.69% |

在 profiler 接入前的独立 10-epoch 配对中，四 rank 平均也由 13.6423 s 降至 13.4645 s，下降 1.30%。两轮独立测试方向一致。

日志：

- `outputs/memory_updater_ab/wikitalk_4gpu_baseline_10epoch.log`
- `outputs/memory_updater_ab/wikitalk_4gpu_skip_local_reduce_10epoch.log`
- `outputs/memory_updater_ab/wikitalk_4gpu_redundant_local_reduce_current_10epoch.log`
- `outputs/memory_updater_ab/wikitalk_4gpu_current_baseline_10epoch.log`

当前结论仅是“WikiTalk 四卡初步有效并保留实现”。它尚未完成双机八卡、WIKI/LASTFM 50 epoch、WikiTalk/Flight 10 epoch与完整精度收敛验证，因此不能升级为论文主技术点。

### 已尝试但回退

| 方案 | 局部变化 | 端到端变化 | 结论 |
| --- | ---: | ---: | --- |
| node feature 先按 UID object 投影再 gather | forward -48.79%；forward+backward -0.90% | 未接入 | backward scatter 抵消收益，暂不推进 |
| 用普通 PyTorch 拆分 GRU 的 UID/time affine | forward +15.33%；forward+backward +11.83% | 未接入 | kernel 增多且 GEMM 变碎，回退 |
| historical submit 跳过未使用 shared mail gather | `shared_update_submit` 1.702→2.052 ms | epoch 13.633→14.244 s，+4.48% | collective 等待吞掉清理收益，回退 |
| node payload 两层 `cat` 合成一次 | microbenchmark -35.9%；fetch-launch 2.261→2.206 ms | epoch 13.633→13.960 s，+2.40% | 只省 0.055 ms/batch，回退 |
| smooth normalization 改为 GPU-only、删除 `.item()` | smooth 0.763→0.745 ms | epoch 13.633→13.763 s，+0.96% | `.item()` 主要吸收前序等待，回退 |
| memory/mail 复用一次 UID latest selector | mail prepare 0.912→0.657 ms，-27.9% | 去首轮稳态均值 +0.04%，中位数 +0.26% | 局部收益未落到关键路径，回退 |
| 延迟 access statistics 的四次 `.item()` | fetch-launch 2.261→2.190 ms，-3.1% | 未形成稳定 epoch 收益 | 端到端上限约 0.23%，回退 |

其中 latest selector 的最终原型由 mail 自己执行原 timestamp selector，再让 memory 复用其位置。这样同 UID、同 timestamp 的并列事件仍由 mail 原规则选择；这些事件经 temporal dedup 后指向同一 memory state，因此 memory 值不变。该方案语义风险已处理，但端到端门槛仍未通过，所以代码已回退。

### 下一条高上限方向

当前可复用率约为每个 UID object 对应 8.75 个 temporal state。`raw_memory_update` 中 mailbox 与 hidden-state 的 UID 部分可先投影一次，time encoding 仍按 `(UID, timestamp)` state 计算。理论上能删除大量 GRU affine FLOPs，但普通 PyTorch 拆分已经实测更慢，原因是把原生 fused GRU/cuBLAS 路径拆成多个短 kernel。

因此下一版只有在“UID affine GEMM + state time affine + GRU pointwise”保持少量大 GEMM并融合 pointwise 时才值得实现。第一门槛应是 forward+backward 局部速度至少提高 15%，否则不接入完整训练。

## Compact UID GRU 第一版拆分实现

### 方法

TGN 的 GRU 输入可写为 mailbox 与 time feature 的拼接：

$$
x_s=[m_{u(s)},\phi(\Delta t_s)],
$$

其中同一 UID 的 mailbox $m_u$ 与 hidden memory $h_u$ 可被多个 temporal state 复用，而 $\phi(\Delta t_s)$ 仍是 state-specific。当前 `b.srcdata['__ID']` 已提供 state 到 compact UID object 的映射，因此不新增 `unique`。

第一版保持原 `torch.nn.GRUCell` 的全部参数，只把 affine 拆成：

$$
G_s^{x}=
\operatorname{gather}(m_u W_{ih,m}^{T},u(s))
+
\phi(\Delta t_s)W_{ih,t}^{T},
$$

$$
G_s^{h}=
\operatorname{gather}(h_u W_{hh}^{T},u(s)).
$$

time GEMM 与 mailbox gate 的相加使用 `torch.addmm`；随后直接调用当前 PyTorch 2.1.1 CUDA GRUCell 自身使用的 `aten::_thnn_fused_gru_cell` 完成 reset/update/new gate pointwise。两个原 bias 分别传入，未合并；state dict、DDP 参数和 checkpoint 格式不变。

数据侧只做两件事：

- block 挂载 compact mailbox/memory tensor 的零拷贝引用；
- 开关打开时跳过 `mailbox[idx]` 的 occurrence 级 `mem_input` 物化。

occurrence 级 `mem`、`mem_ts` 与 `mail_ts` 继续保留，避免改变 historical cache、smooth/Filter 与 time encoding。CPU、不支持的 dtype 或私有 op 不可用时，helper 会临时恢复 occurrence mailbox 并回退原 `GRUCell`。

开关：

```text
--compact_gru_update off/on
```

当前默认关闭。

### 数值与局部性能

CUDA double 测试已比较：

- forward 输出；
- time feature 梯度；
- `weight_ih`、`weight_hh`；
- `bias_ih`、`bias_hh`。

全部通过 `assert_close`。拆分 GEMM 改变浮点累加顺序，因此数学等价但不保证逐 bit 一致；FP32 输出最大绝对误差约 $1.2\times10^{-6}$，模型参数梯度最坏相对 L2 误差低于 $10^{-6}$。

严格生产梯度口径下，mailbox/memory/timestamp 是无梯度通信 buffer，TimeEncode 与 GRU 参数参与 backward。WikiTalk 代表形状 `state=13865`、`UID object=1579`：

| 局部范围 | 变化 |
| --- | ---: |
| forward | -23.20% |
| forward + backward | -16%～-17% |

旧的“所有输入均 requires-grad”合成结果不代表生产链路，已撤回且不作为结论。

### WikiTalk 四卡效率

1-epoch 细 profile：

| 指标 | off | on | 变化 |
| --- | ---: | ---: | ---: |
| raw_memory_update | 1.747 ms/batch | 1.566 ms/batch | -10.34% |
| 完整 epoch（profiler on） | 13.633 s | 13.444 s | -1.38% |

profiler-off、10 epoch、相同 seed 的配对结果：

| 指标 | off | on | 变化 |
| --- | ---: | ---: | ---: |
| 四 rank 全轮平均 | 13.3669 s | 13.0592 s | -2.30% |
| 关键 rank 全轮平均 | 13.3735 s | 13.0688 s | -2.28% |
| 去首轮稳态均值 | 13.3208 s | 13.0122 s | -2.32% |
| 去首轮稳态中位数 | 13.3250 s | 13.0275 s | -2.23% |

因此该版本已通过 WikiTalk 四卡单 seed 的初步端到端效率门槛，代码暂时保留，但仍默认关闭。

### WikiTalk 四卡精度与收敛

另执行了带 val/test 的完整 10 epoch 配对。两个配置的 best validation AP 都出现在 epoch 8：

| best-val epoch 指标 | off | on | 变化（百分点） |
| --- | ---: | ---: | ---: |
| validation AP | 0.972413 | 0.973660 | +0.1247 |
| validation AUC | 0.969988 | 0.971407 | +0.1419 |
| 对应 test AP | 0.960342 | 0.968650 | +0.8308 |
| 对应 test AUC | 0.958002 | 0.967004 | +0.9002 |

当前只允许得出“单 seed 未观察到精度或收敛退化”。浮点累加顺序不同会改变后续 dropout/训练轨迹，表中的正变化不能归因为方法本身；至少还需要三个 seed 与其他数据集。

两次精度任务均完成全部 10 个 epoch 和最后一轮评估，随后在旧的结果保存代码中失败：绝对 `--partition` 路径被直接拼进输出文件名，导致父目录不存在。该错误发生在所有 epoch 指标打印之后，不影响上表，但自动 `best test AP` 汇总行没有生成；表格由逐 epoch global 指标按最大 validation AP 重算。

日志：

- `outputs/memory_updater_ab/wikitalk_4gpu_compact_gru_profile_1epoch.log`
- `outputs/memory_updater_ab/wikitalk_4gpu_compact_gru_addmm_profile_1epoch.log`
- `outputs/memory_updater_ab/wikitalk_4gpu_compact_gru_off_10epoch.log`
- `outputs/memory_updater_ab/wikitalk_4gpu_compact_gru_on_10epoch.log`
- `outputs/memory_updater_ab/wikitalk_4gpu_compact_gru_off_10epoch_eval.log`
- `outputs/memory_updater_ab/wikitalk_4gpu_compact_gru_on_10epoch_eval.log`

### 此前计划（已被八卡结果取代）

四卡初测后原计划按以下轮数继续验证：

1. WIKI、LASTFM 各 50 epoch；
2. WikiTalk、Flight 各 10 epoch；
3. 至少三个 seed；
4. 双机八卡复测 raw updater、epoch time、AP/AUC 与 time-to-target-AP；
5. 若 PyTorch/CUDA 版本变化，必须重新验证私有 fused op，不能静默假定兼容。

后续八卡隔离结果没有确认端到端收益，因此该多数据集长测计划现已停止；以下八卡结论取代四卡阶段的“继续推进”判断。

## Compact UID GRU 第一版拆分实现的双机八卡最终评估

### 测试环境与隔离修正

测试使用 WikiTalk、双机八卡、每机 GPU 0--3、`tgnn_3.10`、PyTorch 2.1.1+cu118 和 CUDA 11.8。跨机通信固定为：

```text
NCCL_IB_DISABLE=1
NCCL_NET=Socket
NCCL_SOCKET_IFNAME=ens4f1np1
GLOO_SOCKET_IFNAME=ens4f1np1
```

检查发现原入口以 `cache_use_rpc=True` 启动了没有业务调用的 TensorPipe RPC。它既不承载 remote-neighbor fetch，也不参与 memory/feature AllToAll，却会建立 IBV transport 并在退出时打印 EOF。当前训练入口已经停止启动这条闲置 RPC；remote-neighbor、feature 和 memory 通信仍由 NCCL AllToAll 承担。

隔离版日志满足：

- NCCL 只出现 `NET/Socket` 和 `ens4f1np1`；
- 不出现 `NET/IB`；
- 不出现 TensorPipe、`transport/ibv` 或 CUDA GDR；
- 两机八卡均完成全部 epoch。

另新增：

```text
--freeze_adaptive_parameters
```

用于固定采样 alpha/beta，防止 Compact GRU 改变计时后又间接改变下一批采样量。

### 自适应 ABBA：只能作为历史信号

第一组运行顺序为 `OFF1 -> ON1 -> ON2 -> OFF2`，使用原自适应参数。epoch 1 以后 alpha/beta 和远程工作量基本一致，得到：

| 指标 | OFF 两次均值 | ON 两次均值 | 变化 |
| --- | ---: | ---: | ---: |
| 稳态 rank mean | 7.1144 s | 6.9217 s | -2.71% |
| 稳态 critical mean | 7.1256 s | 6.9300 s | -2.74% |
| 稳态 critical median | 7.1300 s | 6.9150 s | -3.02% |

两种顺序都复现了约 2.6%--3.2% 的表面收益，同配置漂移低于 0.3%。但该组仍启用了闲置 TensorPipe RPC，且前几轮包含 adaptive timing synchronization，因此不能与后续严格隔离结果混合，也不能作为最终结论。

### 冻结在初始参数：高远程采样量下无收益

固定：

```text
alpha = 0.3
beta = 0.1
```

OFF/ON 的每个 epoch、每个 rank 的 local/remote node 和 edge 数量完全一致。结果为：

| 指标 | OFF | ON | 变化 |
| --- | ---: | ---: | ---: |
| 稳态 rank mean | 7.7664 s | 7.7721 s | +0.07% |
| 稳态 critical mean | 7.7700 s | 7.7789 s | +0.11% |
| 稳态 critical median | 7.7700 s | 7.7900 s | +0.26% |

正负变化在各 epoch 间交替，无法确认加速。该配置每 epoch 约有 139.7 万远程节点和 748.7 万远程边，通信与更大的 occurrence 工作集会稀释局部 GRU 优化。

### 冻结在稳定工作点：最终判定

为了匹配自适应训练收敛后的工作点，固定：

```text
alpha = 0.1
beta = 0.01
```

同样执行 `OFF1 -> ON1 -> ON2 -> OFF2`。四组共 320 个 rank-epoch 的 alpha/beta 和 local/remote node/edge 四元组完全一致；每 epoch 约有 17.3 万远程节点和 97.5 万远程边。

两个相邻配对出现相反结论：

| 配对 | 稳态 rank 变化 | 稳态 critical mean 变化 |
| --- | ---: | ---: |
| OFF1 -> ON1 | +1.29% | +1.32% |
| ON2 -> OFF2 | -1.05% | -1.03% |

抵消运行顺序后：

| 汇总范围 | OFF 两次均值 | ON 两次均值 | ON 变化 |
| --- | ---: | ---: | ---: |
| epoch 1--9 rank mean | 6.9115 s | 6.9198 s | +0.12% |
| epoch 1--9 critical mean | 6.9172 s | 6.9272 s | +0.15% |
| epoch 3--9 rank mean | 6.9204 s | 6.9329 s | +0.18% |
| 最后四轮 critical mean | 6.9163 s | 6.9288 s | +0.18% |

ON 两次重复之间有约 1.8%--1.9% 的系统漂移，明显大于方法均值。两侧 loss 全部有限，轨迹接近，不能用收敛差异解释计时翻转。

因此八卡最终结论是：

> Compact UID GRU 的局部去冗余真实存在，但第一版拆分实现没有可确认的双机八卡端到端收益；方法均值约为持平到回退 0.2%。

第一版拆分实现继续默认关闭，不再单独执行多 seed、多数据集长测。后续只在 Block unique 和 fused attention 联合链路中减少中间物化与 kernel 数量，结果见第八阶段。

日志目录：

```text
outputs/memory_updater_8gpu_ab/20260817/
```

## 最终技术点：Sampler-Native Block-Unique Compact Execution

Compact GRU、segment attention 和 P0 不再拆成独立论文贡献。统一技术点定义为：

> 复用 sampler 已生成的线程局部 occurrence，在采样返回前完成 UID object、temporal state 和 edge object 的分层 compaction；Block builder 与 memory updater 消费同一组映射，attention 再直接消费 builder 缓存的 segment plan，减少 Python/GPU 侧重复 unique、remap、展开和 DGL 编排。

统一表示包含：

- UID object 表；
- `(UID, timestamp)` temporal-state 表；
- edge-object 表；
- `state_to_uid`；
- `edge_src`；
- `edge_to_object`；
- destination-major 的 `dst_ptr`；
- occurrence 级 `dt`。

当前 P0 实现是在采样结束后对各 sampler thread 的局部输出并行 hash compaction，再按 seed、thread ID 和线程内首次出现顺序稳定合并。它消除了 legacy 路径的全局 Python/GPU unique 与 remap，但仍有一次线程局部扫描和一次结果物化，不能表述成“完全零扫描”或“采样时逐 occurrence 在线去重”。

当前状态：

| 组成部分 | 状态 | 结论 |
| --- | --- | --- |
| sampler-native hierarchical compaction（P0） | 已实现，显式开关 | WikiTalk 八卡 ABBA 稳态收益约 4.0%；只在训练 sampler 启用 |
| direct segment builder | 已实现 | P0 输出 COO `row/col`；DGL direct COO 构造一次 Block 并缓存 CSC plan |
| fused segment attention | 已实现，显式开关 | 融合 score、softmax、dropout 和 V aggregation；最终版仍使用 occurrence K/V |
| gather-pointwise Compact GRU | 已实现，显式开关 | 复用 UID object GEMM，并融合 occurrence gather、gate 和 GRU pointwise |
| projected/compact K/V | 已尝试，最终关闭 | 局部 kernel 有收益，八卡端到端回退约 3%，退出最终方案 |
| K/V 公共输入物化 | 已尝试并回退 | 稳态约 0.9%，低于 1% 保留门槛 |

## Fused Segment Attention 设计

### 为什么必须与 Block unique 一起做

当前 attention 在 occurrence 级依次执行 Q/K/V 展开、QK dot、LeakyReLU、DGL edge softmax、dropout、`V * att` 物化和 `update_all`。只替换 K/V projection 已证明无法形成端到端收益。

最终版先在 occurrence 粒度构造：

$$
X_e = [h_{s(e)} \Vert f_{o(e)} \Vert \phi(\Delta t_e)],
$$

$$
K_e = W_K X_e, \qquad V_e = W_V X_e.
$$

K/V 仍完整物化，segment kernel 从 Q/K/V 开始融合，不在寄存器中重建 compact K/V。分层 projected K/V 已完成前后向原型，但八卡端到端回退约 3%，因此不属于最终版本。segment kernel 计算：

$$
z_e = \operatorname{LeakyReLU}(q_{d(e)}^{T}K_e),
$$

$$
a_e = \operatorname{Dropout}
\left(
\operatorname{softmax}_{e\in\mathcal E(d(e))}(z_e)
\right),
$$

$$
o_d = \sum_{e\in\mathcal E(d)} a_e V_e.
$$

实现必须保持当前语义。当前 attention 不包含以下缩放：

$$
\frac{1}{\sqrt d}.
$$

LeakyReLU slope 为 0.2；dropout 位于 softmax 之后，按 edge-head 独立采样，并按以下系数放大且不重新归一化：

$$
\frac{1}{1-p}.
$$

### 执行边界

P0 在 sampler 返回前生成 compact object/state 表以及 occurrence COO：

```text
dist_nodes, dist_edges
state_to_node, state_timestamps, root_mapping
row, col, edge_mapping, delta_ts
```

direct builder 使用这些 `row/col` 通过 DGL direct COO 构造一次 Block，同时取得并缓存 destination-major 的 `dst_ptr` 与 `edge_ids`。sampler 本身不直接输出 CSC，最终版也没有完全绕开 DGL。attention 热路径只读取已缓存 plan，不再临时调用 `b.adj_tensors('csc')` 或重新 sort/unique。

### 最小实现顺序

1. 使用 DGL CSC 完成 correctness prototype；已完成；
2. 实现 Triton segment forward/backward，对齐 DGL 输出及 Q/K/V 梯度；已完成；
3. 对齐 post-softmax dropout 语义并覆盖固定 mask 测试；已完成；
4. 由 Block builder 提供 `dst_ptr` 和 edge mapping，删除 attention 热路径的 CSC 构造与同步校验；已完成；
5. 使用 tiled segment loop 支持长 segment，避免高 degree destination 触发整批 fallback；已完成；
6. 接入 compact state/edge K/V；已完成原型并因八卡端到端回退而放弃；
7. 接入 gather-pointwise Compact GRU；已完成并保留为联合链路内部组件；
8. sampler-native hierarchical compaction 与稳定映射；已完成，训练启用、评估回退 legacy。

第一版仅支持当前 `TGN_large` 的 FP32、非 combined、node+edge+time 完整分支；其他配置继续走 DGL fallback。dropout mask 由 `aten.native_dropout` 使用 PyTorch Philox 路径生成，Triton 只消费 mask；这对齐了 mask 与 RNG state 的消耗方式，但融合算子的浮点归约仍不能声明训练轨迹逐 bit 相同。

### 第一阶段 standalone 原型结果

新增：

```text
starrygl/module/triton_segment_attention.py
starrygl/sample/direct_segment_block.py
tests/test_triton_segment_attention.py
tests/test_direct_segment_block.py
```

该原型接收 materialized Q/K/V 和 destination-major `dst_ptr`。每个 `(destination, head)` 由一个 Triton program 完成 QK dot、LeakyReLU、稳定 segment softmax、post-softmax dropout 和 V reduction；backward 重算 softmax并直接生成 Q/K/V 梯度。当前 dropout 使用 `aten.native_dropout` 生成的 bool edge-head mask。最终训练 layer 已接入该路径，但不包含 compact K/V 重建。

数值测试覆盖空 segment、非均匀 degree、固定 dropout mask、非 identity edge permutation、FP32 forward 和 Q/K/V backward。长 segment 已改为 32-edge 动态分块，degree 630 的 forward/backward 已对齐；CPU、非 FP32 或 head dimension 大于 64 时仍走可求导 PyTorch fallback。

代表 TGN_large 形状的局部测试：

```text
destinations: 6000
edges: 108000
degree: 18
heads: 2
head dimension: 50
dtype: FP32
dropout: 0
Block: 计时前已构建
```

| 局部范围 | DGL baseline | Triton segment | 变化 |
| --- | ---: | ---: | ---: |
| forward | 0.773 ms | 0.322 ms | -58.30% |
| forward + backward | 1.865 ms | 0.672 ms | -63.99% |

加入训练配置使用的 `dropout=0.2` 后，包含外部随机 mask 生成的结果为：

| 局部范围 | DGL baseline | Triton segment | 变化 |
| --- | ---: | ---: | ---: |
| forward | 0.778 ms | 0.356 ms | -54.21% |
| forward + backward | 1.873 ms | 0.691 ms | -63.09% |

固定相同 dropout mask 时，forward 和 forward+backward 分别下降 58.12% 和 64.67%。这说明当前外部 mask 的开销较小，融合主体在训练语义下仍有效。

该结果证明 QK、segment softmax、dropout 和 V aggregation 的融合本身有明显空间，且 backward 收益没有被抵消。它不包含 K/V projection、CSC 构造、Block builder 和完整训练，因此不能直接外推为 epoch 加速。

独立调用保留 `dst_ptr` 内容校验；训练路径信任 builder 附带的 segment plan，不再执行 `.item()` 同步。下一步不再优化第一版 standalone Compact GRU，只验证 Block-Unique 与 fused segment 的全链路组合。

### 第二阶段：运行时 CSC 的 layer 接入（历史初测）

新增 `--fused_segment_attention off/on`，仅替换当前 `TGN_large` 的非 combined、node+edge+time 完整 attention 分支。Q/K/V projection 保持原实现；融合后直接生成 destination 表征，跳过 DGL `edge_softmax`、attention dropout、`V * att` 物化和 `update_all`。其他模型分支原样回退。非 destination-major 边序的 layer forward、输入梯度和全部参数梯度已经与 DGL 路径对齐。

WikiTalk 单机四卡、固定 `alpha=0.1`、`beta=0.01`、2 epoch 的初步 A/B 中，epoch 1 的 critical time 为：

| 顺序 | OFF | ON | ON 变化 |
| --- | ---: | ---: | ---: |
| OFF -> ON | 12.49 s | 12.62 s | +1.04% |
| ON -> OFF | 12.67 s | 12.62 s | -0.39% |

两种顺序符号相反，该运行时 CSC 版本只能判为端到端持平，不能声明训练加速。采样工作量逐 epoch 一致，因此差异不是远程节点或边数量变化造成的。

一轮 457 batch 的诊断显示，各 rank 有 398--429 个 batch 使用 Triton，28--59 个 batch 因最大 destination degree 超过 64 而整批回退；实际最大 degree 为 189--630。也就是说，约 6.1%--12.9% 的 batch 没有使用融合 kernel。同时，当前 layer 每 batch 调用 `b.adj_tensors('csc')`，并通过 `.item()` 校验 `dst_ptr` 与最大 degree，存在强制 GPU 到 CPU 同步。

因此该结果不否定 fused segment 核心，而是明确了生产接入必须同时完成两项：

1. Block builder 直接输出 `dst_ptr`、edge permutation 和最大 degree，删除 attention 热路径的 CSC 构造与同步校验；
2. 支持长 segment，避免少量高 degree destination 触发整批 PyTorch fallback。

日志位于：

```text
outputs/fused_segment_attention_ab/20260817/
```

### 第三阶段：Builder Segment Plan 与长 Segment 的完整轮数初筛

当前版本在 Block unique 构造时保留 `dst_ptr` 和 edge permutation，fused attention 热路径不再构造 CSC 或执行 GPU 到 CPU 的内容校验。Triton kernel 用 32-edge 分块循环处理长 segment，已覆盖最大 degree 630 的 forward/backward 数值测试，不再因 degree 大于 64 触发整批 PyTorch fallback。

WikiTalk 单机四卡、seed 95842、固定 `alpha=0.1`、`beta=0.01`、冻结自适应采样参数并跳过评估，执行了 ON -> OFF 各 10 epoch。两侧逐 epoch 的 local/remote node 和 edge 工作量完全一致。

| 统计范围 | OFF | ON | ON 变化 |
| --- | ---: | ---: | ---: |
| 全 10 轮 rank mean | 12.3773 s | 11.9430 s | -3.51% |
| 全 10 轮 critical mean | 12.3840 s | 11.9500 s | -3.50% |
| epoch 1--9 rank mean | 12.3197 s | 11.8322 s | -3.96% |
| epoch 1--9 critical mean | 12.3267 s | 11.8400 s | -3.95% |
| 最后四轮 critical mean | 12.2875 s | 11.8250 s | -3.76% |
| epoch 0 critical time | 12.9000 s | 12.9400 s | +0.31% |

节点和边的 sampler 工作量虽然完全一致，shared-memory 更新量并未完全固定：全 10 轮 ON 比 OFF 低 1.11%，最后四轮反而高 2.60%。因此，稳态约 4% 的信号不能全部归因于更少的通信，但也还不是完全固定通信轨迹的对比。

两侧的 loss 都有限且下降，末轮平均 loss 仅相差 0.024%。但本轮使用 `--skip_eval`，日志中的 validation/test 零值只是占位符，不能证明 AP、AUC 或收敛速度不变。

第三阶段结束时只允许以下结论：

> Block-Unique + Fused Segment Attention 已在 WikiTalk 单机四卡显示 3.5%--4.0% 的端到端效率信号；后续双机八卡结论见第四、五阶段。

### 第四阶段：双机八卡 ABBA 效率验证

两台机器各使用 GPU 0--3，共 8 卡。NCCL 固定为 Socket，绑定 10GbE `ens4f1np1`；全部日志都显示 `NET/Socket` 和 `GPU Direct RDMA Disabled`，没有 `NET/IB`。固定 `alpha=0.1`、`beta=0.01`，执行：

```text
OFF1 -> ON1 -> ON2 -> OFF2
```

四组的 local/remote node 和 edge 工作量逐 epoch、逐 rank 完全一致。

| 指标 | 配对 1 | 配对 2 | ABBA 方法均值 |
| --- | ---: | ---: | ---: |
| 全 10 轮 rank mean 变化 | -3.14% | -3.80% | -3.47% |
| 全 10 轮 critical mean 变化 | -3.17% | -3.73% | -3.45% |
| epoch 1--9 rank mean 变化 | -4.45% | -4.42% | -4.43% |
| epoch 1--9 critical mean 变化 | -4.49% | -4.35% | -4.42% |
| 最后四轮 critical mean 变化 | -4.33% | -4.35% | -4.34% |

ABBA 方法均值为：

| 统计范围 | OFF | ON | ON 变化 |
| --- | ---: | ---: | ---: |
| epoch 1--9 rank mean | 6.8873 s | 6.5819 s | -4.43% |
| epoch 1--9 critical mean | 6.8933 s | 6.5889 s | -4.42% |
| 最后四轮 critical mean | 6.8875 s | 6.5888 s | -4.34% |

OFF 两次的稳态 critical 漂移为 0.26%，ON 两次为 0.41%，都明显小于 4.42% 的方法差异。在该固定配置和单 seed 的两组 ABBA 配对中，端到端时间下降得到稳定复现；但 shared-memory 更新量的稳态方法均值同时下降 1.60%，不能将全部 4.42% 都归因于 attention kernel 本身。

四组 loss 都有限并稳定下降，但仍使用 `--skip_eval`。因此当前判定为：

> 第四阶段结束时，双机八卡效率门槛已通过；当时尚未执行精度验证，多 seed 门槛也尚未通过。

### 第五阶段：双机八卡精度与收敛初步验证

保持同一固定采样配置，使用 seed 95842，去掉 `--skip_eval`，对 OFF 和 ON 分别执行一次 10 epoch validation/test。两侧的 local/remote node 和 edge 工作量逐 epoch、逐 rank 完全一致。本轮只启用 builder segment plan 和 fused attention；Compact GRU、compact K/V 与 Triton edge projection 均关闭。

| 指标 | OFF | ON | 变化（百分点） |
| --- | ---: | ---: | ---: |
| best val AP | 97.0019% (epoch 9) | 97.0844% (epoch 8) | +0.0825 |
| best val AUC | 96.9604% | 97.0423% | +0.0819 |
| best-val epoch 的 test AP | 96.9726% | 97.1075% | +0.1349 |
| best-val epoch 的 test AUC | 97.1193% | 97.2597% | +0.1404 |
| final val AP | 97.0019% | 97.0683% | +0.0664 |
| final val AUC | 96.9604% | 97.0298% | +0.0694 |
| final test AP | 96.9726% | 97.0310% | +0.0584 |
| final test AUC | 97.1193% | 97.1824% | +0.0631 |

以 val AP 97.0% 为报告阈值，OFF 在 epoch 9 首次达到，ON 在 epoch 8 首次达到。累计 critical train time 从 69.77 s 降到 60.92 s，time-to-target 下降 12.68%。这是单 seed 的探索性观察，且只累加训练段、不含 validation/test 耗时，不能外推为稳定的收敛加速。

OFF 和 ON 的 loss 都有限并稳定下降。该阶段当时没有观察到精度或收敛回退，但只有一个 seed；当时 fused dropout 与 PyTorch dropout 的随机数序列不同，只能要求分布等价。最终版的 RNG 对齐修复与多 seed 结果见第十一阶段。

补充修复：OFF 的 10 轮评估和所有指标均已完整输出，但旧代码在训练结束后把绝对 partition 路径拼入结果文件名，导致最后保存失败。已改为显式创建结果目录，并只使用 partition basename 作为文件标签；该修复不影响训练、评估或计时。

### 第六阶段：标准配置的四数据集完整轮数验证

为与 MemShare 的常用设置对齐，固定 `alpha/SSIM=0.3`、`beta/probability=0.1`、seed 6773，在双机八卡、10GbE Socket 环境执行 WIKI 和 LASTFM 各 50 epoch、WikiTalk 和 Flight 各 10 epoch。四个数据集的 OFF/ON 都逐 epoch、逐 rank 保持完全相同的 local/remote node 和 edge 工作量；只打开 builder segment plan 与 fused attention，其他 compact backend 均关闭。

| 数据集 | 完整轮数平均变化 | 稳态 critical 变化 | 尾段 critical 变化 | best-validation 对应 test AP/AUC 变化（百分点） |
| --- | ---: | ---: | ---: | ---: |
| WIKI | +1.42% | -5.59% | -5.95%（最后 10 轮） | -0.0124 / -0.0081 |
| LASTFM | -3.60% | -4.70% | -5.81%（最后 10 轮） | +0.7996 / +0.4400 |
| WikiTalk | -4.62% | -5.29% | -5.71%（最后 4 轮） | -0.0063 / +0.0025 |
| Flight | -4.09% | -6.33% | -5.88%（最后 4 轮） | +0.0634 / +0.0435 |

WIKI 每轮训练仅约 0.16 s。ON 的 epoch 0 因首次 Triton 编译从约 1.04 s 增至 1.63 s，导致 50 轮 full 平均仍回退约 1.4%；排除首轮后，稳态下降约 5.6%。其余三个数据集在包含启动成本的完整轮数口径下也已经取得 3.6%--4.6% 的下降。

通信量没有为效率结果提供有利偏差：稳态 shared comm 在 WIKI、LASTFM、WikiTalk、Flight 分别变化约 `+0.16%`、`+5.48%`、`+0.74%`、`+2.60%`，而 `memory comm` 均为 0。即使 ON 的同步量持平或增加，四个数据集的稳态训练时间仍下降，因此主要效率信号来自 attention 执行链路。

单 seed 精度方面，WIKI 和 WikiTalk 的 test AP/AUC 变化绝对值不超过 0.013 个百分点；Flight 小幅上升；LASTFM 的 AP 上升较大，但伴随随机训练轨迹和 shared comm 差异，不能解释为 fused backend 带来的精度提升。当前只允许判断四数据集均未观察到实质精度退化，不能声明严格无损或精度提升。

另外，低通信配置 `alpha=0.1`、`beta=0.01` 下的 WIKI 50 epoch 也完成了配对：工作量和 shared comm 基本完全一致，稳态约下降 9%，但该结果只作为压力测试，不与上表的标准配置混算。

本阶段原始日志位于 `outputs/fused_segment_attention_ab/20260817/8gpu_multidataset/`。

### 第六阶段当时状态与继续推进判据（历史）

- forward 和 backward 数值测试通过；
- 不在 attention 中新增 unique、sort 或 CSC 构造；
- 局部 forward+backward 时间已明确下降；峰值显存尚未报告；
- 双机八卡固定工作量的 OFF/ON ABBA 配对有稳定下降；已完成；
- WIKI、LASTFM、WikiTalk、Flight 在该阶段均已完成标准配置单 seed 的完整轮数效率和精度初筛；当时多 seed 尚未完成；
- 最终报告整体 Block-Unique Compact Execution，不把 Compact GRU、K/V split 和 segment softmax拆成多个小贡献。

该阶段定位是保留候选：四数据集八卡稳态效率与单 seed 精度初筛已通过，但当时尚未完成多 seed。最终定位由第十一阶段的直接组合 A/B 覆盖。

### 第七阶段：分层 K/V 直接重建与 Compact GRU 短测

进一步尝试让 segment kernel 直接读取 state-level、edge-object-level 和 time-level 的 K/V 投影分量，在寄存器中重建 K/V，避免 occurrence 级 K/V 物化。现已补齐 projected backward：kernel 同时重建 K/V，并通过 atomic accumulation 回写 state、edge-object、time 三类投影梯度。

WikiTalk 双机八卡、标准配置、2 epoch 短测中：

| 配置 | 稳态 train time |
| --- | ---: |
| fused segment attention | 约 7.37 s/epoch |
| projected K/V + fused attention（旧版 occurrence backward） | 约 7.98 s/epoch |
| fused attention + 第一版拆分 Compact GRU | 约 7.30 s/epoch |

旧版结果不能代表最终判断：它的 backward 会重新物化 occurrence K/V。新版 projected backward 已通过 CUDA 前后向数值测试，完整端到端复测结果见下文；第一版拆分 Compact GRU 的短测只有约 1% 级差异，结合此前八卡冻结 ABBA 结果，不能确认稳定端到端收益。

补充单机 A40 kernel 微基准（FP32，`E=65536`、`state=8192`、`edge-object=4096`、`heads=2`、`head_dim=50`，包含完整 forward+backward）：occurrence K/V materialization 为 3.43 ms，projected backward 为 2.22 ms，局部下降约 35.3%。该结果只说明反向 kernel 的方向有效，尚不能替代双机端到端训练 A/B。

完成双机八卡 WikiTalk 2 epoch 端到端复测后，结论如下：

| 配置 | epoch 0 | epoch 1（稳态） |
| --- | ---: | ---: |
| fused segment baseline | 9.22 s | 7.70–7.71 s |
| projected K/V + fused backward | 9.98–9.99 s | 7.93–7.94 s |

两侧固定 `alpha=0.3`、`beta=0.1`，local/remote node 和 edge 工作量逐 epoch 相同。projected K/V 的稳态端到端时间增加约 3.0%，epoch 0 还增加约 8.3%。因此，局部 kernel 的 35.3% 收益被 compact projection 的额外 GEMM、mapping/reorder、atomic gradient accumulation 和 kernel launch 开销抵消。当前版本判定为端到端无效，继续默认关闭，不并入有效方法。

对 recent 邻居的前缀/后缀复用不再推进：相同 destination 的邻居集合可能存在嵌套，但 attention 权重依赖当前 destination 的 Q，不能直接复用已经完成的 softmax 或聚合结果。projected K/V 的端到端复测也已失败，因此不再继续增加投影梯度归约或前缀复用复杂度。

### 第八阶段：Compact GRU gather-pointwise 融合

在保留 mailbox、time、hidden 三个 cuBLAS GEMM 的前提下，新增 Triton kernel，将两次 UID `index_select`、gate 相加和 GRU `r/z/n` pointwise 合并。forward 不再物化 occurrence 级 mail gate 和 hidden gate；backward 直接把 object gate 梯度归约回 UID 表。原 `GRUCell` 参数、state dict、DDP 参数和 checkpoint 格式不变，非 CUDA FP32 路径仍回退原实现。

WikiTalk 代表形状 `state=13865`、`UID object=1579`、`mail=372`、`time=100`、`hidden=100` 的 A40 局部测试：

| 范围 | 原 Compact GRU | gather-pointwise fused | 变化 |
| --- | ---: | ---: | ---: |
| forward | 0.464 ms | 0.332 ms | -28.5% |
| forward + backward | 1.056 ms | 0.845 ms | -20.0% |

CUDA FP32 输出与 time/GRU 参数梯度测试通过；FP64 回退测试通过。随后在双机八卡 WikiTalk、固定 `alpha=0.3`、`beta=0.1`、Fused Segment 已开启的 2 epoch 短测中：

| 配置 | epoch 0 | epoch 1（稳态） |
| --- | ---: | ---: |
| Fused Segment，Compact GRU off | 9.22 s | 7.70–7.71 s |
| Fused Segment，fused Compact GRU on | 9.13 s | 7.37–7.38 s |

两侧 local/remote node 和 edge 工作量逐 epoch 相同；短测稳态下降约 4.3%。该短测只用于触发完整验证，最终判断以下面的完整轮数结果为准。

#### 标准配置四数据集完整验证

固定 `alpha/SSIM=0.3`、`beta/probability=0.1` 和 seed 6773，在双机八卡、10GbE Socket 环境执行 WIKI 和 LASTFM 各 50 epoch、WikiTalk 和 Flight 各 10 epoch。基线为 `Fused Segment on + Compact GRU off`，实验组只增加 gather-pointwise Compact GRU；compact K/V 和 Triton edge projection 均关闭。因此下表是 Compact GRU 在 Fused Segment 链路上的增量收益，不能与第六阶段的 Fused Segment 收益直接相加。

两台机器统一使用 `tgnn_3.10`、CUDA 11.8 和 Triton 3.1.0。两机的 segment attention 与 Compact GRU 前后向测试各通过 10 项；本机完整相关测试为 18 项通过。

| 数据集 | epochs | full rank / critical 变化 | steady rank / critical 变化 | 尾段 critical 变化 | best-val 对应 test AP / AUC 变化（百分点） | final test AP / AUC 变化（百分点） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WIKI | 50 | -4.09% / -3.29% | -2.44% / -2.26% | -2.47%（最后 10 轮） | +0.1927 / +0.1452 | +0.1150 / +0.1366 |
| LASTFM | 50 | -3.93% / -3.94% | -4.40% / -4.40% | -4.28%（最后 10 轮） | +0.1624 / -0.0627 | -0.0219 / -0.2602 |
| WikiTalk | 10 | -3.06% / -3.09% | -2.93% / -2.97% | -3.14%（最后 4 轮） | -0.0376 / -0.0270 | -0.0376 / -0.0270 |
| Flight | 10 | -2.27% / -2.34% | -2.76% / -2.81% | -2.25%（最后 4 轮） | -0.0618 / -0.0474 | -0.0618 / -0.0474 |

其中 steady 表示排除 epoch 0，critical 表示每个 epoch 八个 rank 的最大训练时间。WIKI 稳态单轮仅约 0.16 s，而日志只保留到 0.01 s，因此其约 2.3%--2.5% 稳态信号精度较粗；full rank 使用未舍入的尾部 `avg_time`。其余三个数据集的稳态信号为约 2.8%--4.4%。

四个数据集的 local/remote node 和 edge 工作量均逐 epoch、逐 rank 完全一致，`memory comm` 均为 0。steady shared comm 在 WIKI、LASTFM、WikiTalk、Flight 分别变化约 `-0.11%`、`+5.63%`、`-0.47%`、`-1.34%`。LASTFM 在通信量增加时仍下降 4.40%，WikiTalk 的通信变化也不足以解释约 2.97% 的收益；Flight 存在约 1.34% 的有利通信变化，不能把其全部收益归因于 GRU kernel。

Time-to-target 采用每个数据集共同且自然的 validation AP 阈值：WIKI 95.0% 为持平；LASTFM 90.0% 的累计 critical 时间下降 1.30%，但晚一个 epoch 达标；WikiTalk 97.0% 下降 3.14%；Flight 96.0% 下降 2.29%。这些都是单 seed 观察，不能解释为稳定的收敛速度变化。

精度不能表述为严格无损。WIKI 是正向波动；WikiTalk 和 Flight 的最终 test AP/AUC 下降约 0.03--0.06 个百分点；LASTFM 最终 test AUC 下降 0.2602 个百分点。当前结论是：

> gather-pointwise Compact GRU 在 Fused Segment 已开启时，四数据集都出现了固定采样工作量下的端到端正效率信号，因此保留为统一执行链路的内部组件；本段仍是单 seed 的增量实验，最终直接组合与多 seed 结果见第十一阶段。

完整日志位于：

```text
outputs/compact_gru_fused_ab/20260818/8gpu_multidataset/
```

### 第九阶段：剩余 occurrence/temporal 去重空间验证

本阶段只测试不改变模型语义的公共子表达式删除与 temporal compaction，不新增 custom kernel。基线固定开启 Builder Segment、Fused Segment Attention 和 gather-pointwise Compact GRU，关闭 compact K/V 与 Triton edge projection；WikiTalk 双机八卡采用固定 `alpha=0.3`、`beta=0.1`、seed 6773，并保证 OFF/ON 的 local/remote node 和 edge 工作量逐 epoch 完全一致。

#### K/V 输入只物化一次

普通 occurrence K/V 路径原先对完全相同的 `state[edge_src] + edge feature + time feature` 分别执行两次 gather 和 `cat`。第一版只生成一次 `kv_input`，仍分别调用原有 `w_k` 和 `w_v`，不改变 GEMM、参数或 attention 语义。

2 epoch、OFF1 -> ON1 -> OFF2 -> ON2 的两组配对短测中，方法均值为：

| 指标 | OFF | ON | ON 变化 |
| --- | ---: | ---: | ---: |
| epoch 1 rank mean | 7.2294 s | 7.1625 s | -0.93% |
| epoch 1 critical | 7.2350 s | 7.1700 s | -0.90% |
| full exact rank mean | 8.0231 s | 7.8928 s | -1.62% |

稳态收益不足预设的 1% 保留门槛，且 full 口径受首次启动影响。本实现已经回退，不进入四数据集完整测试。日志位于：

```text
outputs/dedup_kv_input_ab/20260818/
```

#### Temporal-first 重排

该版本先在 raw `(UID, timestamp)` occurrence 上生成 temporal state，再只对 state 表执行 distributed UID mapper 与 UID unique。虽然 mapper gather 和 UID unique 的输入行数下降，新增的 raw-ID H2D、state-ID D2H/H2D 与二维 unique 开销更高。细分 profile 中 `to_block_total` 从 16.55 ms/batch 增至 17.08 ms/batch，即回退 3.23%。该版本已回退，不再推进 temporal-first。

#### Packed temporal key

保留 UID-first 顺序，只把二维 `float64 (compact UID, timestamp)` key 改为一维无碰撞 `int64` key，并使用原生 `scatter_reduce_(amin)` 恢复首次出现顺序。WIKI、LASTFM、WikiTalk、Flight 均满足 timestamp 为非负整数且位于 32-bit unsigned 范围、节点数位于 31-bit signed 范围；不满足约束的数据自动走旧路径。

单轮细分 profile 显示：

| 指标 | 双列 key | Packed key | 变化 |
| --- | ---: | ---: | ---: |
| key materialize + temporal unique + stable remap | 3.086 ms | 1.456 ms | -52.82% |
| `to_block_total` | 16.550 ms | 15.551 ms | -6.04% |

由于同轮 epoch 时间没有同步下降，继续执行了 profiler-off 的 OFF1 -> ON1 -> ON2 -> OFF2。ABBA 结果为：

| 指标 | OFF 方法均值 | ON 方法均值 | ON 变化 |
| --- | ---: | ---: | ---: |
| epoch 1 rank mean | 7.2238 s | 7.2513 s | +0.38% |
| epoch 1 critical | 7.2300 s | 7.2550 s | +0.35% |
| full exact rank mean | 8.0320 s | 7.9954 s | -0.46% |

稳态没有端到端收益，full 的微小正向值来自 epoch 0；因此 packed key、native stable remap 和临时 profiler 均已从生产代码回退。日志位于：

```text
outputs/temporal_first_profile/20260818/
outputs/temporal_packed_profile/20260818/
outputs/temporal_packed_abba/20260818/
```

本阶段结论是：现有 Python/GPU temporal unique 确实有局部算子空间，但单独替换 key 或调整 unique 顺序没有形成端到端收益。后续不再为 standalone temporal unique 编写自定义 kernel；第十一阶段的 P0 采用另一条路径，在 sampler 线程局部输出上并行 hash compaction 并稳定合并，直接绕过整段 post-sampling unique/remap。

### 第十阶段：硬件计数器诊断与保留 kernel 调优

本阶段按 `awesome-kernel-skills` 的 profiling、bottleneck diagnosis、verification 和 benchmark 流程，只分析已经保留的 Fused Segment Attention 与 gather-pointwise Compact GRU。新增统一局部入口 `tools/bench_retained_kernels.py`，先比较 forward 输出和 backward 梯度，再用 CUDA Event 执行交错多轮计时；NCU 只捕获 Triton forward/backward kernel。所有测试均使用 A40、`tgnn_3.10`、CUDA 11.8、Triton 3.1.0 和 FP32。

#### 初始硬件瓶颈

代表形状为：Segment `dst=6000, edge=108000, heads=2, head_dim=50, degree=18`；Compact GRU `state=13865, UID object=1579, mail=372, time=100, hidden=100`。

| kernel | forward DRAM / SM | backward DRAM / SM | achieved occupancy（forward / backward） | 结论 |
| --- | ---: | ---: | ---: | --- |
| Segment，4 warps | 58.60% / 33.01% | 70.62% / 54.00% | 40.42% / 40.18% | 88--89 registers/thread 限制 occupancy，并受访存延迟影响 |
| Compact GRU | 90.88% / 18.88% | 90.09% / 17.32% | 91.22% / 93.70% | 已经是明确的 DRAM bandwidth bound |

因此没有继续改写 GEMM 或增加计算融合：Segment 只测试 warp 配置；GRU 只删除生产训练中不需要的梯度写回。

#### Segment warp 配置

在连续 EID 口径下，局部完整 forward+backward 为：

| 配置 | 时间 | 相对 4 warps 变化 |
| --- | ---: | ---: |
| 2 warps | 0.9068 ms | +44.35% |
| 4 warps | 0.6282 ms | 基线 |
| 8 warps | 0.5252 ms | -16.40% |

随机 EID 压力测试中，4/8 warps 分别为 0.7843/0.7118 ms，8 warps 仍下降 9.24%。NCU 同口径对比显示，8 warps 将 forward registers/thread 从 89 降至 48、occupancy 从 40.42% 提高到 80.30%；backward registers/thread 从 88 降至 56、occupancy 从 40.18% 提高到 64.74%。forward/backward kernel duration 分别下降 13.42% 和 19.18%。

真实 WikiTalk 单机四卡前 32 个 block 的 edge permutation 统计显示，四个 rank 的原位比例仅为 0.39%--0.52%，但相邻 EID 连续率达到 97.88%--98.04%。这说明真实映射主要是连续区间换位，不能使用全随机 permutation 估计生产收益，也没有必要为此增加额外 CSC 重排或 K/V 拷贝。

随后执行 WikiTalk 双机八卡 10 epoch 的单次 4-warps -> 8-warps 配对；固定 `alpha=0.3`、`beta=0.1`、seed 6773，Builder Segment、Fused Segment 与 gather-pointwise Compact GRU 均开启，只改变 Segment 的 `num_warps`。两侧 80 个 rank-epoch 的 local/remote node 和 edge 工作量逐项完全一致。8 warps 的 shared-sync 数量在 steady 和最后四轮分别增加 0.20% 和 0.63%，动态状态轨迹不是逐 bit 相同。

| 统计范围 | 4 warps rank / critical | 8 warps rank / critical | 8 warps 变化 |
| --- | ---: | ---: | ---: |
| full | 7.4599 / 7.4650 s | 7.4576 / 7.4640 s | -0.03% / -0.01% |
| steady（epoch 1--9） | 7.2876 / 7.2933 s | 7.2638 / 7.2700 s | -0.33% / -0.32% |
| 最后四轮 | 7.2934 / 7.3000 s | 7.2719 / 7.2775 s | -0.30% / -0.31% |

两侧 loss 均有限且正常下降，但本轮使用 `--skip_eval`，不提供 AP/AUC 结论。虽然 8 warps 的 kernel 收益真实，端到端稳态仅约 0.32%，低于 1% 保留门槛，因此生产代码已恢复 4 warps，不扩展到其余三个数据集。

#### Compact GRU 无效梯度写回

生产路径中的 fetched `state_memory` 不参与梯度计算，但旧 backward 仍分配并写回 `[state, hidden]` 梯度。删除该输出后，GRU backward kernel 从 0.2084 ms 降至 0.1951 ms（-6.39%），DRAM write 从 48.85 MB 降至 41.14 MB（-15.79%）；包含三个 cuBLAS GEMM 的完整 Compact GRU forward+backward 从 0.8018 ms 降至 0.7865 ms（-1.91%）。同时，对显式要求 `state_memory` gradient 的非生产调用回退原生 GRUCell，补齐了该输入边界的梯度语义。

该项删除的是确定无效工作且简化了 kernel，因此仅作为内部清理保留；1.91% 是 Compact GRU 子路径的局部变化，不记为新的端到端效率贡献，也不触发四数据集长测。

本阶段结果位于：

```text
outputs/kernel_skill_loop/20260818/
```

当前停止判据是：Segment 的进一步 shared-memory layout 或直接 CSC 重排需要显著增加实现复杂度，而 8-warps 已证明局部 16.4% 只对应约 0.32% 端到端；GRU 的 UID-segment backward reduction 还需要额外排序/CSR，却只作用于约 0.20 ms 的 backward kernel。两者都不再沿 standalone kernel 微优化继续扩展。

### 第十一阶段：Sampler-Native Hierarchical Compaction（P0）与最终组合

#### P0 实现

legacy 路径在 sampler 返回 occurrence 后，再在 Python/GPU 侧分别执行 UID、EID 和 `(UID,timestamp)` unique、首次出现顺序恢复、metadata remap 和 Block 构造。P0 复用 sampler 已持有的每线程 `TemporalGraphBlock`：

1. 每个 sampler thread 独立对 temporal state 和 edge ID 做 hash compaction，无全局锁；
2. root state 先插入，再按 thread ID 和线程内首次出现顺序稳定合并；
3. UID/EID object 按 owner 分桶，并同步生成 state/edge 到 object 的 inverse mapping；
4. 一次输出 compact object/state 表、root mapping 和 occurrence COO；
5. Python 只负责 H2D 和 direct Block 构造，不再执行 legacy global unique/remap。

输出契约为：

```text
dist_nodes, dist_edges
state_to_node, state_timestamps, root_mapping
row, col, edge_mapping, delta_ts
num_src, num_dst
```

随机 parity 测试覆盖 50 组 recent 和 30 组 boundary-decay 输入；occurrence 数量与顺序、timestamp、`row/col/delta_ts`、EID 和 root mapping 均与 legacy 对齐。owner object 表的内部顺序可以不同，但 inverse mapping 能逐项恢复原 occurrence。相同 UID、不同 timestamp 仍是两个 temporal state，不改变采样概率、边的重数或训练目标。

当前 P0 只支持一层 temporal sampling、带负样本、非 identity/non-reversed 的当前 `TGN_large` 链路；其他配置 fail fast。P0 只在训练 sampler 开启。evaluation sampler 强制使用 legacy compaction，但仍使用 Fused Segment 和 Compact GRU；这是因为 P0 在 evaluation 小 batch 上没有摊薄固定编排成本。

#### P0 独立 ABBA

在 Fused Segment 与 gather-pointwise Compact GRU 两侧共同开启时，WikiTalk 双机八卡执行 `OFF1 -> ON1 -> ON2 -> OFF2`、每次 10 epoch 的 P0 独立 ABBA：

| 范围 | P0 OFF critical | P0 ON critical | 变化 |
| --- | ---: | ---: | ---: |
| full | 7.4780 s | 7.1865 s | -3.90% |
| steady（epoch 1--9） | 7.3328 s | 7.0389 s | -4.01% |
| 最后四轮 | 7.3250 s | 7.0250 s | -4.10% |

两组相邻配对的 steady critical 收益分别为 4.07% 和 3.95%，同配置漂移不超过 0.41%。local/remote node 和 edge 工作量逐 epoch 完全一致；steady shared count 方法均值约下降 0.96%，所以不能把全部 4.01% 都归因于 compaction，但收益明显大于通信轨迹变化。

#### 最终 all-off 对 all-on 直接 A/B

最终对比不叠加各模块的边际百分比，而是在统一源码上直接比较：

```text
all-off:
  hierarchical_compaction=off
  fused_segment_attention=off
  compact_gru_update=off

all-on:
  hierarchical_compaction=on（仅训练 sampler）
  fused_segment_attention=on
  compact_gru_update=on

两侧共同：
  compact_kv_projection=off
  triton_edge_projection=off
  alpha/SSIM=0.3
  beta/probability=0.1
```

环境为双机八卡 A40、10GbE、`tgnn_3.10`、CUDA 11.8 和 Triton 3.1.0。WIKI 使用 3 个 seed、每个 50 epoch；LASTFM 使用 seed 6773、50 epoch；WikiTalk 和 Flight 使用 seed 6773、各 10 epoch。steady 排除 epoch 0，tail 为 WIKI/LASTFM 最后 10 轮和 WikiTalk/Flight 最后 4 轮。

| 数据集 | full train critical | steady train critical | tail train critical | steady total-wall critical |
| --- | ---: | ---: | ---: | ---: |
| WIKI（3 seeds） | -2.29% ± 5.96% | **-9.67% ± 2.72%** | -9.33% ± 3.78% | **-7.07% ± 2.64%** |
| LASTFM | -11.48% | **-12.59%** | -11.86% | **-9.93%** |
| WikiTalk | -10.07% | **-10.97%** | -10.13% | **-7.93%** |
| Flight | -5.84% | **-7.97%** | -7.81% | **-5.69%** |

对应的绝对耗时如下；每个单元格均为 `all-off → all-on`。仓库现存的是 `gpu07` 上 rank 0--3 的 stdout；每轮训练计时结束前包含 8-rank barrier，因此这里取已归档四个 rank 的 epoch 最大值作为同步作业 critical 的可复现近似。另一节点的逐 rank stdout 未归档，无法重新计算严格的 8-rank 最大值；正式百分比沿用实验完成时记录的汇总值。逐 epoch 日志只打印到 `0.01 s`，下表四位小数来自多 epoch 平均，不代表单次计时具有四位小数精度。

| Run | full train critical（s/epoch） | steady train critical（s/epoch） | tail train critical（s/epoch） |
| --- | ---: | ---: | ---: |
| WIKI seed 6773 | 0.1842 → 0.1926 | 0.1624 → 0.1518 | 0.1610 → 0.1530 |
| WIKI seed 95842 | 0.1922 → 0.1822 | 0.1702 → 0.1508 | 0.1700 → 0.1500 |
| WIKI seed 12345 | 0.1896 → 0.1776 | 0.1692 → 0.1502 | 0.1690 → 0.1500 |
| LASTFM seed 6773 | 1.2784 → 1.1314 | 1.2600 → 1.1012 | 1.2480 → 1.1000 |
| WikiTalk seed 6773 | 7.9580 → 7.1570 | 7.8700 → 7.0067 | 7.7975 → 7.0075 |
| Flight seed 6773 | 2.5670 → 2.4180 | 2.4644 → 2.2689 | 2.4650 → 2.2700 |

| Run | full total-wall critical（s/epoch） | steady total-wall critical（s/epoch） | tail total-wall critical（s/epoch） |
| --- | ---: | ---: | ---: |
| WIKI seed 6773 | 0.2800 → 0.2926 | 0.2584 → 0.2480 | 0.2580 → 0.2510 |
| WIKI seed 95842 | 0.2880 → 0.2746 | 0.2661 → 0.2431 | 0.2620 → 0.2410 |
| WIKI seed 12345 | 0.2864 → 0.2708 | 0.2659 → 0.2431 | 0.2680 → 0.2400 |
| LASTFM seed 6773 | 2.1666 → 1.9646 | 2.1476 → 1.9339 | 2.1080 → 1.9190 |
| WikiTalk seed 6773 | 12.8800 → 11.9300 | 12.7933 → 11.7800 | 12.7100 → 11.7775 |
| Flight seed 6773 | 4.1270 → 3.9480 | 4.0233 → 3.7944 | 4.0300 → 3.8000 |

`total-wall` 包含每轮 validation 和 test，不只是训练计时。四个数据集的 steady critical train 与 total wall 全部下降；WIKI 的 full 方差来自短 epoch 下首次 Triton 编译和初始化，三个 seed 的 steady/tail 均为正收益。

质量变化均以百分点表示：

| 数据集 | best-val 对应 test AP / AUC 变化 | final test AP / AUC 变化 |
| --- | ---: | ---: |
| WIKI（3 seeds） | -0.0687 ± 0.2062 / -0.0883 ± 0.2469 | -0.1349 ± 0.2935 / -0.1262 ± 0.2976 |
| LASTFM | -0.1556 / -0.2869 | -0.1556 / -0.2869 |
| WikiTalk | -0.0246 / -0.0364 | -0.0246 / -0.0364 |
| Flight | +0.0008 / -0.0051 | +0.0008 / -0.0051 |

WIKI 三个 seed 的 best-checkpoint test 最坏变化为 AP -0.3055、AUC -0.3726 个百分点，final 最坏为 -0.4710/-0.4684 个百分点。native dropout RNG 对齐后，WIKI 三 seed 的 validation/test 轨迹 MAE 与 RMSE 比旧 unmatched 版本下降约 24%--29%，best-test 跨 seed 平均回退缩小到 0.1 个百分点以内。当前数据通过“WIKI best-test 均值回退小于 0.1 个百分点、所有已测 run 最坏回退小于 0.5 个百分点”的工程检查线；但不能表述为精度完全无损或统计非劣，LASTFM、WikiTalk 和 Flight 也仍只有单 seed。

#### 工作量、通信与收敛

- 每个数据集和 seed 的 local/remote node、local/remote edge 工作量在 OFF/ON 间逐 epoch、八个 rank 完全一致；`alpha/beta` 固定为 `0.3/0.1`，`memory comm` 始终为 0。
- steady shared count 的变化为：WIKI `+0.01%`（三 seed 均值）、LASTFM `-1.40%`、WikiTalk `-0.31%`、Flight `-0.88%`。后三个数据集存在轻微有利变化，因此不能把全部时间下降归因于计算去冗余；变化幅度仍明显小于 7.97%--12.59% 的训练收益。
- 共同 validation AP 阈值下，含评估 TTT 的 OFF→ON 累计时间为：LASTFM `78.60→75.18 s`（-4.34%）、WikiTalk `103.45→95.79 s`（-7.42%）、Flight `33.18→31.86 s`（-3.98%）；WIKI 三 seed 算术均值为 `11.36→11.78 s`，paired 变化为 +3.96% ± 8.39%，没有稳定收敛加速。最终结论应强调每 epoch 吞吐，而不是声称四数据集都改善收敛。
- 所有 run 均完整结束，loss 有限；NCCL 使用 `NET/Socket` 与 `ens4f1np1`，`NET/IB` 为 0，GPU Direct RDMA 关闭。

#### 当前最终判断

> Sampler-Native Block-Unique Compact Execution 在固定采样工作量下，把 sampler-side hierarchical compaction、builder segment plan、Fused Segment Attention 和 UID-aware Compact GRU 合并成一条执行链。all-off 对 all-on 的直接 A/B 在四数据集上取得 7.97%--12.59% 的 steady 训练关键路径下降，以及 5.69%--9.93% 的含评估整轮 wall-time 下降。

当前版本作为最终工程候选保留，不再继续 standalone projected K/V、edge projection、K/V 公共输入、warp 数或 temporal-key 微优化。质量只能表述为“当前最坏回退小于 0.5 个百分点，WIKI 三 seed best-test 均值回退小于 0.1 个百分点”；正式论文结论仍需为 LASTFM、WikiTalk 和 Flight 补足多 seed。

P0 和最终组合日志位于：

```text
outputs/p0_hierarchical_8gpu_20260818/
outputs/p0_hierarchical_full_8gpu_20260818/
```
