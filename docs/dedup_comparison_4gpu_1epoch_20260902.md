# 全链路不去重、普通不去重、普通去重与新结构：4 卡单 epoch 对照

## 实验口径

- 日期：原三组为 2026-09-02；补充的普通不去重为 2026-09-03。
- 硬件：均为单机 4 × NVIDIA A40 44 GiB；原三组运行于 gpu07，普通不去重运行于 gpu06。这不是此前讨论的 8 卡结果。
- 数据集：LASTFM、WikiTalk、StackOverflow、GDELT，均使用完整训练 split。
- 模型：`TGN_large`、`APAN_large`。
- 共同参数：每 rank batch size 3000，fanout 20，4-way `topk=0.1` partition，seed 6773，`boundery_recent_decay`，probability 0.1，historical memory，shared-memory SSIM 0.3；采样参数固定为 `alpha=0.3, beta=0.1`。
- 只执行 epoch 0；关闭 validation/test/checkpoint。表中数值为四个 rank 的最大 `train time`，即同步临界 rank 的秒/epoch。
- 计时包含 epoch 0 内首次执行的新 kernel/JIT 成本，不包含约数秒到二十余分钟的数据加载、图构建和模型初始化。
- 每个配置只跑一次，不提供方差或置信区间，因此结论只针对本次 epoch-0 开销。

## 四组定义

1. 全链路不去重（极端消融）：`--block_dedup off`，状态设置继承对象设置。节点、eid 对象请求和 $node_{u,t}$ 状态均不去重，只按 owner partition 排序以满足 all-to-all 路由。此前表中的“完全不去重”实际是这一口径，不应作为主要的不去重基线。
2. 普通不去重：`--block_dedup on --state_dedup off`。通信与对象读取仍对节点/eid 请求去重，只是不合并相同的 $node_{u,t}$ 状态。这是本次补充的基线。
3. 普通去重：`--block_dedup on --state_dedup on`，其余新结构开关关闭，即原 legacy block dedup。
4. 新结构：
   - TGN：普通去重 + `--hierarchical_compaction on --fused_segment_attention on --compact_gru_update on`。
   - APAN：普通去重 + `--compact_memory_ops on`，使用 compact Transformer memory consumer。APAN 是 identity GNN、no-negative/reversed 链路，当前不支持 TGN 的 hierarchical sampler 与 fused GNN，因此 APAN 行不是完整的 TGN all-on 结构。

## 结果

`普通去重 vs 普通不去重` 和 `新结构 vs 普通去重` 中负值表示更快，正值表示更慢。普通不去重运行于 gpu06，其余三组运行于 gpu07；两台机器的 GPU 型号和配置相同，但下列跨主机百分比只能作为初步趋势，严格加速比需要把其余三组也在 gpu06 重跑。

| 数据集 | 模型 | 全链路不去重 gpu07 (s) | 普通不去重 gpu06 (s) | 普通去重 gpu07 (s) | 新结构 gpu07 (s) | 普通去重 vs 普通不去重* | 新结构 vs 普通去重 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LASTFM | TGN | 6.888850 | 4.700926 | 3.113864 | 3.821801 | -33.76% | +22.74% |
| LASTFM | APAN | 2.919430 | 2.595005 | 2.471852 | 3.234385 | -4.75% | +30.85% |
| WikiTalk | TGN | 29.577611 | 19.195261 | 13.514850 | 12.361320 | -29.59% | -8.54% |
| WikiTalk | APAN | 16.011285 | 13.327800 | 13.227604 | 12.719332 | -0.75% | -3.84% |
| StackOverflow | TGN | 260.095587 | 169.796244 | 126.503458 | 113.054028 | -25.50% | -10.63% |
| StackOverflow | APAN | 123.919901 | 106.414035 | 104.798810 | 96.072215 | -1.52% | -8.33% |
| GDELT | TGN | 1670.200417 | 719.173104 | 519.938158 | 443.632504 | -27.70% | -14.68% |
| GDELT | APAN | 316.206557 | 284.650810 | 250.313156 | 245.874972 | -12.06% | -1.77% |

四条原始 rank 时间分别见 `outputs/dedup_compare_4gpu_20260902/results.csv` 和 `outputs/ordinary_no_state_dedup_gpu06_20260903/results.csv`，完整 stdout/stderr 见对应目录各配置日志。

## 结论

- 原来的“完全不去重”同时关闭了通信/对象请求去重，属于极端消融。GDELT/TGN 从该口径的 1670.200417 s 变为普通不去重的 719.173104 s，说明此前的大幅差距不能全部归因于 $node_{u,t}$ 状态去重。
- 以普通不去重为基线，普通去重的初步跨主机对比中，TGN 降低 25.50%–33.76%，更接近状态重复本身的收益；APAN 降低 0.75%–12.06%，说明 APAN 当前 identity/reversed 路径从状态合并中获得的收益较小。
- TGN 新结构在 WikiTalk、StackOverflow、GDELT 上相对普通去重继续降低 8.54%、10.63%、14.68%；但 LASTFM 首个 epoch 反而慢 22.74%。
- APAN compact Transformer 在 WikiTalk、StackOverflow、GDELT 上相对普通去重继续降低 3.84%、8.33%、1.77%；但 LASTFM 首个 epoch 慢 30.85%。
- 因而当前结果不能表述为“新结构全面优于普通去重”。更准确的结论是：规模较大时新结构能摊薄固定成本并继续获益；LASTFM 这种短 epoch 上，epoch-0 编译/调度与 compact consumer 固定成本超过节省量。
- 本实验只有 epoch 0，不能计算 `steady train critical` 或 `tail train critical`；要得到稳态结论必须额外跑多个 epoch，并从中段/尾段取统计量。
- 标有 `*` 的普通去重相对普通不去重百分比跨 gpu06/gpu07，只能用于判断量级，不能作为最终论文加速比。
- 此处 4 卡结果不能与此前提到的 8 卡 WikiTalk 约 7 s、LASTFM 约 2 s 直接比较；GPU 数、partition 数和执行链路不同。
