# CoRe-MemShare 四数据集实验记录

更新时间：2026-07-29

代码分支：`core-memshare`

固定数据集：WIKI、LASTFM、WikiTalk、Flight

## 1. 当前结论

本轮没有把任何 CoRe 算法变体升级为默认训练方案。原始 MemShare
`boundery_recent_decay` 仍是当前保留方案。

保留的是可复用且已经验证的实验基础设施与正确性修复：

- 强制覆盖四个数据集的矩阵运行器和结构化指标；
- 对比实验冻结 adaptive sampling/synchronization 参数，避免算法与预算互相
  改写；
- Flight 空 edge batch 保护；
- predictor 只观测真实远程获取节点；
- edge cache 批量写入与 O(1) 空槽管理；
- hard-negative 候选恢复原 localism proposal，并加入两阶段 importance
  correction；
- 17 个 sampling、cache、negative、planner、MFG 和 native sampler 回归测试。

下列模块保留为 **默认关闭的实验代码**，只用于复现和下一轮重构，不属于当前
推荐组合：

- 在线 last-fetched residual feedback；
- probability-aware edge cache；
- communication-aware hard negative；
- static communication-aware sampler；
- newest top-1 deterministic protection；
- 上述模块的完整组合。

原因不是这些模块没有局部通信收益，而是它们没有同时通过四数据集的端到端
时间与训练质量门槛。

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

第一次 30-batch matrix 的动态 adaptive 参数会让慢模块在后续 batch 得到不同
的 `alpha/beta`，看上去像是模块节省了大量通信。加入
`--core_freeze_adaptive_parameters` 后才消除这一混杂，因此冻结前的数据只
记录为失败尝试。

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

## 7. 保留与暂缓清单

| 项目 | 状态 | 依据 |
| --- | --- | --- |
| 原 MemShare baseline | **保留** | 四数据集质量基准，所有 CoRe 开关默认关闭 |
| 四数据集 matrix + JSON metrics | **保留** | 防止单数据集和动态预算混杂 |
| correctness/performance 修复 | **保留** | 17/17 tests；减少无意义 Python/cache 工作 |
| online residual predictor | 暂缓 | 通信下降但 batch 时间 +50.08% |
| probability-aware edge cache | 暂缓 | 命中有效但 batch 时间 +200.4% |
| hard negative | 暂缓 | batch 时间 +238.15%，ESS 仅 3.79%–12.22% |
| core-static | 暂缓 | 系统收益稳定，WIKI test AP -0.01012 |
| core-top1 | 暂缓 | WIKI test AP -0.02214，WikiTalk 通信增加 |
| full CoRe combination | 暂缓 | 组成模块尚未分别通过门槛 |

“暂缓”表示不进入推荐配置和后续长训练，不表示删除复现实验入口。下一轮只有在
完成 vectorized predictor/cache、真正的非零预测残差聚合、以及更稳定的
attention tail 校正后，才重新进入同样的四数据集门槛。

## 8. 复现实验

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
  --variants baseline residual core-static edge-cache residual-cache \
             hard-negative full \
  --batches 30 \
  --repeats 1 \
  --native-lib /tmp/core_memshare_native_build/libstarrygl_sampler.so \
  --run-name core-screen
```

运行带 validation/test 的三 epoch 对照：

```bash
/home/zlj/.miniconda3/envs/tgnn_3.10/bin/python \
scripts/run_core_memshare_matrix.py \
  --variants baseline core-static core-top1 \
  --screen-epochs 3 \
  --repeats 1 \
  --native-lib /tmp/core_memshare_native_build/libstarrygl_sampler.so \
  --run-name core-quality
```

原始日志和 metrics 默认写入被 git 忽略的
`outputs/core_memshare_screen/<run-name>/`；`matrix_summary.json` 中
`complete=true` 才表示本次四数据集实验有效。
