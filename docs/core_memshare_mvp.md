# CoRe-MemShare MVP 设计与实现说明

本分支在固定图划分、固定热点集合和固定 ownership 的前提下，增加一条可逐项启用的实验链路：

```text
last-fetched memory residual
        + edge prediction error
        + cache-aware communication cost
                    |
                    v
communication-budgeted boundary sampling
        + grouped coordinated randomness
                    |
                    v
probability-corrected attention
                    |
        +-----------+-----------+
        |                       |
probability-aware edge cache    cost-aware hard negatives
        |                       |
        +-----------+-----------+
                    |
             unified request plan
```

所有 CoRe 开关默认关闭。未指定新参数时，训练保持原 MemShare 路径。

## 1. 当前已经落地的内容

### 1.1 残差与通信代价感知的边界采样

原生 sampler 新增策略：

```text
boundery_recent_core
```

对 root 的 top-fanout 远程候选边，计算：

\[
\widehat\rho_i =
\mathrm{temporal}_i
\left(
\beta_u U_i+
\beta_a A_i+
\beta_e E_i
\right),
\]

\[
c_i =
c_i^{mem}
+c_i^{request}
+\mathbb I[e_i\notin Cache]c_i^{edge},
\]

\[
p_i=\operatorname{clip}
\left(
\lambda\frac{\widehat\rho_i}{\sqrt{c_i}},
p_{\min},1
\right).
\]

每个 root 内通过二分搜索求解 \(\lambda\)，使期望通信成本满足由
`--probability` 给定的预算比例。采样器支持两类稀疏在线更新：

- `update_core_node_state`：更新 last-fetched memory 的观测残差与获取时间；
- `update_core_edge_state`：更新边预测误差与 cache 命中状态。

未观测节点和边使用保守默认值，因此自然退化到时间衰减采样。

### 1.2 分组协同采样

相同 group 内的相同远程对象共享由

```text
(object_id, group_id, batch_id, layer, seed)
```

确定的随机数。每个 occurrence 的边际采样概率仍为 \(p_i\)，而同组内对象的
唯一请求概率降为 \(\max_i p_i\)。`--core_group_size` 控制 group 大小。

### 1.3 概率校正 attention

原生 sampler 会把选中边的 inclusion probability 写入 MFG 的：

```text
block.edata["sample_prob"]
```

attention 层使用：

\[
\operatorname{softmax}(\ell_i-\log p_i),
\]

等价于分别对零 baseline 下的 softmax 分子与分母使用
Horvitz--Thompson 修正，再计算 self-normalized ratio。

CoRe 路径只在固定的 top-fanout candidate set 内做随机选择，不会在远程候选
被拒绝后用更老的本地边回填；否则回填边的 inclusion probability 取决于前序
随机事件，不能再写成 \(p=1\)。原有采样策略不注入 `sample_prob`，因此不开启
CoRe 时不会改变 attention 行为。

### 1.4 last-fetched memory predictor

`LastFetchedMemoryPredictor` 提供分级回退：

1. 节点 last-fetched memory；
2. owner memory centroid；
3. zero baseline。

它使用固定容量 LRU，只保存最近获取的 memory，并维护预测 RMS residual EMA。
DataLoader 在远程 memory 返回后自动观测 residual，并稀疏更新原生 sampler。

### 1.5 概率与方差收益感知的 edge cache

`ProbabilityAwareEdgeCache` 是固定容量的机会式静态缓存。边被真实拉取后，其
admission score 为：

\[
S_e =
\frac{
p_e\cdot c_e^{edge}\cdot reuse_e\cdot bytes_e
+\lambda_v\Delta V_e
}{bytes_e}.
\]

实现包含：

- score EMA；
- 最小驻留 batch 数；
- admission hysteresis；
- 每 batch 最大替换数；
- object hit rate、byte hit、eviction 和显存开销统计。

pipeline 路径会把本地边、cache hit 和 remote miss 分开，仅对 remote miss
发起 All-to-All。edge feature 不可变，因此不需要版本一致性协议。

### 1.6 通信感知 hard negative

每个负样本 slot 先从原目标分布 \(q_0\) 独立产生 \(r\) 个候选，再使用：

\[
q(j\mid C)
\propto
\frac{H_j^\beta}{\sqrt{c_j+\epsilon_c}}
\]

进行候选 occurrence 重采样，并与 uniform proposal 做
\(\epsilon\)-mixture。

最终权重为：

\[
w_j=\frac{1/r}{q(j\mid C)}.
\]

给定候选池 \(C\)，它无偏估计候选池均值；再对候选池取期望后，严格恢复
原 \(q_0\) 负样本目标。该结果不依赖全局 ANN，也不把有限候选池近似误写成
全局 proposal 概率。

当前 hard score 使用最近缓存的 root/negative embedding cosine score。
本地、embedding cache hit 和正样本 root 已请求节点使用更低通信成本。

### 1.7 统一请求 planner

`UnifiedRequestPlanner` 在 fetch 前按对象类型统计并去重：

- static node feature；
- dynamic node memory；
- immutable edge feature。

随后按 owner 分桶，并从 edge 请求中排除 cache hit。master 的 MFG builder
本来已经对正负 root 和邻居做一次唯一化；planner 将这个事实显式化并报告：

- raw/unique request 数；
- node/edge dedup ratio；
- remote request 数；
- cache 消除的 edge request 数。

## 2. 构建与测试

推荐使用与训练环境一致的 Python 3.10：

```bash
PYTHON_BIN=/home/zlj/.miniconda3/envs/tgnn_3.10/bin/python \
BUILD_DIR=/tmp/core_memshare_native_build \
bash scripts/build_native_sampler.sh
```

无需覆盖已有 `.so`，可通过环境变量选择刚编译的 sampler：

```bash
export STARRYGL_SAMPLER_LIBRARY=/tmp/core_memshare_native_build/libstarrygl_sampler.so
```

运行算法和 native smoke tests：

```bash
CORE_MEMSHARE_NATIVE_LIB=/tmp/core_memshare_native_build/libstarrygl_sampler.so \
/home/zlj/.miniconda3/envs/tgnn_3.10/bin/python -m pytest -q \
tests/test_core_memshare.py tests/test_core_native_sampler.py \
tests/test_core_distributed_cache.py
```

## 3. MVP 训练示例

从 `examples/` 目录启动，保留原有分区、热点和 All-to-All：

```bash
export STARRYGL_SAMPLER_LIBRARY=/tmp/core_memshare_native_build/libstarrygl_sampler.so

torchrun --standalone --nproc_per_node=4 train_boundery.py \
  --model TGN_large \
  --dataname WIKI \
  --partition divided_nodes_seed_starrygl \
  --topk 0.1 \
  --sample_type boundery_recent_core \
  --probability 0.1 \
  --memory_type historical \
  --core_min_probability 0.02 \
  --core_group_size 8 \
  --core_memory_cost 1.0 \
  --core_edge_cost 1.0 \
  --core_request_cost 0.05 \
  --core_edge_cache_ratio 0.01 \
  --core_edge_cache_mb 512 \
  --core_neg_pool_factor 4 \
  --core_neg_hardness_beta 1.0 \
  --core_neg_exploration 0.1 \
  --core_request_planner \
  --core_smoke_batches 10
```

`--core_smoke_batches N` 在完成 \(N\) 个训练 batch 后做跨 rank 同步并退出，
跳过验证、测试和 checkpoint，适合先检查真实分区上的 collective 与数据布局；
设为 0 时执行完整训练。

### 3.1 四卡真实分区 smoke 记录

2026-07-29 在 WIKI、固定 4-way partition、4 张 A40 上启用完整组合并消费 1 个
训练 batch，原生采样、cache miss-only All-to-All、hard negative、MFG、
forward/backward 和 optimizer step 均正常完成，进程返回 0。rank 0 的启动期
统计为：

- unique node/edge requests：3,556 / 10,738；
- remote node/edge misses：1,070 / 3,184；
- edge cache：3,367 次查询、183 次命中、125,904 bytes saved；
- negative ESS：1,776.99，remote ratio 0.1357，positive-request reuse ratio
  0.3878。

这是 correctness smoke，不是性能结论。pipeline 会预取下一个 batch，因此
planner/cache 的通信统计包含 2 个 request plan，而模型只消费了 1 个 batch；
首次 cache 仍处于冷启动状态。

建议第一次实验只打开一个模块：

1. `boundery_recent_core`，cache ratio 为 0，negative pool factor 为 1；
2. 在 1 的基础上打开 edge cache；
3. 单独打开 hard negative；
4. 最后运行完整组合。

## 4. 当前边界与下一阶段

MVP 已经用 last-fetched residual 调整 \(p_i\)，并把 \(p_i\) 用于 attention
分子/分母校正；但原生 sampler 仍只返回被选中的远程邻居。因此当前 attention
使用的是 **zero-baseline probability correction**，尚未把未采样邻居的
非零预测贡献 \(\bar y_i\) 放进 MFG。

完整 prediction-corrected residual aggregation 的下一步是：

1. sampler 同时返回 candidate、selected mask 与 \(p_i\)；
2. fetch planner 只拉取 selected remote object；
3. 未选中对象从 last-fetched memory 和 edge cache 填充预测分支；
4. attention 分别构造实际与预测分子/分母；
5. 使用已经实现并测试的 `residual_ratio_aggregate` 做
   \(\bar A+z(A-\bar A)/p\) 与 \(\bar Z+z(Z-\bar Z)/p\)。

这一步会改变 MFG/fetch 数据布局，适合在当前 MVP 获得残差分布、cache hit
和端到端收益证据后再做，避免一次性扩大正确性风险。

另外还有三项执行层边界：

- hard-negative sampler 已排除当前 batch 的正目标，并暴露
  `validity_filter(source, candidate, timestamp)`；训练脚本尚未构建全历史
  temporal-positive 索引，因此若数据集协议要求排除全部历史正边，需要接入该
  callback；
- request planner 当前完成 typed dedup、owner bucketing 与统计，edge cache
  已实际执行 miss-only All-to-All；node feature 与 dynamic memory 仍沿用
  MemShare 的合并路由，尚未实现 version-aware JIT memory 协议；
- 当前合并粒度是 trainer/MFG，尚未增加同机多 GPU 的 machine-level request
  aggregator。
