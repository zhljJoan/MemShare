# Coordinated Boundary-Decay 独立验证计划

更新时间：2026-08-03

当前分支：`coordinated-sampling-test`

## 1. 分支边界

本分支从 `master` 的 `454ff32` 直接创建，只验证 coordinated
boundary-decay。以下机制均不引入：

- 残差预测或 residual-aware probability；
- edge feature cache；
- hard negative sampling；
- unified request planner；
- 动态热点、迁移或重划分。

baseline 使用 `boundery_recent_decay`，实验方法使用
`boundery_recent_decay_coordinated`。两者共享完全相同的候选集合和 occurrence
级概率公式，只改变 Bernoulli 随机数的生成方式。

## 2. 方法定义

独立采样对每个 root-occurrence 使用独立随机数：

$$
z_{r,i}=\mathbb I[U_{r,i}<p_{r,i}].
$$

协调采样在一个 root group 内，对相同远程节点对象使用由 batch、layer、group
和对象 ID 确定的共享随机数：

$$
U_{b,l,g,i}=\operatorname{HashUniform}(b,l,g,i),
$$

$$
z_{r,i}=\mathbb I[U_{b,l,g,i}<p_{r,i}].
$$

这里并没有把较小的 `p_{r,i}` 改成组内最大概率。每个 occurrence 仍与自己的
`p_{r,i}` 比较，因此：

$$
\Pr(z_{r,i}=1)=p_{r,i}.
$$

如果同一对象的两个 occurrence 满足 `p_1 < p_2`，则共享随机数只会使
`z_1 <= z_2`，而不会把 `p_1` 提升到 `p_2`。该对象至少请求一次的概率由独立
采样的

$$
1-(1-p_1)(1-p_2)
$$

变为

$$
\max(p_1,p_2).
$$

## 3. 当前代码验证

原生测试覆盖两种情况：

1. 两个 occurrence 的概率均为 0.4：验证两者的经验边际概率都保持 0.4，
   coordinated 的唯一对象请求率约为 0.4，独立采样约为 0.64；
2. 同一对象的 occurrence 概率分别约为 0.622 和 0.378：验证两者经验概率不变，
   且低概率 occurrence 的选中集合是高概率 occurrence 的子集。

执行方式：

```bash
PYTHON_BIN=/home/zlj/.miniconda3/envs/tgnn_3.10/bin/python \
BUILD_DIR=/tmp/memshare-coordinated-sampler-build \
bash scripts/build_coordinated_sampler.sh

COORDINATED_SAMPLER_NATIVE_LIB=/tmp/memshare-coordinated-sampler-build/libstarrygl_sampler.so \
/home/zlj/.miniconda3/envs/tgnn_3.10/bin/python -m pytest -q \
tests/test_coordinated_sampler.py
```

当前结果：`2 passed`。

## 4. 双机八卡验证矩阵

网络固定为两机八卡、10GbE Socket，并设置：

```text
NCCL_IB_DISABLE=1
NCCL_NET=Socket
NCCL_SOCKET_IFNAME=ens4f1np1
GLOO_SOCKET_IFNAME=ens4f1np1
```

完整训练轮数：

| 数据集 | Epoch |
| --- | ---: |
| WIKI | 50 |
| LASTFM | 50 |
| WikiTalk | 10 |
| Flight | 10 |

baseline/coordinated 必须使用相同 seed 配对，至少执行三个 seed。group size 测试
`1/4/8/16`；其中 group size 1 是 keyed-random 控制组，不提供组内复用。

效率指标包括：

- 每 epoch 训练时间和 time-to-target-AP；
- sampler、AllToAll route、AllToAll launch、暴露的 fetch wait；
- unique remote node/edge、真实网卡字节和 p95 batch latency。

质量指标包括：

- 每 epoch validation/test AP、AUC；
- 最终 AP、AUC 的百分点变化；
- 三 seed 均值、标准差和收敛轨迹。

在双机八卡完整训练完成前，本分支只证明概率语义和实现隔离正确，不把该方法定为
最终期刊方法。
