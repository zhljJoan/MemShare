# MemShare Batch 去重与分层紧凑执行：最终方案设计

## 1. 文档定位

本文描述的是 **MemShare Batch 去重方案的升级版本**，不是一个新的 GNN 架构或独立系统。它只保留经过当前完整轮数测试与已有统计验证后的工程候选；历史探针、失败原型和逐阶段 A/B 见 [调研记录](journal_extension_dual_dedup_probe.md)。面向组会叙事的短版见 [MemShare Batch 去重与分层复用：组会叙事底稿](docs/memshare_batch_dedup_group_meeting_source.md)。

推荐阅读顺序：只想理解方案时，先看第 2、3、8 和 12 节；第 4 至 7 节是代码契约与公式；第 13 节只是失败方案记录。

正文统一把它称为：

> **MemShare Batch 分层紧凑执行升级**。

这只是对 MemShare Batch 表示和执行链的描述，不作为新的架构名称。

它不修改 MemShare 的图划分、热点集合、节点 ownership、邻居采样概率或负样本分布。优化对象是采样结果生成之后的训练执行链：避免 sampler、Block builder、memory updater 和 attention 对同一批对象重复执行 unique、映射、展开与短 kernel 编排。

最终方案只作为一个整体技术点：

> 在继承原 MemShare `(UID,timestamp)` non-redundant Block 语义的基础上，把它提升为 sampler 直接生成、由下游共同消费的 object/state/occurrence 分层表示；memory update 与 segment attention 再分别按不改变语义的最小粒度执行紧凑计算。

实现上包含四个连续阶段，但不把它们分别计作四项贡献：

1. **采样器内分层整理（P0，仅训练）**：在 sampler 输出处生成 object/state 表和索引映射；
2. **直接 Block 构造与目标节点分组索引**：把同一套 state/occurrence 映射变成训练 Block 和 segment plan；
3. **按节点复用的 Compact GRU**：在 object/state 两级执行 memory update；
4. **分段融合注意力（Fused Segment Attention）**：在 state/occurrence 两级完成目标节点聚合。

未保留的实验方案统一列在第 13 节，不作为理解最终链路的前置条件。

## 2. 名词约定与统一设计原则

### 2.1 一句话理解

> 该方案不删除任何采样边或训练样本。它只把重复的节点、图边和 `(UID,timestamp)` 状态各存成一张紧凑表，再用整数映射恢复原 batch；GRU 复用节点级计算，attention 保留每条采样边，但把多个短算子融合执行。

最终必须同时满足：

1. 同一节点或图边的基础数据只远程获取一次；
2. 同一 UID 在不同 timestamp 下仍是不同状态；
3. sampler 产生的每次边使用都保留：顺序不变，同一条边原来出现几次就仍保留几次；
4. 只有与 timestamp 无关的计算才能按 UID 复用。

### 2.2 一个最小例子

假设一个 batch 中有三条采样记录。每行的 UID/timestamp 表示采样边的 source temporal state；`destination` 是接收该边消息的目标 root state：

| 记录 | source UID | source timestamp | 图边 EID | 目标 root（destination） |
| --- | ---: | ---: | ---: | --- |
| A | 42 | 100 | 7 | root-0 |
| B | 42 | 100 | 7 | root-1 |
| C | 42 | 120 | 7 | root-0 |

这三条记录对应：

- `1` 个节点对象：UID 42；
- `1` 个图边对象：EID 7；
- `2` 个时序节点状态：`(42,100)` 和 `(42,120)`；
- `3` 个采样边实例：A、B、C 全部保留。

A 和 B 可以共享节点特征、edge feature 和 `(42,100)` 的状态计算，但不能合成一条边，因为它们属于不同 destination，并分别参与 softmax、dropout 和聚合。C 的 UID 相同但 timestamp 不同，因此不能与 A/B 共享完整 memory update 结果。

### 2.3 核心数据名词

后文固定使用下表中的含义：

| 名词 | 代码中的主要表示 | 通俗含义 | 合并边界 |
| --- | --- | --- | --- |
| `UID` / `EID` | 原图节点 ID / 图边 ID | 图中对象的长期身份，不包含本 batch 的时间和位置 | ID 相同只说明基础对象相同 |
| **distributed ID** | `nid_mapper[UID]` / `eid_mapper[EID]` | 把 raw UID/EID 转成“owner rank + owner 内局部行号”的分布式地址 | 只改变寻址方式，不改变图对象身份 |
| `owner` | distributed ID 中的 partition | 真正存放该节点或图边数据的 GPU/进程 | 决定本地读取还是 remote fetch，不决定能否合并 state |
| **节点对象**（UID object） | `dist_nodes[j]` | 本 batch 中一个去重后的节点 distributed ID | 静态 node feature、节点动态 memory、mailbox 的 fetch 可合并 |
| **图边对象**（edge object） | `dist_edges[j]` | 本 batch 中一个去重后的图边 distributed ID | 不可变 edge feature 的 fetch 可合并，采样边实例不能删除 |
| **时序节点状态**（temporal state） | `state_to_node[s]`、`state_timestamps[s]` | 唯一的 `(UID,timestamp)` 组合 | 完全相同的 `(UID,t)` 可合并；同 UID、不同时间不可合并 |
| **root** | `seed[i]`、`seed_ts[i]` | sampler 围绕其取邻居的一个目标节点出现项，包含正样本端点和负样本端点 | 重复 root 可指向同一个 state，但在标签、decoder 和 loss 中的角色不能删除 |
| **采样边实例**（edge occurrence） | `row[k]`、`col[k]`、`edge_mapping[k]`、`delta_ts[k]` | sampler 在当前 batch 中对一条图边的一次实际使用 | 始终保留顺序和重复出现次数；即使 UID/EID 相同也不能直接去重 |
| **state-to-UID mapping** | 正文 `state_to_uid`；P0 输出 `state_to_node`；公式 `m(s)` | temporal state 指向节点对象行号 | 同一逻辑映射的不同代码名字，不是三份数据 |
| **root-to-state mapping** | P0 输出 `root_mapping` | root occurrence 指向 temporal state 行号 | 可复用 state 计算，但不能删除 root 在 loss 中的角色 |
| **edge-to-object mapping** | P0 输出 `edge_mapping` | edge occurrence 指向图边对象行号 | 可复用 edge feature fetch，但不能删除 occurrence |
| `memory` / `mailbox` | owner 上的节点状态 / 待写入 memory 的交互消息 | MemShare memory updater 的持久状态和输入消息 | fetch 可按 UID 合并；时间相关 update 结果仍是 state 级 |

### 2.4 执行名词

| 名词 | 本文中的固定含义 |
| --- | --- |
| **compaction（紧凑化）** | 建立去重后的 object/state 表，同时生成恢复原 root 和 edge occurrence 的映射；不是删除采样边 |
| **legacy compaction** | 原路径：C++ sampler 返回完整 occurrence 后，Python 再执行 mapper、`unique`、时序去重、顺序恢复和 Block 构造；evaluation 固定使用该路径 |
| **P0** | 实验过程中“最高优先级优化”的内部代号，不是论文算法名，也不是概率。正文技术名称是“采样器内分层紧凑化”：采样结束后先在线程局部去重，再按固定顺序合并 |
| **stable merge** | 保留 root-first、线程顺序和线程内首次出现顺序的合并，保证 Block 的 destination-prefix 约束 |
| **DGL Block** | GNN 一层消息传递使用的二部计算图；左侧 `src state` 包含 root 与采样邻居，右侧 `dst state` 是去重后的 root state，并且是 src 表的前缀 |
| **COO** | 按原采样边顺序保存的 `row/col` 两个等长数组；每个位置对应一个 edge occurrence |
| **CSC / segment plan** | 把 COO occurrence 按 destination 分组后的读取计划，当前显式表示为 `dst_ptr` 和 `edge_ids`；它只重排访问顺序，不去重 |
| **segment** | 同一个 destination state 的全部入边 occurrence；Fused Segment 对每个 `(destination, attention head)` 处理一个 segment |
| `edge_ids` | CSC position 到原 COO occurrence index 的排列；更直观的名字是 `csc_to_occurrence`。DGL API 称它为 `edge_ids`，但它不是 graph EID |
| **materialize（物化）** | 根据 mapping 真正分配并生成展开后的 tensor，例如把 edge-object feature 展开成每个 occurrence 一行 |
| **fused kernel（融合算子）** | 把原来连续启动的多个 GPU 算子合成一次 kernel 调用，减少中间 tensor 和 launch 开销 |
| `Q/K/V` | attention 的 query/key/value；最终版本中 Q 按 destination state 计算，K/V 按 edge occurrence 计算 |
| **direct builder** | 直接用 P0 的 COO 创建 DGL Block，并把 attention 后续使用的 CSC plan 显式挂到该 Block 上 |
| **raw memory update / smooth / Filter / feature attach** | GRU 先产生原始节点 memory 更新；原 MemShare 随后执行历史预测、平滑混合、Filter 状态更新，再结合 node feature 形成 attention 使用的时序状态表征。最终方案只优化第一步，不删除后三步 |
| DGL / Triton / cuBLAS | 分别承担图结构与通用图算子、自定义融合 GPU kernel、原生高效矩阵乘法 |
| **fallback / fail-fast** | fallback 是条件不满足时自动走兼容实现；fail-fast 是检测到不支持配置后立即报错，二者不是同义词 |
| **all-off / all-on** | 当前统一源码中最终三项全部关闭 / 全部开启；all-off 不是原始 master，all-on 也不包含已放弃原型 |

最容易混淆的关系是：

- 节点对象相同，只说明 UID 相同，不代表 timestamp 相同；
- 图边对象相同，只说明使用了同一条原图边，不代表多次采样边实例可以合成一次；
- `graph EID`、`edge-object slot` 和 `edge_ids` 分别回答“原图中是哪条边”“紧凑表的哪一行”“CSC 当前要读取原 COO 的第几个 occurrence”，属于三个不同层次的编号。

例如，原 COO 中有三次 edge occurrence：

| 原 COO occurrence index | graph EID | edge-object slot |
| ---: | ---: | ---: |
| 0 | 7 | 0 |
| 1 | 9 | 1 |
| 2 | 7 | 0 |

edge-object 表只有 `[EID 7, EID 9]` 两行，所以 occurrence 0 和 2 都指向 slot 0。假设 CSC 按 destination 分组后需要依次读取原 occurrence `2、0、1`，那么：

```text
edge_ids = [2, 0, 1]
```

因此 `edge_ids` 保存的是“重排后应读取哪个原 occurrence”，不是 `[7,9,7]` 这样的 graph EID，也不是 `[0,1,0]` 这样的 edge-object slot。

### 2.5 三层数据的数学定义

一个 batch 中存在三种不同的计算粒度。

#### 2.5.1 节点对象与图边对象

同一个节点 UID 的静态特征、当前 owner memory 和 mailbox 在 remote fetch 时只按 object 表获取一份；后续 consumer 是否展开由计算粒度决定。记有序节点 object 表为：

$$
\mathcal O_n = (u_0,u_1,\ldots,u_{U-1}).
$$

不可变 edge feature 也按有序 edge-object 表获取：

$$
\mathcal O_e = (e_0,e_1,\ldots,e_{O-1}).
$$

#### 2.5.2 时序节点状态

相同 UID 在不同 timestamp 下具有不同时间编码和 memory update 结果，因此必须保留为不同 state：

$$
\mathcal S = ((u_s,t_s))_{s=0}^{S-1}.
$$

正文统一把 state 到 UID object 的映射写成 `state_to_uid`，公式记作：

$$
m(s)=\operatorname{state\_to\_uid}[s].
$$

#### 2.5.3 采样边实例

即使两个 occurrence 指向相同 UID 或 edge object，它们仍可能具有不同 destination、时间差、softmax segment 和 dropout mask，因此不能合并：

$$
\mathcal E_b = ((s_{\mathrm{src},k},s_{\mathrm{dst},k},o_k,\Delta t_k))_{k=0}^{E-1}.
$$

其中 `o_k=edge_to_object[k]` 是 edge-object 表的行号，不是 graph EID，也不是 occurrence index。`\mathcal E_b` 会保留原顺序以及每条边重复出现的次数，不是会自动删除重复项的数学集合。

最终方案遵循的边界是：

> object 级原始数据只获取到一份 base table；下游仍按需要把 node feature 展开到 temporal state、把 edge feature 展开到 edge occurrence。仅对已验证有效的 GRU mailbox/hidden 分量执行 object 级投影；temporal-state 计算和 attention K/V、softmax、边聚合仍按各自所需粒度执行。

### 2.6 Temporal state 本身的贡献与边界

原 MemShare `master` 已经在 Python `to_block` 中对 `(UID,timestamp)` 执行 unique，并恢复 first-occurrence 顺序。因此不能把“提出 temporal state”或“首次按 `(UID,timestamp)` 去重”写成本文的新贡献。

最终方案中，temporal state 单独承担四个作用：

1. **语义边界**：只合并完全相同的 `(UID,timestamp)`；同一 UID 的不同 timestamp 始终保留，避免错误复用时间编码和动态 memory；
2. **统一中间表示**：P0 直接输出 state 表、`state_to_uid`、`root_mapping` 和 occurrence-to-state COO，builder、memory updater 与 attention 使用同一套编号，不再各自重建映射；
3. **稳定 Block 接口**：root state 按首次出现顺序占据 source-state 表前缀，重复 root 可以共享 state 计算，同时仍通过 `root_mapping` 保留各自的训练样本身份；
4. **复用边界暴露**：UID 不变的输入可在 object 级复用，随 timestamp 变化的计算保留在 state 级，依赖 destination、时间差或 dropout 的计算保留在 occurrence 级。

其中真正新增的执行方法是 P0 的 thread-local hash compaction 与 root-first stable merge：它把原本采样后的全局 `unique/scatter/remap` 后处理改成 sampler-native 的分层输出。它不会进一步减少 legacy 已有的 state 数、Block 边数或远程 fetch 对象数。

目前没有“关闭 temporal-state 去重”的独立端到端 A/B，因为那会改变原模型的 non-redundant Block 表示。WikiTalk 在 Fused Segment 与 Compact GRU 两侧共同开启时，P0 增量 ABBA 约下降 4.01%，但其中同时包含 UID object、edge object、temporal state、稳定合并和映射物化，不能把该数值单独归因于 temporal state。WIKI 约 8.75 个 temporal state 对应一个 UID object，只说明层次复用空间，不是相对 legacy 的新增压缩收益。

## 3. 总体执行链

```text
固定图划分与原 MemShare sampler
                 │
                 ▼
每个采样线程的 TemporalGraphBlock（内含 occurrence 列表）
                 │
                 ▼
P0：thread-local state/edge compaction + stable merge
                 │
     ┌───────────┴────────────────────┐
     ▼                                ▼
UID/EID object 表                temporal state + COO occurrence
     │                                │
     ▼                                ▼
remote feature/memory fetch    direct Block + attached CSC plan
     │                                │
     ├── UID memory/mailbox ──► Compact GRU raw update ─► 原 smooth/Filter ─► 时序状态表征 h
     └── edge feature ───────────────────────────┐
                                                ▼
                       destination Q + occurrence K/V materialization
                                                │
                         attached CSC plan ──────┤
                                                ▼
                                     Fused Segment Attention
                                                │
                                                ▼
                                       原训练损失与优化器
```

P0 只替换训练 sampler 的 compaction。evaluation sampler 使用 legacy compaction；Fused Segment 和 Compact GRU 在训练、验证和测试中均保持开启。

## 4. P0：采样器内分层紧凑化

P0 的核心不是改变 sampler，而是把 sampler 之后原本由 Python 重做的一轮全局去重，前移到更靠近 C++ sampler 输出的位置。

```text
旧路径：各线程采样结果
          ↓ 全部拼接
        Python 对整批 UID、EID、(UID,timestamp) 再做 unique/remap
          ↓
        构造 Block

P0：    各线程采样结果
          ↓ 各线程独立建立局部去重表
        按 root-first、线程顺序稳定合并
          ↓
        直接输出紧凑表 + 恢复原 batch 的映射 + COO
          ↓
        构造 Block
```

用“输入—处理—输出—保持不变”概括：

- **输入**：每个采样线程已经生成的 occurrence 列表；
- **处理**：各线程先独立给重复节点状态和 EID 分配局部表行号，再把各线程的小表按固定顺序合并；
- **输出**：节点对象表、图边对象表、`(UID,timestamp)` state 表，以及 root/edge occurrence 指向这些表的映射；
- **保持不变**：邻居采样结果、每条边重复出现的次数、原 occurrence 顺序、负样本和训练目标。

因此 P0 节省的是“采样完成后的第二次全局 unique、索引恢复和中间 tensor 物化”，不是通过少采样邻居获得加速。它仍会扫描线程局部结果并做一次稳定合并，所以不能称为零扫描或 sampling emission 时在线去重。

### 4.1 原链路的重复工作

legacy 路径先把 C++ sampler 的 occurrence 返回到 Python，再执行：

1. UID mapper 与 UID unique；
2. EID mapper 与 EID unique；
3. `(UID,timestamp)` temporal unique；
4. first-occurrence 顺序恢复；
5. metadata remap；
6. COO row/col 恢复；
7. DGL Block 构造。

这些步骤会再次扫描 sampler 已经生成的全部 occurrence，并产生多组临时 inverse、mask 和中间表。

### 4.2 P0 算法

P0 在 `neighbor_sample_from_nodes` 完成后保留每个 sampler thread 的 `TemporalGraphBlock`，随后执行：

1. 每个线程独立建立 `(UID,timestamp) -> local_state` hash 表；
2. 每个线程独立建立 `EID -> local_edge` hash 表；
3. 记录 occurrence 到线程局部 state/edge slot 的映射；
4. 全局 state 表先插入 root state，再按 thread ID 和线程内首次出现顺序稳定合并；
5. UID 与 EID 根据 distributed mapper 按 owner 分桶；
6. 生成 local slot 到全局 object/state slot 的映射；
7. 按原 occurrence 顺序物化 `row/col/edge_mapping/delta_ts`。

线程局部 hash 阶段无全局锁。稳定合并仍有一次全局遍历，因此 P0 应描述为“消除 post-sampling 全局 unique/remap”，不能描述为“零扫描”或“采样 emission 时在线去重”。

输入 `seed`、`seed_ts`、`nid_mapper` 和 `eid_mapper` 必须是 contiguous CPU `int64` Tensor；`seed` 与 `seed_ts` 行数相同，`num_parts` 为正。内部 slot 使用 `int32`，因此单 batch 的 state/object 数不能达到 `INT32_MAX`。

### 4.3 输出契约

| Tensor | Shape | 含义 |
| --- | ---: | --- |
| `dist_nodes` | `[U]` | 按 owner 分组的 distributed node object 表 |
| `dist_edges` | `[O]` | 按 owner 分组的 distributed edge object 表 |
| `state_to_node` | `[S]` | temporal state 到 node object 的映射 |
| `state_timestamps` | `[S]` | 每个 temporal state 的 timestamp |
| `root_mapping` | `[R]` | root occurrence 到 temporal state 的映射 |
| `row` | `[E]` | edge occurrence 的 source state |
| `col` | `[E]` | edge occurrence 的 destination state |
| `edge_mapping` | `[E]` | occurrence 到 edge object 的映射 |
| `delta_ts` | `[E]` | occurrence 级时间差 |
| `sizes` | `[2]` | `num_src_state, num_dst_state` |

以上十个输出最初均为 CPU `int64` Tensor；前九个数据与映射 Tensor 随后由 Python 搬到目标 GPU，`sizes` 留在 CPU 并作为 host 端尺寸使用。

root state 必须是 source state 表的前缀：

$$
\mathcal S_{\mathrm{dst}}=\mathcal S[0:D].
$$

这是 DGL Block 的结构约束，也是 P0 使用 root-first stable merge 的原因。

### 4.4 Metadata 映射

只有以下 root-occurrence 索引通过 `root_mapping` 转成 state index：

- `src_pos_index`；
- `dst_pos_index`；
- `src_neg_index`；
- `dst_neg_index`。

路由、统计和其他 metadata 不做盲目 remap，避免把非 root index Tensor 误当成 Block state index。

### 4.5 保持的语义

P0 保持：

- 相同的 root 与负样本；
- 相同的采样随机数和边 occurrence；
- 相同的 occurrence 顺序，以及每条边原有的重复出现次数；
- 相同的 UID、timestamp、EID 和 `delta_ts`；
- 相同的 root 到 state 关系。

owner object 表内部顺序可以与 sorted `torch.unique` 不同，但 inverse mapping 必须恢复完全相同的 object 值。不同 timestamp 的同一 UID 始终是不同 temporal state。

## 5. 直接 Block 构造与分段索引

这一阶段不再去重，也不减少边数。它接收 P0（训练）或 legacy compaction（evaluation）已经生成的同语义 COO `row/col`，完成两件事：

1. 更直接地创建同一张 DGL Block；
2. 提前取得“每个 destination 应处理哪些 occurrence”的分段索引，并把它作为融合 kernel 的显式输入挂到 Block 上。

### 5.1 与原始版本的区别

| 阶段 | 原始版本 | 最终版本 | 收益属于谁 |
| --- | --- | --- | --- |
| 生成紧凑表与 COO | C++ sampler 返回 occurrence，Python 再做 UID/EID unique、时序去重、映射和 `row/col` 恢复 | 训练时由 P0 生成；evaluation 仍走 legacy compaction | P0，不属于 builder |
| 创建 Block | 调用高层 `dgl.create_block(row,col)`，由 DGL 管理内部稀疏格式 | 调用底层 COO unit-graph constructor 创建同语义 DGL Block | direct builder |
| destination 分组 | 分组结构由 DGL 内部格式管理，后续由通用图算子读取 | 构造后取得 `dst_ptr/edge_ids`，并显式挂到 Block 上 | builder 与 fused attention 的接口 |
| attention | DGL 依次执行 `edge_softmax → dropout → V×attention → update_all` | 融合 kernel 直接按显式分段计划完成这条算子链 | Fused Segment Attention |

原始版本和最终版本使用相同的 source/destination state、edge occurrence 和边数；变化的是 Block 的构造入口以及 destination 分组信息何时、以什么形式交给 attention。

### 5.2 一个具体例子

假设 P0 或 legacy compaction 最终都得到相同的 COO：

```text
occurrence index:  0  1  2  3
source state row: [2, 3, 4, 2]
destination col:  [0, 1, 0, 1]
```

即 occurrence 0、2 发往 destination 0，occurrence 1、3 发往 destination 1。Direct builder 从同一组 `row/col` 创建 Block，并取得：

```text
dst_ptr  = [0, 2, 4]
edge_ids = [0, 2, 1, 3]
```

读取方式是：

- destination 0 对应 `edge_ids[0:2] = [0,2]`；
- destination 1 对应 `edge_ids[2:4] = [1,3]`。

`edge_ids` 只是把 occurrence 按 destination 排成便于连续处理的访问顺序，没有删除或合并任何边。

### 5.3 当前实现

输入 `row/col` 必须具有相同的 device、dtype 和 shape。builder 使用 DGL 的 direct unit-graph constructor 创建支持 COO/CSC 的 Block：

```text
P0 或 legacy row/col
    ↓
DGL direct COO Block
    ↓
adj_tensors("csc") 只生成/取得一次
    ↓
block._memshare_segment_plan = (dst_ptr, edge_ids)
```

其中：

- `dst_ptr` 描述每个 destination 的 occurrence segment；
- `edge_ids` 是 CSC position 到原始 COO occurrence index 的排列；
- `block._memshare_edge_src` 保留原始 occurrence source state；
- graph EID 可以被多个 occurrence 复用；node feature 由 node object 展开到 temporal state，edge feature 由 edge object 展开到 edge occurrence。

这里仍然需要 COO，原因是 sampler/P0 天然按采样 occurrence 顺序输出每条边的 `(source state,destination state)`，K/V、dropout mask 和梯度也都按这个原始 occurrence 顺序存放。若完全跳过 COO，sampler 就必须直接完成 destination 分组，同时额外输出 CSC source 与“CSC position → 原 occurrence”的排列；这只是把当前转换继续前移，并不会让映射本身消失。

这里没有新增跨 batch 的 cache 子系统。每个 Block 只是持有三项与结构有关的结果：

1. DGL Block 内部已经生成的 CSC 稀疏格式；
2. `block._memshare_segment_plan=(dst_ptr,edge_ids)`；
3. `block._memshare_edge_src=row`，即原 occurrence 顺序的 source state。

这些张量只在当前 batch 的当前 Block 生命周期内有效，不跨 batch，也不保存 Q/K/V、attention weight 或节点特征。DGL 原本就会管理已生成的 COO/CSC 格式；这里额外做的只是把 fused kernel 正好需要的 `dst_ptr/edge_ids` 暴露成稳定接口。`layers.py` 先用 `_memshare_edge_src` 构造 occurrence 级 K/V，再把 `dst_ptr/edge_ids` 传给 fused segment attention；forward 和 backward kernel 都用 `dst_ptr` 找到每个 destination 的边范围，并用 `edge_ids` 回到原 occurrence 顺序读取 K/V、dropout mask及写回梯度。

### 5.4 收益归因边界

- direct builder 本身不做 P0 的 object/state 去重；
- direct builder 也不执行 attention，只提供 Block 与分段索引；
- 它没有消除 COO→CSC 的转换，只是把转换前移到 builder，并把结果显式交给 fused kernel；也不能声称原 DGL 会为每个算子重复生成 CSC，因为 DGL 自身会管理和缓存已生成的稀疏格式；
- 当前开启 Fused Segment Attention 时会同时启用 direct builder，因此该开关的 OFF/ON 结果实际包含“direct builder + 显式 segment plan + fused attention”，不能全部归因于纯 attention kernel；
- direct builder 尚无单独的端到端收益结论，最终论文应把它作为融合 attention 的执行接口，而不是独立算法贡献。

## 6. 分段融合注意力（Fused Segment Attention）

### 6.1 最终计算边界

最终版本仍然物化 occurrence 级 K/V：

- `e`：一个 edge occurrence；
- `src(e)`、`dst(e)`、`obj(e)`：该 occurrence 对应的 source state、destination state 和 edge object；
- `h_s`：temporal state `s` 的输入表征；
- `f_o`：edge object `o` 的不可变特征；
- `φ(Δt)`：时间差编码；
- `∥`：特征拼接。

$$
X_e=[h_{\mathrm{src}(e)}\Vert f_{\mathrm{obj}(e)}\Vert\phi(\Delta t_e)],
$$

$$
K_e=W_KX_e+b_K,\qquad V_e=W_VX_e+b_V.
$$

对 destination state `d`，Q 只计算一次：

$$
Q_d=W_Q[h_d\Vert\phi(0)]+b_Q.
$$

`X_e` 只是数学记号；当前代码为 K 和 V 分别执行一次 occurrence gather、`cat` 和 Linear，没有共享同一份 `X_e` buffer。Projected K/V 不属于最终版本。segment kernel 的输入是已经物化的 Q/K/V、`dst_ptr` 和 `edge_ids`。

### 6.2 Forward

每个 `(destination,head)` 由一个 Triton program 处理。kernel 以 32 个 edge 为一块遍历长 segment，并执行 online softmax。这里 `𝓔(d)` 表示 destination 为 `d` 的全部 occurrence，`m_d` 是该 segment 的最大 score，`M_e` 是取值为 0 或 1 的 dropout mask，`p_drop` 是 dropout 概率，`O_d` 是最终聚合结果：

$$
z_e=\operatorname{LeakyReLU}(Q_d^TK_e),
$$

$$
p_e=\frac{\exp(z_e-m_d)}{\sum_{j\in\mathcal E(d)}\exp(z_j-m_d)},
$$

$$
\widetilde p_e=\frac{M_e}{1-p_{\mathrm{drop}}}p_e,
$$

$$
O_d=\sum_{e\in\mathcal E(d)}\widetilde p_eV_e.
$$

这里：

- LeakyReLU slope 为 `0.2`；
- 不使用按 head dimension 开平方倒数的 attention scaling；
- dropout 位于 softmax 之后；
- dropout 后不重新归一化；
- `edge_ids` 用于从 CSC segment position 间接读取原始 occurrence K/V。

Forward 额外保存每个 destination/head 的 log-sum-exp：

$$
L_d=m_d+\log\sum_{e\in\mathcal E(d)}\exp(z_e-m_d).
$$

Autograd 仍保留 backward 所需的 Q/K/V、`dst_ptr`、dropout mask 和 `edge_ids`，但不保存完整 attention probability 或 `V * attention` 中间张量。

### 6.3 Backward

Backward 使用保存的 `L_d` 重算 softmax：

1. 第一遍计算 softmax Jacobian 所需的 segment center；
2. 第二遍生成 `grad_Q`、`grad_K` 和 `grad_V`；
3. 每个原始 COO occurrence index 只对应一个 CSC position，因此每个 K/V gradient 都写入唯一位置，不需要原子加法。

如果未来允许 `edge_ids` 中出现重复 occurrence index，多个线程就可能同时写同一位置，此时必须改成原子累加；当前 DGL/builder 保证 `edge_ids` 是 occurrence index 的排列。多个 occurrence 共享同一个 graph EID 不影响该性质。

### 6.4 Dropout RNG

mask 由：

```text
aten.native_dropout(empty([E, H]), p, training=True)
```

生成，Triton 只消费 bool mask。这里 `E` 是 edge occurrence 数，`H` 是 attention head 数。该做法使 mask 和 CUDA 随机数状态的消耗方式与 PyTorch Dropout 对齐。Fused reduction 与原 DGL kernel 的浮点累加顺序仍不同，因此不能承诺完整训练轨迹逐 bit 相同。

## 7. 按 UID 复用的 Compact GRU

### 7.1 可复用与不可复用部分

对于 temporal state `s=(u,t)`：

- mailbox 和节点动态 memory 是 UID object 级输入；
- time feature 是 temporal-state 级输入；
- GRU raw update 仍是 temporal-state 级结果。

因此不能合并不同 timestamp 的完整 GRU，只能复用其中与 UID 相同的 affine projection。Compact GRU 只替换原 `raw_memory_update` 内部的 GRU 执行；后续默认的 `smooth_prediction`、`smooth_blend`、`smooth_filter_update` 和 `feature_attach` 均保持原样。

### 7.2 Object/state 分解

本节符号约定如下：

- `u`：节点对象行号；`s`：temporal state 行号；`m(s)`：state `s` 对应的节点对象行号；
- `M_u`：节点对象 `u` 的 mailbox 向量；
- `H̄_u`：节点对象 `u` 更新前的动态 memory；
- `t_u^mem`：该动态 memory 的时间戳；
- `φ`：时间编码器；
- `G^mail`、`G^time`、`G^hidden`：分别由 mailbox、时间编码和旧动态 memory 得到的三门预激活量；
- `W_ih^mail` 和 `W_ih^time`：原 GRUCell 输入权重中对应 mailbox 与时间特征的列切片，不是新参数。

保留三个原生 cuBLAS GEMM：

$$
G_u^{mail}=W_{ih}^{mail}M_u,
$$

$$
G_s^{time}=W_{ih}^{time}\phi(t_s-t^{mem}_{m(s)})+b_{ih},
$$

$$
G_u^{hidden}=W_{hh}\overline H_u+b_{hh}.
$$

对 state `s`，Triton kernel 通过 `state_to_uid` 读取 object gate。为避免与 attention 的 state 表征 `h_s` 混淆，这里把 GRU 的 input-side gate 向量记为 `I_s`，hidden-side gate 向量记为 `J_s`：

$$
I_s=G_{m(s)}^{mail}+G_s^{time},
$$

$$
J_s=G_{m(s)}^{hidden}.
$$

按照 PyTorch GRUCell 的 `r/z/n` gate 顺序：

$$
r_s=\sigma(I_{r,s}+J_{r,s}),
$$

$$
z_s=\sigma(I_{z,s}+J_{z,s}),
$$

$$
n_s=\tanh(I_{n,s}+r_sJ_{n,s}),
$$

$$
H'_s=n_s+z_s(H_s-n_s).
$$

其中上一时刻 memory 满足：

$$
H_s=\overline H_{m(s)}.
$$

`H_s` 是 state `s` 的旧动态 memory，`H'_s` 只是 raw GRUCell 更新结果 `updated_memory0`，还不是最终送入 attention 的表征。它随后仍经过原 MemShare 的 smooth prediction、smooth blend、Filter update 和 feature attach。两个 bias 保持分离，尤其 hidden candidate bias 仍位于 reset gate 乘法内部，不改变 GRUCell 参数化。

### 7.3 Backward

Backward：

- `grad_time_gate` 按 temporal state 直接写回；
- `grad_mail_gate` 和 `grad_hidden_gate` 按 `state_to_uid` atomic accumulate 回 UID object；
- 生产路径 fetched `state_memory` 不求梯度，因此不分配无效的 state-memory gradient；
- 显式要求 `state_memory.requires_grad=True` 的调用回退原生 GRUCell。

原 `GRUCell` Parameter、DDP hook、state dict 和 checkpoint 名称均保持不变。

UID gradient 使用 atomic accumulation，fused pointwise 的浮点计算顺序也与原 GRUCell 不同；Compact GRU 只保证容差内数值与梯度对齐，不保证逐 bit 相同。

### 7.4 对其他 memory updater 的支持边界

最终版本中的 Compact GRU 仍只支持 `memory_update='gru'`、`dim_time>0` 的 time-aware `torch.nn.GRUCell`。构造期遇到 RNN、Transformer 或无时间编码配置会直接报错；只有在已经属于受支持 GRU、但 dtype/device/shape 不满足 Triton kernel 条件时，才回退原生 GRUCell/ATen 路径。当前端到端结果只验证了 `TGN_large + historical`。

实验开关 `compact_memory_ops` 另外提供了 Compact RNN 和 Compact Transformer consumer。它们复用同一份 object/state 映射：RNN 保留原生 PyTorch 路径，Transformer 在受支持的 CUDA FP32 路径上把 mailbox slot 视为固定长度 segment，复用现有 Fused Segment Attention kernel。它们都不复用 GRU 三门 kernel，也不属于第 12 节 final all-on 结果。

| memory updater | 当前实现 | 边界 |
| --- | --- | --- |
| TGN-style GRU | 支持 | mailbox/hidden 是 UID object 输入，time feature 是 temporal-state 输入 |
| JODIE-style RNNCell | 实验支持 | mailbox/hidden affine 在 UID object 上计算，time affine 与 tanh/relu 保持 state 级；可消费 legacy Block 的 `state_to_uid`，但 JODIE 的 P0 identity/reversed 图链仍不支持 |
| APAN/Transformer memory | 实验支持 | Q 与 K/V 的 mailbox 分量在 UID object 上计算；TimeEncode 保持 state 级，mailbox-slot attention 复用 fixed-degree Segment Attention kernel，residual、LayerNorm 和 MLP 保持 state 级；APAN 的 P0 no-neg/reversed 图链仍不支持 |
| DistTGL Smart updater | 不支持 | 使用另一套模型与 updater 入口，需要独立设计和验证 |

因此可泛化的是 object/state/occurrence 分层表示，不是 `Compact GRU` 这一个 kernel。Compact RNN/Transformer 只在显式开启 `compact_memory_ops` 且 object 属性完整时执行，否则回退原 updater；它们是否支持 P0/Fused Segment 仍要分别满足第 9 节的整体边界。

### 7.5 Persistent state、smooth/Filter 与 feature attach 如何使用分层信息

Compact GRU 只优化 raw memory update。后续阶段已经在去重后的 temporal-state 表上运行，因此不会重复计算完全相同的 `(UID,timestamp)` 行；实验开关 `compact_memory_ops` 进一步复用同一 UID 的不同 timestamp state 之间仍相同的只读分量。

| 阶段 | 正确粒度 | 当前是否直接复用 `state_to_uid` |
| --- | --- | --- |
| raw GRU | UID object projection + temporal-state output | 是；Compact GRU 的核心路径 |
| raw RNN/Transformer | UID-invariant affine + temporal-state output | 实验支持；见第 7.4 节 |
| smooth prediction | Filter increment 按 shared UID 索引，但预测值展开到 temporal state | 实验支持；只对本 Block 使用的 shared UID 计算 |
| smooth blend | temporal state | 否；raw update 随 timestamp 变化，不能按 UID 合并 |
| Filter update | temporal-state change 写回 shared UID 状态 | 只复用只读索引；change、顺序和重数仍按 state 保留 |
| state snapshot | temporal state | 否；保留每个 state 的 memory、UID 和 timestamp |
| persistent memory/mailbox write | 从事件端点 state 中按 UID 选择最新 timestamp，再写 owner object 行 | 实验支持 dense object selector；跨 rank reduction 不变 |
| feature attach | `updated_memory[state] + node_feature_projection[state]` | 实验支持 object-first Linear，再按 `state_to_uid` gather |

这些兼容版均默认关闭、尚无端到端收益证据，因此未进入最终版本。persistent mail 仍依赖对端节点和 edge feature；Filter change、smooth blend、state snapshot 和最终 state 表征也不能按 UID 合并。

### 7.6 其余 memory 阶段的 compact-native 兼容版

为避免新的 object/state 表只能被 Compact GRU 消费，代码已增加实验开关：

```text
--compact_memory_ops on
```

该开关默认关闭，尚未进入第 12 节的 final all-on 结果。raw RNN/Transformer 只需要 legacy Block 已有的 object 表和 `state_to_uid`，不依赖 P0；smooth/Filter、persistent write 与 object-first feature attach 仍要求 P0 hierarchical marker。自定义 producer 缺少 object 属性或无 memory 时自动保留原 state-level 路径。

| 阶段 | compact-native 处理 | 必须保留的 state-level 语义 |
| --- | --- | --- |
| raw RNN | mailbox 与 hidden affine 在 UID object 表执行一次，再 gather 并加 state time affine | tanh/relu 输出仍逐 temporal state；小 batch 可能因额外 GEMM/gather 更慢 |
| raw Transformer | Q 及 K/V mailbox 分量在 UID object 表执行一次；mailbox slot 作为固定长度 segment 复用现有 attention kernel | time K/V、residual/LN/MLP 仍逐 state；不支持的 device/dtype/shape 回退原 PyTorch attention |
| smooth prediction | 只对本 Block 实际使用的 shared UID object 读取 historical memory 和 Filter increment、执行预测与归一化，再按 `state_to_uid` 展开 | raw update、smooth blend 仍随 timestamp/state 变化 |
| Filter update | 复用 object→state 展开计划得到原 state 顺序的 `shared_ind` | `change` 和重复索引写入的顺序、重数完全保留，不改成 UID sum/mean |
| persistent memory write | 用 dense object slot 按 memory timestamp 选择每个 UID 的最新 state，删除 distributed-ID `unique` | 跨 rank/shared receive 端的 latest reduction 仍保留 |
| persistent mailbox write | 用同一 object slot 表、但按 mailbox timestamp 独立选择 winner | mail 仍依赖对端 state 与 edge feature；`deliver_to=neighbors` 和 score reduction 回退 legacy |
| feature attach | 静态 node feature 的 Linear 在 UID object 表执行一次，再 gather 到 state 与最终 memory 相加 | 最终输出仍是一行一个 temporal state；缺 object 属性时回退原实现 |

memory 与 mailbox 使用两次 dense selector，而不是强行共用 winner。这样在 `memory_ts` 与 `mailbox_ts` dtype 不同、时间戳 cast 后产生 tie 时，仍分别保持原路径的选择语义。object 表必须保持 P0 的 owner-grouped 顺序，才能直接进入原 all-to-all route；该约束由 P0 marker 和结构测试保证。

RNNCell 没有被改写成 GRU kernel。共用 Triton pointwise 原型在 `S/U=3000/2200` 上受 launch/Autograd 延迟限制，比原 RNNCell 更慢，因此已回退；保留的原生 Compact RNN 在该形状回退约 `52.1%`，在 `9000/1500` 时下降约 `18.9%`。Transformer 不改写 Q/K/V GEMM，只复用 fixed-degree Segment Attention；A40 FP32、mailbox size 10 的交错测量中，完整 forward+backward 在 `S/U=3000/2200` 下从 `2.8522 ms` 降至 `2.3254 ms`（`-18.47%`），在 `9000/1500` 下从 `7.7189 ms` 降至 `4.4984 ms`（`-41.72%`）。两者仍只作为默认关闭的实验 consumer，待端到端 A/B 后再决定是否进入 final all-on。

## 8. 每个训练 batch 的执行顺序

1. 原 MemShare sampler 生成 temporal neighbor occurrence；
2. P0 在线程局部结果上完成 state/edge compaction 和稳定合并；
3. P0 输出 UID/EID object 表与 COO occurrence；
4. direct builder 创建 Block，并把 CSC segment plan 显式挂到 Block 上；
5. 按 owner 对 object 表执行原 remote fetch；
6. `prepare_input` 将 node feature 展开到 temporal state、将 edge feature 展开到 edge occurrence，并保留 object-level memory/mailbox 与 state mapping；
7. Compact GRU 在 UID object 上复用 projection，并生成 temporal-state 级 raw memory update；
8. 原 MemShare 继续执行 smooth prediction、smooth blend、Filter update 和 feature attach，得到时序状态表征；
9. K/V 继续按 occurrence 物化；
10. Fused Segment Attention 直接输出 destination representation；
11. 执行原 decoder、loss、backward、optimizer 和 memory/mailbox 更新。

P0 不减少网络请求对象，也不修改通信协议。shared count 的小幅变化与非 bitwise 训练轨迹经过 HistoricalCache 的 cosine/alpha refresh 判定后被放大这一解释一致，但现有日志不能证明唯一因果；不能把它表述为 P0 的通信优化。

### 8.1 为什么不合成一个巨型 kernel

统一的是分层表示和消费原则，不是把 memory updater 与 attention 物理合成一个 kernel。Compact GRU 的 raw update 之后仍必须经过原 MemShare 的 smooth prediction、smooth blend、Filter 状态更新和 feature attach；Attention 只能读取这些步骤产生的最终时序状态表征。两侧还分别依赖 object/state GEMM、Q/K/V GEMM、不同的并行粒度和不同的 backward reduction。跨过这些边界强行融合会重写 cuBLAS GEMM 与有状态 Filter，增加寄存器压力和正确性风险。

当前保留的融合边界已经按数据依赖划分：

- Compact GRU 在 state-level Triton kernel 中融合 object gate gather、time gate 相加和 GRU pointwise，三个大矩阵乘法继续交给 cuBLAS；
- Fused Segment Attention 在 destination-segment Triton kernel 中融合 score、softmax、dropout 和 value aggregation，Q/K/V 矩阵乘法继续交给 cuBLAS；
- Compact Transformer 把每个 state 的 mailbox slot 视为 fixed-degree segment，直接复用同一 attention kernel，不再增加第三颗 attention kernel；
- 两者直接消费同一套 `state_to_uid`、state rows、occurrence COO 和 segment plan，不在中间重新执行 unique 或 remap。

因此论文上应把它们合并为一个“分层紧凑执行”方法下的两个 compact consumers，而不是实现一个跨 memory/attention 的 mega-kernel。

### 8.2 代码对应关系

| 组件 | 主要文件 |
| --- | --- |
| P0 C++ compaction | `csrc/sampler/include/sampler.h`、`csrc/sampler/export.cpp` |
| sampler 接线与训练/eval gate | `starrygl/sample/sample_core/neighbor_sampler.py`、`examples/train_boundery.py` |
| P0 consumer 与 Block metadata | `starrygl/sample/batch_data.py` |
| direct segment builder | `starrygl/sample/direct_segment_block.py` |
| Fused Segment | `starrygl/module/triton_segment_attention.py`、`starrygl/module/layers.py` |
| Compact GRU | `starrygl/module/compact_gru.py`、`starrygl/module/triton_compact_gru.py`、`starrygl/module/memorys.py` |
| Compact RNN/Transformer 与其余 compact memory stages（实验） | `starrygl/module/compact_gru.py`、`starrygl/module/compact_memory_ops.py`、`starrygl/module/memorys.py`、`starrygl/sample/data_loader.py`、`starrygl/sample/memory/shared_mailbox.py` |
| 正确性测试 | `tests/test_hierarchical_compaction.py`、`tests/test_direct_segment_block.py`、`tests/test_triton_segment_attention.py`、`tests/test_compact_gru.py`、`tests/test_compact_memory_ops.py` |

## 9. 支持范围与回退

本节中的特殊模式含义如下：

- `identity`：不执行常规 temporal neighbor sampling、只保留 root 的 sampler policy；
- `reversed`：identity 架构使用的特殊反向 Block 构造路径；最终 P0 训练路径不支持它；
- `combined attention`：node、edge、time 分别使用独立 Linear 后再相加的旧 attention 参数化；最终 Fused Segment 只覆盖非 combined 的拼接输入分支；
- `sampling layers=1`：只进行一层 temporal neighbor sampling。

| 项目 | 当前支持 | 不满足时 |
| --- | --- | --- |
| 端到端已验证模型 | 当前 `TGN_large` | 其他模型尚未验证且不会整体自动回退；应保持相关 flags all-off |
| P0 sampling layers | 1 | fail fast |
| P0 sampling | temporal、带负样本、非 identity、non-reversed 的训练采样 | fail fast |
| reversed evaluation | 不启用 P0 | legacy compaction |
| Attention | CUDA FP32、非 combined、node+edge+time 完整分支、K/V per-head dim 均不超过 64 | DGL attention |
| Compact GRU 配置 | time-aware GRU | 非 time-aware GRU 在启用开关时 fail fast |
| Compact GRU kernel | contiguous CUDA FP32、非空 state、hidden dim 不超过 128、state memory 不求梯度 | 原生 GRUCell/ATen fused cell |
| Compact RNN/Transformer（实验） | object memory/mailbox 属性与合法 `state_to_uid`；不要求 P0 | 缺 object 属性时原 updater fallback |
| 其余 compact memory stages（实验） | P0 hierarchical Block、node memory、historical self-delivery 主路径 | 缺 object 属性时 state-level fallback；neighbor/score persistent write固定走 legacy |
| evaluation compaction | legacy | 固定设计 |

所有新增开关默认关闭，以保证旧 checkpoint、其他模型和非目标 dtype 的兼容路径不变。

## 10. 最终配置

### 10.1 All-off baseline

这里的 baseline 是当前统一源码中三项全关，不是未经同步的原始 master 代码。

```text
--hierarchical_compaction off
--fused_segment_attention off
--compact_gru_update off
--compact_memory_ops off  # 新增实验兼容版，不属于第 12 节结果
--compact_kv_projection off
--triton_edge_projection off
--dual_dedup_view off
```

### 10.2 Final all-on

```text
--hierarchical_compaction on
--fused_segment_attention on
--compact_gru_update on
--compact_memory_ops off  # 待独立 A/B 后再决定是否并入 final all-on
--compact_kv_projection off
--triton_edge_projection off
--dual_dedup_view off
```

共同实验配置：

```text
model: TGN_large
sample_type: boundery_recent_decay
--shared_memory_ssim 0.3  # alpha：HistoricalCache 余弦距离刷新阈值
--probability 0.1         # beta：原 MemShare 通信采样的基础概率参数
--freeze_adaptive_parameters
environment: tgnn_3.10 + CUDA 11.8 + Triton 3.1.0
hardware: two nodes, 8 × A40, 10GbE
network: NCCL Socket, ens4f1np1, IB disabled
```

## 11. 正确性验证

已完成：

- 一次性执行的 50 组 recent randomized parity；
- 一次性执行的 30 组 boundary-decay randomized parity；
- P0 occurrence/state/root/edge mapping 对齐；
- 相同 UID、相同 timestamp 和不同 timestamp 边界；
- Fused Segment forward、Q/K/V backward；
- 空 segment、非均匀 degree、degree 630；
- 非 identity edge permutation；
- fixed dropout mask 下的 forward/backward parity；
- native-generated mask 与调用后 CUDA RNG state 对齐；
- Compact GRU output、time/GRU 参数梯度与 fallback；
- compact smooth/Filter 的 object→state 展开、未使用 object 排除、边界归一化与 gradient parity；
- persistent memory/mail 独立 timestamp selector 与 feature-attach parity；
- Compact RNN tanh/relu 与 Transformer forward、参数梯度、dropout RNG 消耗对齐；
- fixed-degree mailbox Segment Attention forward 与 Q/K/V backward parity；
- 非 hierarchical Block 的 raw object consumer 与缺属性时的 state-level fallback；
- 长 segment forward/backward stress；
- 双机源码 SHA256 一致性。

当前相关测试：

```text
36 passed
py_compile passed
git diff --check passed
```

## 12. 最终实验结果

最终结果必须使用 all-off 对 final all-on 的直接 A/B，不能把 P0、Fused Segment 和 Compact GRU 的边际百分比相加。

WIKI 使用 3 个 seed、每个 50 epoch；LASTFM 使用 seed 6773、50 epoch；WikiTalk 和 Flight 使用 seed 6773、各 10 epoch。

### 12.0 统计术语与口径

| 名词 | 本文中的固定含义 |
| --- | --- |
| **rank** | 一个分布式训练进程，当前通常对应一张 GPU |
| **rank mean** | 每个 epoch 先对 8 个 rank 的耗时求平均，再在指定 epoch 范围内求平均；表示平均负载，不是任务必须等待的最慢路径 |
| **critical** | 每个 epoch 先取 8 个 rank 的最大耗时，再在指定 epoch 范围内求平均；同步训练必须等待最慢 rank，因此它是本文的主效率指标 |
| **full / steady / tail** | full 使用全部 epoch，并包含 epoch 0 计时区间内的首次 kernel/JIT 编译、缓存预热等开销；不包含 epoch 计时开始前的进程、NCCL、模型和 dataloader 初始化。steady 使用 epoch 1 至末轮；tail 使用最后若干轮，WIKI/LASTFM 为最后 10 轮，WikiTalk/Flight 为最后 4 轮 |
| **train time** | 该 epoch 的训练阶段耗时，包括采样、数据搬运、forward、backward、参数更新、训练通信和同步，不包含 validation/test |
| **total wall** | 当前日志中的每轮总耗时：从该 epoch 开始到 train、validation、test、同步和本轮计时内簿记结束；计时在 checkpoint/model 文件写入前完成，因此不包含保存开销 |
| **A/B** | 相同 seed 和配置下的 OFF/ON 配对；耗时报告 ON 相对 OFF 的百分比变化，因此负值表示更快 |
| **ABBA** | `OFF → ON → ON → OFF` 的运行顺序，用于抵消缓存、机器状态和执行顺序漂移 |
| **paired mean ± std** | 先对相同 seed 的 OFF/ON 计算变化，再对不同 seed 的 paired 变化求均值和样本标准差；`±` 表示跨 seed 波动，不是 rank 或 epoch 波动 |
| **pp（百分点）** | AP/AUC 的绝对差。例如 `95.0% → 94.8%` 是 `-0.2 pp`，不是相对下降 `-0.2%` |
| **best-val checkpoint** | 只按 validation AP 选择 AP 最高的 epoch，再报告该 checkpoint 对应的 test AP/AUC；不按 test 指标挑 checkpoint |
| **TTT** | time-to-target：validation AP 第一次达到指定阈值时，从 epoch 0 累计到命中 epoch 的 critical 时间；未达到时应记为 N/A |
| `alpha` | HistoricalCache 的余弦距离刷新阈值；旧 CLI 名为 `shared_memory_ssim`，这里不是在报告图像 SSIM 指标 |
| `beta` | 原 MemShare 的通信采样基础参数；同时参与跨分区 root/event 保留、boundary-decay 远程邻居概率缩放和默认 local/global negative proposal 混合。单条远程邻居的最终概率不一定等于 `beta`；CLI 名为 `probability` |
| **shared count** | 参与 MemShare shared-node memory 更新的对象计数，不是 CPU/CUDA shared memory、通信字节或耗时 |
| **memory comm** | 日志中的动态 memory 通信计数，不是 GPU memory 容量、通信字节或耗时 |

除非另有说明，耗时变化为负表示更快，AP/AUC 的 pp 变化为负表示质量下降。

### 12.1 效率

负号表示耗时下降：

每个单元格依次给出 `train critical / total-wall critical`：

| 数据集 | full | steady | tail |
| --- | ---: | ---: | ---: |
| WIKI（3 seeds） | -2.29% ± 5.96% / -1.85% ± 5.51% | **-9.67% ± 2.72% / -7.07% ± 2.64%** | -9.33% ± 3.78% / -7.06% ± 3.96% |
| LASTFM | -11.48% / -9.30% | **-12.59% / -9.93%** | -11.86% / -8.92% |
| WikiTalk | -10.07% / -7.38% | **-10.97% / -7.93%** | -10.13% / -7.35% |
| Flight | -5.84% / -4.34% | **-7.97% / -5.69%** | -7.81% / -5.71% |

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

四个数据集的 steady train 和 total wall 均为正收益。WIKI 单个 steady epoch 约 0.15 秒，而逐 epoch 日志只保留到 0.01 秒；其 steady/critical 百分比存在较明显量化误差，高精度 full `avg_time` 的方向与主结论一致。

#### 12.1.1 P0 增量 ABBA（OFF→ON→ON→OFF 补充消融）

在 WikiTalk 单 seed、`--skip_eval`、每次 10 epoch、Fused Segment 与 Compact GRU 两侧共同开启且 projected K/V/Triton edge projection 关闭的补充 ABBA 中，开启 P0 后 steady critical 从 7.3328 秒变为 7.0389 秒，差异为 -4.01%；两组相邻配对分别为 -4.07% 和 -3.95%。steady shared count 同时下降约 0.96%，因此不能把全部差异归因于 P0 的 compaction 计算链。该消融不与最终 all-off/all-on 百分比相加。

### 12.2 AP/AUC

表内数值为 all-on 减 all-off；负值表示指标下降：

| 数据集 | best-val 对应 test AP / AUC | final test AP / AUC |
| --- | ---: | ---: |
| WIKI（3 seeds） | -0.0687 ± 0.2062 / -0.0883 ± 0.2469 pp | -0.1349 ± 0.2935 / -0.1262 ± 0.2976 pp |
| LASTFM | -0.1556 / -0.2869 pp | -0.1556 / -0.2869 pp |
| WikiTalk | -0.0246 / -0.0364 pp | -0.0246 / -0.0364 pp |
| Flight | +0.0008 / -0.0051 pp | +0.0008 / -0.0051 pp |

当前允许的质量表述是：

- WIKI 三 seed 的 best-val checkpoint 对应 test 平均回退小于 0.1 个百分点；
- 所有已测 run 的 best-val checkpoint 对应 test 最坏回退小于 0.5 个百分点；
- 不能表述为精度完全无损或统计非劣；
- LASTFM、WikiTalk 和 Flight 仍需补多 seed。

### 12.3 Time-to-target

每个数据集内 OFF/ON 共用下表的 validation AP 阈值；不同数据集使用不同阈值。TTT 从 epoch 0 起累计，train 列累计 epoch-critical train time，含评估列累计 epoch-critical total wall；负值表示时间缩短。这些阈值结果是描述性分析，不作为预注册的确认性收敛结论。

| 数据集 | 目标 AP | 首次命中 epoch（OFF → ON） | train TTT（OFF → ON） | 含评估 TTT（OFF → ON） |
| --- | ---: | ---: | ---: | ---: |
| WIKI（3 seeds 均值） | 95% | 见下文 | 7.60 → 7.85 s（+3.58% ± 8.72%） | 11.36 → 11.78 s（+3.96% ± 8.39%） |
| LASTFM | 90% | 35 → 37 | 46.33 → 43.37 s（-6.37%） | 78.60 → 75.18 s（-4.34%） |
| WikiTalk | 97% | 7 → 7 | 64.03 → 57.57 s（-10.09%） | 103.45 → 95.79 s（-7.42%） |
| Flight | 96% | 7 → 7 | 20.73 → 19.63 s（-5.40%） | 33.18 → 31.86 s（-3.98%） |

WIKI 三个 seed 的首次命中 epoch 分别为：6773 `37 → 40`、95842 `41 → 42`、12345 `36 → 38`。表中 WIKI 绝对 TTT 是三个 seed 累计时间的算术均值，百分比仍按 seed 配对后计算均值与样本标准差。

WIKI 三个 seed 均晚 1--3 个 epoch 越过 95%；单 epoch 加速只在部分 seed 抵消了额外 epoch，paired mean TTT 仍回退约 4%。最终贡献应强调训练吞吐与整轮 wall time，不声称四数据集都加速收敛。

### 12.4 工作量与网络

- 同一数据集/seed 每轮经 all-reduce 得到的全局 local/remote node/edge 计数在 OFF/ON 间 bit-exact，八个 rank 打印值一致；日志未单独记录归约前每个 rank 的计数；
- `alpha/beta` 全程固定为 `0.3/0.1`；
- `memory comm` 全程为 0；
- steady shared count：WIKI 三 seed paired percentage mean `+0.01%`、LASTFM `-1.40%`、WikiTalk `-0.31%`、Flight `-0.88%`；
- 所有日志均使用 `NET/Socket` 和 `ens4f1np1`，`NET/IB` 为 0，GPU Direct RDMA 关闭。

WIKI/LASTFM 每个 run 均有 400 条 rank-epoch 记录，WikiTalk/Flight 每个 run 均有 80 条；loss 全部有限，未出现运行或 NCCL 错误。

shared count 存在轻微轨迹漂移，因此不能把全部时间下降归因于计算去冗余；其变化仍明显小于 7.97%--12.59% 的 steady train 收益。

## 13. 已放弃方案

| 方案 | 局部结果 | 端到端结果 | 判定 |
| --- | ---: | ---: | --- |
| 分层投影 K/V（projected/compact K/V） | forward+backward 约 -35.3% | WikiTalk 八卡稳态约 +3.0% | 放弃 |
| Triton edge projection（只融合 edge feature 投影） | 局部投影明显更快 | WikiTalk 稳态约 +3.6% 至 +3.8% | 放弃 |
| K/V 公共输入复用（CSE，公共子表达式消除） | forward+backward 局部约 -8.1% | 稳态约 -0.9% | 低于门槛，回退 |
| temporal-first（先生成 temporal state，再做 UID object 去重） | 输入 unique 行数下降 | `to_block` 约 +3.23% | 放弃 |
| packed temporal key（把 UID/timestamp 编码为单个整数 key） | unique 子段约 -52.8% | steady 持平或回退 | 放弃 |
| Segment 8 warps（每个 Triton program 使用 8 个 warp） | kernel 约 -16.4% | steady 约 -0.32% | 低于门槛，回退 |
| recent 邻居前后缀复用 | 未进入长测 | Q/softmax/dropout 仍依赖 occurrence | 不推进 |

这些方案的实验日志与历史记录仍保留；只有部分原型代码仍在仓库中且默认关闭。它们均不属于最终执行链。

## 14. 论文贡献组织

不建议把原 temporal dedup、P0、Fused Segment 和 Compact GRU 写成四个零散优化。论文贡献列表只保留一个总技术点：

> 在继承 MemShare `(UID,timestamp)` non-redundant Block 语义的基础上，把原直接去重升级为跨阶段的 Batch 分层紧凑执行：训练 sampler 原生生成 object/state/occurrence 三粒度表示，evaluation/fallback 由 legacy compaction 生成兼容表示；Block builder、memory updater 与 segment attention 直接消费同一套映射，并在各自最低合法粒度融合执行，从而减少重复 compaction、映射、展开、中间张量和 kernel 编排。

方法说明可以分成两个阶段，但它们不是两项独立贡献：

1. **构造阶段**：稳定生成 UID object、temporal state、edge object 与 occurrence 映射；
2. **消费阶段**：UID-aware memory update 和 direct segment attention 在同一分层表示上执行紧凑计算。

不能宣称：

- 减少了远程采样概率或通信对象数量；
- sampler 直接输出完整 CSC；
- 最终 attention 不物化 occurrence K/V；
- 精度严格无损；
- WIKI 的 time-to-target 已改善；
- 其他三个数据集已经完成多 seed 统计非劣验证。

## 15. 当前结论与剩余验证

当前工程结论是：

> 最终 all-on 在四数据集上使 steady 训练关键路径下降 7.97%--12.59%，使包含 validation/test 的整轮 wall time 下降 5.69%--9.93%。WIKI 三 seed 均获得 steady 效率收益；当前所有 best-val checkpoint 对应 test 回退小于 0.5 个百分点。

当前最终版本不再扩展 standalone kernel 优化，只补足论文所需的统计验证：

1. LASTFM、WikiTalk 和 Flight 至少 3 seeds；
2. 预先固定质量容忍界限和 TTT 阈值；
3. 报告 paired mean、标准差和置信区间；
4. 保留 all-off 对 all-on 的直接比较，不累加模块边际收益。

实验日志位于：

```text
outputs/p0_hierarchical_8gpu_20260818/
outputs/p0_hierarchical_full_8gpu_20260818/
```

## 16. 后续扩展：错误感知的跨分区正负样本选择（未实现）

这是一项独立的训练算法扩展，不属于当前只减少执行冗余的最终版本。目标是在固定的跨分区样本预算内，记录模型对不同类型样本的历史错误程度，并优先选择当前更难学习的正样本和负样本。

### 16.1 第一版只记录什么

第一版不保存逐 EID 或逐负边的巨大字典，而是按以下三元组分桶：

```text
(样本标签 positive/negative, source owner, destination owner)
```

在 8 卡场景下只有 `2 x 8 x 8 = 128` 个桶。每个桶记录：

- 观察数与实际入选数；
- soft error 的和与平方和；
- `soft error >= 0.5` 的样本数；
- 实际 inclusion probability；
- 该桶引起的 root、采样节点、采样边和通信量。

若 logit 为 `z`，正样本记录 `sigmoid(-z)`，负样本记录 `sigmoid(z)`。它表示模型当前的 soft error 或样本难度，未经校准时不能称为真实错误概率。统计先在 detached GPU tensor 上按桶归约，每个 epoch 只同步一次小表；第一个版本必须处于 `observe-only` 模式，不改变样本、随机数、loss、memory 或 mailbox。

### 16.2 后续如何用于选择

使用上一 epoch 冻结的统计生成下一 epoch 的桶级难度，避免 dataloader producer 与 forward 在线争用同一状态：

1. 正样本在 dataloader 现有的 cross-partition 保留位置按桶选择；local 正样本仍全部保留；
2. 负样本先按桶选择 destination partition，再在该分区已有节点池内均匀选择；
3. 正、负样本分别维持原有数量预算，`beta` 继续表示跨分区总预算，不与难度分数混用；
4. 每个桶保留非零探索下限和目标分区 quota，避免困难样本全部集中到单个 owner；
5. validation/test 始终使用原采样策略。

若需要保持原 loss 分布，可记录实际入选概率并使用裁剪后的 importance weight；但对正样本而言，未被选择的 event 也不会触发 memory/mailbox 更新，单纯 loss reweight 无法恢复这段状态转移。因此正样本选择会改变训练语义，必须按新算法单独进行多 seed 的质量和收敛验证。

### 16.3 对性能的影响

- **固定样本预算时没有天然吞吐收益**：多出的分桶统计和概率选择只会增加少量工作；其价值应先看质量和 time-to-target。
- **低开销实现的目标**：GPU `bincount/scatter_add` 可按 stride 采样统计，每个 epoch 只 all-reduce 128 个桶；发布门槛设为 steady critical overhead 不超过 `0.5%`，仍需 ABBA 实测。
- **主要性能风险不是统计表，而是负载倾斜**：如果高错误桶集中在某个远端 owner，即使样本总数不变，也可能增加 remote unique、fetch、通信和最慢 rank 时间。
- **可能产生系统收益的条件**：错误感知选择在相同质量下允许降低 `beta` 或更早达到目标精度；否则它不是系统加速方法。
- **明确不做**：逐样本 CPU 哈希表、每 batch D2H/`.item()`、全节点 `multinomial` 和每 batch 跨 rank 同步。这些路径容易造成数个百分点回退并破坏 pipeline。

推进顺序固定为：`observe-only -> 负样本选择 -> 正样本选择`。负样本不进入 memory 更新，风险较低；正样本决定时序状态和持久化更新，最后再做。启用选择前还需先修正并验证当前 `LocalNegativeSampling` 的 generator 初始化，以及 local/remote inclusion probability 与 loss weight 是否一致。
