"""单机四卡训练，启动指令：torchrun --nproc_per_node 4 --standalone demo_dtdg.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch import Tensor
from typing import *

from starrygl.distributed import DistributedContext
from starrygl.data import GraphData
from starrygl.parallel import Route, SequencePipe, LayerPipe
from starrygl.parallel.utils import *

from torch_scatter import scatter

from torch_geometric_temporal.dataset import TwitterTennisDatasetLoader

import math

import logging
logging.getLogger().setLevel(logging.INFO)

def prepare_data(root: str, num_parts):
    dataset = TwitterTennisDatasetLoader().get_dataset()

    x = []
    y = []
    edge_index = []
    edge_times = []
    edge_attr = []
    snapshot_count = 0
    for i, data in enumerate(dataset):
        x.append(data.x[:,None,:])
        y.append(data.y[:,None])
        edge_index.append(data.edge_index)
        edge_times.append(torch.full_like(data.edge_index[0], i)) # 利用snapshot id作为时间，也可以用真实时间戳
        edge_attr.append(data.edge_attr)
        snapshot_count += 1
    x = torch.cat(x, dim=1)
    y = torch.cat(y, dim=1)
    edge_index = torch.cat(edge_index, dim=1)
    edge_times = torch.cat(edge_times, dim=0)
    edge_attr = torch.cat(edge_attr, dim=0)

    g = GraphData(edge_index, num_nodes=x.size(0))
    g.node()["x"] = x
    g.node()["y"] = y
    g.edge()["time"] = edge_times
    g.edge()["attr"] = edge_attr
    g.meta()["num_snapshots"] = snapshot_count # 快照数量

    logging.info(f"GraphData.meta().keys(): {g.meta().keys()}")
    logging.info(f"GraphData.node().keys(): {g.node().keys()}")
    logging.info(f"GraphData.edge().keys(): {g.edge().keys()}")

    g.save_partition(root, num_parts, algorithm="random") # 采用随机划分算法
    return g

class SageConv(nn.Module):
    """基础的GraphSAGE卷积层，采用平均聚合函数
    """
    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, num_nodes: int):
        assert edge_attr.dim() == 1
        
        x = scatter(
            src=x[edge_index[0]] * edge_attr[:,None],
            index=edge_index[1],
            reduce="sum",
            dim=0,
            dim_size=num_nodes,
        )

        e = scatter(
            src=edge_attr,
            index=edge_index[1],
            reduce="sum",
            dim=0,
            dim_size=num_nodes,
        ).clamp_min(1) # 避免孤立点除0
        
        x = x / e[:,None]
        return self.linear(x)

class SyncSAGE(nn.Module):
    def __init__(self,
        graph: GraphData,
        hidden_dim: int,
        num_layers: int,
        group: Any,
    ) -> None:
        super().__init__()
        self.graph = graph # 图数据
        self.route = graph.to_route(group) # 在进程组上建立通信路由
        self.group = group

        self.num_features = self.graph.node("dst")["x"].size(-1) # 每个分区是一个二分图，只有dst节点有数据
        self.num_snapshots = self.graph.meta()["num_snapshots"]

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        last_ch = self.num_features
        for i in range(num_layers):
            self.layers.append(SageConv(last_ch, hidden_dim))
            if i == num_layers - 1:
                break

            self.norms.append(nn.LayerNorm(hidden_dim))
            last_ch = hidden_dim
    
    def get_snapshot(self, i: int):
        num_nodes = self.graph.node("dst").num_nodes
        x = self.graph.node("dst")["x"][:,i,:]
        time_mask = self.graph.edge()["time"] == i
        edge_attr = self.graph.edge()["attr"][time_mask]
        edge_index = self.graph.edge_index()[:,time_mask]
        return num_nodes, x, edge_index, edge_attr

    def forward(self, snapshot_id: Optional[int] = None):
        if snapshot_id is None: # 如果没有传入snapshot_id，则forward所有snapshot并拼接
            xs = []
            for i in range(self.num_snapshots):
                xs.append(self.forward(i).unsqueeze(1))
            return torch.cat(xs, dim=1)
        
        num_nodes, x, edge_index, edge_attr = self.get_snapshot(snapshot_id)
        for i in range(self.num_layers):
            x = self.route.apply(x) # 将所有dst节点的表征同步到src节点上
            x = self.layers[i](x, edge_index, edge_attr, num_nodes)
            if i == self.num_layers - 1:
                break
            x = self.norms[i](x)
            x = F.relu(x)
        return x

class AsyncSAGE(LayerPipe):
    """采用快照并行的异步训练
    """
    def __init__(self,
        graph: GraphData,
        hidden_dim: int,
        num_layers: int,
        group: Any,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.route = graph.to_route(group)
        self.group = group

        self.num_features = self.graph.node("dst")["x"].size(-1) # 每个分区是一个二分图，只有dst节点有数据
        self.num_snapshots = self.graph.meta()["num_snapshots"]

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        last_ch = self.num_features
        for i in range(num_layers):
            self.layers.append(SageConv(last_ch, hidden_dim))
            if i == num_layers - 1:
                break
            
            self.norms.append(nn.LayerNorm(hidden_dim))
            last_ch = hidden_dim
    
    def get_snapshot(self, i: int):
        num_nodes = self.graph.node("dst").num_nodes
        time_mask = self.graph.edge()["time"] == i
        edge_attr = self.graph.edge()["attr"][time_mask]
        edge_index = self.graph.edge_index()[:,time_mask]
        return num_nodes, edge_index, edge_attr
    
    def get_route(self) -> Route:
        """必须重写该函数，底层执行引擎需要识别用于通信的route
        """
        return self.route
    
    def layer_inputs(self, inputs: Sequence[Tensor] | None = None) -> Sequence[Tensor]:
        """必须重写该函数，用于准备数据
        """
        if self.layer_id == 0: # 如果是第一层，则采用原始的图数据
            x = self.graph.node("dst")["x"][:,self.snapshot_id,:] # 可以通过layer_id和snapshot_id来判断当前正在处理第几层和第几个快照
        else:
            x, = inputs # 如果不是第一层，则直接返回上一层的输出
        self.register_route(x) # 通过该方法对输出打上标记，被标记的输出会通过route被同步成src节点，否则依然是dst节点
        return (x,)
    
    def layer_forward(self, inputs: Sequence[Tensor]) -> Sequence[Tensor]:
        """必须重写改函数，用于执行模型
        """
        num_nodes, edge_index, edge_attr = self.get_snapshot(self.snapshot_id)
        x = self.layers[self.layer_id](*inputs, edge_index, edge_attr, num_nodes)
        return (x,)
    
    def forward(self):
        """调用apply方法后，拼接所有的快照
        """
        xs = self.apply(self.num_layers, self.num_snapshots)
        return torch.cat([x.unsqueeze(1) for x, in xs], dim=1)
    

class SimpleRNN(SequencePipe):
    def __init__(self,
        graph: GraphData,
        hidden_dims: int,
        num_layers: int,
        device: Any,
        group: Any,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.device = device
        self.group = group

        self.num_layers = num_layers
        self.hidden_dims = hidden_dims

        self.gru = nn.GRU(
            input_size = hidden_dims,
            hidden_size = hidden_dims,
            num_layers = num_layers,
            batch_first = True,
        )
        self.out = nn.Linear(hidden_dims, 1)
    
    def forward(self, inputs, states):
        """每个GPU上的时序单元
        """
        x, = inputs # (N, L, H) (节点数量，快照数量，隐藏层维度)
        h, = states # (N, L, H)

        h = h.transpose(0, 1).contiguous() # (L, N, H) 需要节点顺序放在第0维，并且是连续张量
        x, h = self.gru(x, h) # (N, L, H), (L, N, H)
        h = h.transpose(0, 1).contiguous() # (N, L, H)
        x = self.out(x)
        return (x,), (h, )
    
    def loss_fn(self, inputs, labels) -> Tensor:
        """如果调用fast_backward()，则需要重写该方法计算每个micro-batch的损失函数
        """
        x, = inputs
        y, = labels

        loss_scale = x.size(0) / self.graph.node("dst").num_nodes
        return F.mse_loss(x, y) * loss_scale # 最好对每一个micro batch的loss进行缩放，保证最终损失值一致
    
    def get_group(self) -> Any:
        """必须重写该方法，用于识别子进程组
        """
        return self.group
    
    def get_init_states(self):
        """必须重写该方法，返回每个节点的初始状态模板
        """
        s = torch.zeros(self.num_layers, self.hidden_dims).to(self.device)
        return (s,)

def get_graph(
    root: str,
    sp_group: Any,
    pp_group: Any,
):
    g = GraphData.load_partition(
        root, part_id=dist.get_rank(pp_group), num_parts=dist.get_world_size(pp_group), # 采用分区并行加载图快照
        algorithm="random", # load_partition和save_partition所用的分区算法要一致
    )

    snap_part_id = dist.get_rank(sp_group)
    num_snap_parts = dist.get_world_size(sp_group)
    num_snapshots = g.meta()["num_snapshots"]

    stride = math.ceil(num_snapshots * 1.0 / num_snap_parts)
    start = snap_part_id * stride
    end = min(num_snapshots, start + stride)
    
    # 只保留在[start, end)内的快照
    time = g.edge()["time"]
    mask = (start <= time) & (time < end)
    
    edge_attr = g.edge()["attr"][mask]
    edge_index = g.edge_index()[:,mask]
    
    sg = GraphData.from_bipartite(
        edge_index=edge_index,
        num_src_nodes=g.node("src").num_nodes,
        num_dst_nodes=g.node("dst").num_nodes,
        raw_src_ids=g.node("src")["raw_ids"],
        raw_dst_ids=g.node("dst")["raw_ids"],
    )
    sg.node("dst")["x"] = g.node("dst")["x"][:,start:end].clone()
    sg.node("dst")["y"] = g.node("dst")["y"][:,start:end].clone()
    sg.edge()["time"] = time[mask] - start # 快照偏移从0开始
    sg.edge()["attr"] = edge_attr
    sg.meta()["num_snapshots"] = end - start
    return sg

if __name__ == "__main__":
    data_root = "./dataset"
    
    ctx = DistributedContext.init(backend="nccl", use_gpu=True)
    hybrid_matrix = ctx.get_hybrid_matrix().view(2, 2) # (2, 2)
    """返回进程矩阵: [[0, 1], [2, 3]]
    其中[0, 1]和[2, 3]分别组成两个pp_group，用于训练t0-t1和t1-t2的两段图快照
    [0, 2]有相同的dst节点，可以组成一个sp_group用于流水线训练时序模型，[1, 3]同理
    """

    sp_group, pp_group = ctx.new_hybrid_subgroups(hybrid_matrix)

    if ctx.rank == 0: # ctx.rank和ctx.world_size返回的是全局进程组（默认进程组）的属性
        prepare_data(data_root, dist.get_world_size(pp_group)) # 这里只需要划分成pp_group进程组数量的分区即可
    dist.barrier() # 这里加一个barrier，保证其它进程在读数据的时候，rank 0已经写完数据

    g = get_graph(data_root, sp_group, pp_group).to(ctx.device)
    ctx.sync_print(f'x: {g.node("dst")["x"].size()}')
    ctx.sync_print(f'y: {g.node("dst")["y"].size()}')
    ctx.sync_print(f'edge_index: {g.edge_index().size()}')
    ctx.sync_print(f'edge_time: {g.edge()["time"].size()}')
    ctx.sync_print(f'edge_attr: {g.edge()["attr"].size()}')

    hidden_dim = 128
    num_layers = 3

    # 创建支持分区并行的GNN模型
    sync_gnn = SyncSAGE(g, hidden_dim, num_layers, pp_group).to(ctx.device) # 原始的分区并行，只依赖于route.apply()
    async_gnn = AsyncSAGE(g, hidden_dim, num_layers, pp_group).to(ctx.device) # 快照并行，重叠通信和计算，依赖于LayerPipe

    # 创建时间并行模型
    rnn = SimpleRNN(g, hidden_dim, num_layers, ctx.device, sp_group).to(ctx.device)

    # 创建优化器
    params = []
    params.extend(sync_gnn.parameters())
    params.extend(async_gnn.parameters())
    params.extend(rnn.parameters())
    opt = torch.optim.Adam(params)

    # 训练标签
    y = g.node("dst")["y"].unsqueeze(-1)

    if True:
        """案例1：采用采用sync_gnn配合rnn训练
        """
        ctx.main_print("\nCase 1:")
        opt.zero_grad()

        x = sync_gnn.forward()  # 计算GNN
        h, = rnn.apply(32, x)   # 计算RNN, micro_batch_size = 32
        loss = F.mse_loss(h, y) # 计算损失

        # 反向传播
        loss.backward()

        # 合并梯度
        rnn.all_reduce()
        all_reduce_gradients(sync_gnn)
        all_reduce_buffers(sync_gnn)

        opt.step()
        ctx.sync_print(f"loss: {loss.item():.6f}")
    
    if True:
        """案例2：采用采用async_gnn配合rnn训练
        """
        ctx.main_print("\nCase 2:")
        opt.zero_grad()

        x = async_gnn.forward()  # 计算GNN
        h, = rnn.apply(32, x)   # 计算RNN, micro_batch_size = 32
        loss = F.mse_loss(h, y) # 计算损失

        # 反向传播
        loss.backward()
        async_gnn.backward() # !!! 目前快照并行的backward需要手动调用

        # 合并梯度
        rnn.all_reduce()
        async_gnn.all_reduce()

        opt.step()
        ctx.sync_print(f"loss: {loss.item():.6f}")
    
    if True:
        """案例3：采用采用async_gnn配合rnn.fast_backward训练。fast_backward只能在最后一层rnn调用，可以减少一次前向计算
        """
        ctx.main_print("\nCase 3:")
        opt.zero_grad()

        x = async_gnn.forward()  # 计算GNN

        # 计算RNN的同时直接求反向梯度
        loss = rnn.fast_backward(32, inputs=(x,), labels=(y,))

        # 反向传播
        # loss.backward() # 此时RNN的backward不需要调用
        async_gnn.backward() # async_gnn的反向传播依然需要调用，如果是sync_gnn则不需要调用反向传播

        # 合并梯度
        rnn.all_reduce()
        async_gnn.all_reduce()

        opt.step()
        ctx.sync_print(f"loss: {loss.item():.6f}")
