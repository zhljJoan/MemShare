from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch import Tensor
from typing import *

from starrygl.distributed import DistributedContext
from starrygl.data import GraphData
from starrygl.parallel import Route, SequencePipe
from starrygl.parallel.utils import *

import torch_geometric.nn as pyg_nn
import torch_geometric.datasets as pyg_datasets
import torch_geometric.utils as pyg_utils

import logging
logging.getLogger().setLevel(logging.INFO)

def prepare_data(root: str, num_parts, part_algo: str = "metis"):
    ctx = DistributedContext.get_default_context()

    data = pyg_datasets.Planetoid(root, "Cora")[0]
    if data.is_directed():
        data.edge_index, _ = pyg_utils.to_undirected(data.edge_index)
    data.edge_index, _ = pyg_utils.add_remaining_self_loops(data.edge_index)
    data.num_classes = data.y.max().item() + 1

    logging.info(f"num_nodes: {data.num_nodes}")
    logging.info(f"num_edges: {data.num_edges}")
    logging.info(f"num_features: {data.num_features}")
    logging.info(f"num_classes: {data.num_classes}")
    g = GraphData.from_pyg_data(data)

    logging.info(f"GraphData.meta().keys(): {g.meta().keys()}")
    logging.info(f"GraphData.node().keys(): {g.node().keys()}")
    logging.info(f"GraphData.edge().keys(): {g.edge().keys()}")

    g.save_partition(root, num_parts, algorithm=part_algo)
    return g

class SimpleConv(pyg_nn.MessagePassing):
    def __init__(self, in_feats: int, out_feats: int):
        super().__init__(aggr="mean")
        self.linear = nn.Linear(in_feats, out_feats)
    
    def forward(self, x: Tensor, edge_index: Tensor, route: Route):
        dst_len = x.size(0)
        x = route.apply(x) # exchange features
        return self.propagate(edge_index, x=x)[:dst_len]
    
    def message(self, x_j: Tensor):
        return x_j
    
    def update(self, x: Tensor):
        return F.relu(self.linear(x))

class SimpleGNN(nn.Module):
    def __init__(self,
        num_features: int,
        hidden_dims: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            in_ch = hidden_dims if i > 0 else num_features
            out_ch = hidden_dims
            self.layers.append(SimpleConv(in_ch, out_ch))
    
    def forward(self, x: Tensor, edge_index: Tensor, route: Route):
        for layer in self.layers:
            x = layer(x, edge_index, route)
        return x

class SimpleRNN(SequencePipe, nn.Module):
    def __init__(self,
        num_classes: int,
        hidden_dims: int,
        num_layers: int,
        device: Any,
        group: Any,
    ) -> None:
        super().__init__()
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
        self.out = nn.Linear(hidden_dims, num_classes)
    
    def forward(self, inputs, states):
        x, = inputs # (N, L, H)
        h, = states # (N, L, H)

        h = h.transpose(0, 1).contiguous() # (L, N, H)
        x, h = self.gru(x, h) # (N, L, H), (L, N, H)
        h = h.transpose(0, 1).contiguous() # (N, L, H)
        return (x,), (h, )
    
    def loss_fn(self, inputs, labels) -> Tensor:
        x, = inputs
        return x.square().mean()
    
    def get_group(self) -> Any:
        return self.group
    
    def get_init_states(self):
        s = torch.zeros(self.num_layers, self.hidden_dims).to(self.device)
        return (s,)
    

if __name__ == "__main__":
    data_root = "./dataset"
    
    ctx = DistributedContext.init(backend="nccl", use_gpu=True)
    
    hybrid_matrix = ctx.get_hybrid_matrix()
    if hybrid_matrix.size(0) == 1:
        hybrid_matrix = hybrid_matrix.view(2, -1)
    # sp is sequence parallel
    # pp is partition parallel
    sp_group, pp_group = ctx.new_hybrid_subgroups(hybrid_matrix)

    # partition data
    if ctx.rank == 0:
        prepare_data(data_root, dist.get_world_size(pp_group))
    dist.barrier()

    g = GraphData.load_partition(
        data_root,
        dist.get_rank(pp_group), dist.get_world_size(pp_group),
    ).to(ctx.device)
    route = g.to_route(pp_group) # only on subgroup

    num_features = g.node("dst")["x"].size(-1)
    num_classes = g.meta()["num_classes"]
    hidden_dims = 128
    num_layers = 3

    gnn = SimpleGNN(num_features, hidden_dims, num_layers).to(ctx.device)
    rnn = SimpleRNN(num_classes, hidden_dims, num_layers, device=ctx.device, group=sp_group).to(ctx.device)

    opt = torch.optim.Adam([p for p in gnn.parameters()] + [p for p in rnn.parameters()])

    for ep in range(1, 100+1):
        seq_len = 200
        xs = []
        
        opt.zero_grad()

        for _ in range(seq_len): # snapshot parallel between partition parallel subgroups
            z = gnn(
                x = g.node("dst")["x"],
                edge_index = g.edge_index(),
                route = route,   # 
            )
            xs.append(z.unsqueeze(1))
        x = torch.cat(xs, dim=1) # (N, S, H)

        # loss = rnn.apply(32, x)[0].square().mean()
        # loss.backward() # sequence and pipeline parallel on each graph nodes

        loss = rnn.fast_backward(32, (x,), (g.node("dst")["train_mask"],))
        
        # all reduce
        all_reduce_gradients(rnn)
        all_reduce_buffers(rnn)

        all_reduce_gradients(gnn)
        all_reduce_buffers(gnn)

        opt.step()
        ctx.sync_print(loss)
