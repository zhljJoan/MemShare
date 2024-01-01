from typing import Any, List, Optional, Sequence, Tuple
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

import torch_geometric.nn as pyg_nn
import torch_geometric.datasets as pyg_datasets
import torch_geometric.utils as pyg_utils
from torch_scatter import scatter_mean

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


class SageConv(nn.Module):
    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_feats, in_feats))
        self.bias = nn.Parameter(torch.empty(out_feats))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: int):
        x = F.linear(x, self.weight)
        x = x[edge_index[0]]
        x = scatter_mean(x, edge_index[1], dim=0, dim_size=num_nodes)
        return x + self.bias

class SageGNN(LayerPipe):
    def __init__(self,
        graph: GraphData,
        hidden_dim: int,
        num_layers: int,
        num_snapshots: int,
        group: Any,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.route = graph.to_route(group)
        self.group = group

        self.num_layers = num_layers
        self.num_snapshots = num_snapshots

        num_features = self.graph.node("dst")["x"].size(1)
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            out_ch = in_ch = hidden_dim
            if i == 0:
                in_ch = num_features
            self.layers.append(SageConv(in_ch, out_ch))
    
    def get_route(self) -> Route:
        return self.route
    
    def layer_inputs(self, inputs: Sequence[Tensor] | None = None) -> Sequence[Tensor]:
        if self.layer_id == 0:
            x = self.graph.node("dst")["x"]
        else:
            x, = inputs
        self.register_route(x)
        return (x,)
    
    def layer_forward(self, inputs: Sequence[Tensor]) -> Sequence[Tensor]:
        x, = inputs
        edge_index = self.graph.edge_index()

        x = self.layers[self.layer_id](x, edge_index, self.route.dst_len)
        return (x,)

class SimpleRNN(SequencePipe):
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
        x = self.out(x)
        return (x,), (h, )
    
    def loss_fn(self, inputs, labels) -> Tensor:
        x, = inputs
        mask, y = labels
        
        x = x[mask, -1]
        if x.numel() > 0:
            y = y[mask]
            return F.cross_entropy(x, y)
        else:
            return x.mul(0.0).sum()
    
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

    hidden_dim = 128
    num_layers = 3
    num_snapshots = 200
    num_classes = g.meta()["num_classes"]

    gnn = SageGNN(g, hidden_dim, num_layers, num_snapshots, group=pp_group).to(ctx.device)
    rnn = SimpleRNN(num_classes, hidden_dim, num_layers, device=ctx.device, group=sp_group).to(ctx.device)

    params = []
    for _, net in gnn.get_model():
        params.extend(net.parameters())
    for _, net in rnn.get_model():
        params.extend(net.parameters())
    opt = torch.optim.Adam(params)

    labels = (
        g.node("dst")["train_mask"],
        g.node("dst")["y"],
    )

    for ep in range(1, 100+1):
        seq_len = 200
        xs = []
        
        opt.zero_grad()

        xs = gnn.apply(num_layers, num_snapshots)
        x = torch.cat([x.unsqueeze(1) for x, in xs], dim=1) # (N, S, H)

        # loss = rnn.apply(32, x)[0].square().mean()
        # loss.backward() # sequence and pipeline parallel on each graph nodes

        loss = rnn.fast_backward(32, (x,), labels)
        rnn.all_reduce()
        
        gnn.backward()
        gnn.all_reduce()

        opt.step()
        ctx.sync_print(loss)
