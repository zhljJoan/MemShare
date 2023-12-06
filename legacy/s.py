import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import *

import os
import time
import psutil

from starrygl.core import NodeProbe
from starrygl.nn.convs.s_gat_conv import GATConv
from starrygl.graph import DistGraph

from starrygl.parallel import init_process_group, convert_parallel_model
from starrygl.parallel import compute_gcn_norm, SyncBatchNorm, with_nccl

from starrygl.utils import train_epoch, eval_epoch, partition_load, main_print, sync_print


class Net(nn.Module):
    def __init__(self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        last_ch = in_channels
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            out_ch = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(GATConv(last_ch, out_ch, heads))
            last_ch = out_ch
    
    def forward(self, g: DistGraph, probe: NodeProbe):
        for i in range(self.num_layers):
            shrink = g.cache.layers[i].get_shrink_data()
            if i == 0:
                x = g.cache.fetch_pull_tensor(i)
            else:
                # x = torch.ones(dst_idx.size(0), self.hidden_channels, device=x.device)
                x, _ = g.cache.update_cache(i, x, dst_idx)
            dst_idx = shrink.dst_idx
            # print(shrink.edge_index[1].unique().size(0), shrink.dst_size)
            # print(x.size(0), shrink.edge_index[0].unique().size(0), shrink.src_size)
            # e = x[shrink.edge_index[0]]
            # from torch_scatter import scatter
            from starrygl.utils.printer import main_print
            main_print(x.requires_grad)
            # print(e.size(0), shrink.edge_index[1].size(0), shrink.dst_size, shrink.edge_index[1].max().item())
            # scatter(e, shrink.edge_index[1], dim=0, dim_size=shrink.dst_size)
            # return
            x = self.convs[i](shrink, x)
            x = F.relu(x)
            x = probe.apply(i, x, dst_idx)
        return x

if __name__ == "__main__":
    # 启动分布式进程组，并分配计算设备
    device = init_process_group(backend="nccl")
    
    # 加载数据集
    pdata = partition_load("./cora", algo="metis").to(device)
    g = DistGraph(
        ids=pdata.ids,
        edge_index=pdata.edge_index,
        num_features=64,
        num_layers=3,
        cache_device="cpu"
    ).to(device)

    cached_data_0, _ = g.route.forward_a2a(pdata.x).get()
    g.cache.replace_layer_data(0, cached_data_0.to(g.cache.cache_device))

    probe = NodeProbe(
        g.dst_size,
        num_layers=3,
        num_samples=128,
    ).to(device)
    probe.assign_message_cache(g.cache)
    probe.layers[-1].warmup_sample()

    net = Net(pdata.num_features, 64, pdata.num_classes, num_layers=3).to(device)
    net(g, probe).sum().backward()