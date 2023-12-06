import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
import torch.distributed.rpc as rpc

from torch import Tensor
from typing import *

import os
import time
import psutil

# from starrygl.loader import DataLoader, TensorBuffers
from starrygl.loader.route import Route
from starrygl.loader.buffer import TensorBuffer
from starrygl.loader.utils import init_local_edge_index, local_partition_fn
from starrygl.parallel import init_process_group
from starrygl.utils import partition_load, main_print, sync_print

all_nparts = [
    [0, 1],
    [2, 3, 4],
    [5, 6],
]

all_eparts = [
    [
        [2, 0], [1, 0], [0, 1], [3, 1], [4, 1],
    ],
    [
        [0, 2], [3, 2], [5, 2], [1, 3], [2, 3],
        [4, 3], [6, 3], [1, 4], [3, 4],
    ],
    [
        [2, 5], [6, 5], [5, 6], [3, 6],
    ],
]


# def get_route_table(device = "cpu") -> Tuple[RouteTable, Tensor]:
#     assert dist.get_world_size() == 3
#     rank = dist.get_rank()
#     ids = torch.tensor(all_nparts[rank], dtype=torch.long, device=device)
#     edge_index = torch.tensor(all_eparts[rank], dtype=torch.long, device=device).t()

#     src_ids, local_edge_index = init_local_edge_index(ids, edge_index)
#     return RouteTable(src_ids, ids.size(0)), local_edge_index

# def get_features(device = "cpu") -> Tensor:
#     assert dist.get_world_size() == 3
#     rank = dist.get_rank()
#     return torch.tensor(all_nparts[rank], dtype=torch.float32, device=device) + 1.0

if __name__ == "__main__":
    # 启动分布式进程组，并分配计算设备
    device = init_process_group(backend="nccl")
    
    # 加载数据集
    pdata = partition_load("./cora", algo="metis").to(device)
    route, local_edge_index = Route.from_edge_index(pdata.ids, pdata.edge_index)

    # route_table = RouteTable(src_ids, dst_size)
    # sync_print(route_table.fw_0)

    # route_table = get_route_table()[0]
    # for name, buffer in route_table.named_buffers():
    #     if name.startswith("fw_") or name.startswith("bw_"):
    #         sync_print(name, buffer.tolist())

    tb = TensorBuffer(pdata.num_features, route.src_size, route).cpu()

    print(tb.data.device, pdata.x.device)

    tb.data[:pdata.num_nodes] = pdata.x
    dist.barrier()
    tb.broadcast()

    print(tb.data[pdata.num_nodes:].sum(dim=-1))
    # assert (tb.data[:pdata.num_nodes] == pdata.x).all()
    # local_src_ids = torch.arange(route.src_size)
    # for local_ids, remote_ids in route.parts_iter(local_src_ids):
    #     sync_print(local_ids.size(), remote_ids.size())

    # router, edge_index = get_route_table()
    # x = get_features()[:,None]
    # buffer = TensorBuffers(router, channels=[1])
    
    # buffer.local_set(0, x)
    # sync_print(buffer.local_get(0).view(-1).tolist())
    # buffer.broadcast()
    # main_print("="*64)
    # sync_print(buffer.local_get(0).view(-1).tolist())

    # main_print("+"*64)

    # buffer.zero_grad()
    # buffer.local_set(0, x)
    # sync_print(buffer.local_get(0).view(-1).tolist())
    # buffer.broadcast2()
    # main_print("="*64)
    # sync_print(buffer.local_get(0).view(-1).tolist())

    rpc.shutdown()

    # buffer.remote_get(0, 0, async_op=True).then()

    # num_parts = 50
    # node_parts = local_partition_fn(dst_size, local_edge_index, num_parts)
    # feat_buffer = TensorBuffers(src_ids.size(0), [pdata.num_features] * 4)
    # grad_buffer = TensorBuffers(src_ids.size(0), [pdata.num_features] * 4)

    # node_time = torch.rand(dst_size)
    # edge_time = torch.rand(local_edge_index.size(1))
    # edge_attr = torch.randn(local_edge_index.size(1), 10)
    
    # loader = DataLoader(node_parts, feat_buffer, grad_buffer, local_edge_index, edge_attr, edge_time, node_time)

    # feat_buffer.set(0, None, collect_feat0(src_ids, src_ids[:dst_size], pdata.x))

    # for handle, edge_index, edge_attr in loader.iter(2, layer_id=0, device=device, filter=lambda x: x < 0.5):
    #     print(handle.feat0.sum(dim=-1))
    #     print(edge_index)
    #     print(edge_attr.sum(dim=-1))
    #     handle.push_and_pull(handle.feat0[:handle.dst_size], 0)
    #     assert (handle.fetch_feat() == handle.feat0).all()