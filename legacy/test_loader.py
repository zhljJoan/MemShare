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

from starrygl.loader import NodeLoader, TensorBuffer, RouteContext
from starrygl.parallel import init_process_group
from starrygl.utils import partition_load, main_print, sync_print

if __name__ == "__main__":
    # 启动分布式进程组，并分配计算设备
    device = init_process_group(backend="nccl")
    
    # 加载数据集
    pdata = partition_load("./cora", algo="metis")
    loader = NodeLoader(pdata.ids, pdata.edge_index, device)

    hidden_size = 64

    buffers: List[TensorBuffer] = [
        TensorBuffer(pdata.num_features, loader.src_size, loader.route),
        TensorBuffer(hidden_size, loader.src_size, loader.route),
    ]
    buffers[0].data[:loader.dst_size] = pdata.x
    buffers[0].broadcast()

    for handle in loader.iter(128):
        with RouteContext() as ctx:
            sync_print(handle.src_size, handle.dst_size)
            dst_feats = handle.get_dst_feats(buffers[0]).wait()
            ext_fut = handle.get_ext_feats(buffers[1])
            sync_print(dst_feats.size())

            src_feats, fut = handle.push_and_pull(dst_feats[:,:hidden_size], ext_fut, buffers[1])
            ctx.add_futures(fut)
            sync_print(src_feats.size())
    rpc.shutdown()

    