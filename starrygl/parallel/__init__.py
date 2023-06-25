import torch
import torch.nn as nn
import torch.distributed as dist

import os
from typing import *

from .degree import compute_in_degree, compute_out_degree, compute_gcn_norm
from .sync_bn import SyncBatchNorm

def convert_parallel_model(
    net: nn.Module,
    find_unused_parameters=False,
) -> nn.parallel.DistributedDataParallel:
    net = SyncBatchNorm.convert_sync_batchnorm(net)
    net = nn.parallel.DistributedDataParallel(net,
        find_unused_parameters=find_unused_parameters,
        # broadcast_buffers=False,
    )
    # for name, buffer in net.named_buffers():
    #     if name.endswith("last_embd"):
    #         continue
    #     if name.endswith("last_w"):
    #         continue
    #     dist.broadcast(buffer, src=0)
    return net

def init_process_group(backend: str = "gloo") -> torch.device:

    local_rank = os.getenv("LOCAL_RANK") or os.getenv("OMPI_COMM_WORLD_LOCAL_RANK")
    if local_rank is not None:
        local_rank = int(local_rank)
    
    dist.init_process_group(backend)
    
    if backend == "nccl" or backend == "mpi":
        if local_rank is None:
            device = torch.device(f"cuda:{dist.get_rank()}")
        else:
            device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    global _COMPUTE_DEVICE
    _COMPUTE_DEVICE = device
    return device

_COMPUTE_DEVICE = torch.device("cpu")
def get_compute_device() -> torch.device:
    global _COMPUTE_DEVICE
    return _COMPUTE_DEVICE