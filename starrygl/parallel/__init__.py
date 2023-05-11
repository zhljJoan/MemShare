import torch
import torch.nn as nn
import torch.distributed as dist

import os

from .degree import compute_degree, compute_gcn_norm
from .sync_bn import SyncBatchNorm
from ..core import with_gloo, with_nccl
from ..core.route import get_executor

def convert_parallel_model(
    net: nn.Module,
    find_unused_parameters=False,
) -> nn.parallel.DistributedDataParallel:
    net = SyncBatchNorm.convert_sync_batchnorm(net)
    net = nn.parallel.DistributedDataParallel(net,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=False,
    )
    for name, buffer in net.named_buffers():
        if name.endswith("last_embd"):
            continue
        if name.endswith("last_w"):
            continue
        dist.broadcast(buffer, src=0)
    return net

def init_process_group(backend: str = "gloo") -> torch.device:
    dist.init_process_group(backend)
    if backend == "nccl":
        local_rank = os.getenv("LOCAL_RANK")
        if local_rank is None:
            device = torch.device(f"cuda:{dist.get_rank()}")
        else:
            local_rank = int(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    get_executor() # initialize route executor
    return device