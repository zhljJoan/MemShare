import torch
import torch.nn as nn
import torch.distributed as dist

import os

def convert_parallel_model(
    net: nn.Module,
    find_unused_parameters=False,
) -> nn.parallel.DistributedDataParallel:
    if dist.get_backend() == "nccl":
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = nn.parallel.DistributedDataParallel(net,
        find_unused_parameters=find_unused_parameters,
    )
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
    return device