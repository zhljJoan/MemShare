import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.rpc as rpc

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
    )
    return net

def init_process_group(backend: str = "gloo") -> torch.device:

    rank = int(os.getenv("RANK") or os.getenv("OMPI_COMM_WORLD_RANK"))
    world_size = int(os.getenv("WORLD_SIZE") or os.getenv("OMPI_COMM_WORLD_SIZE"))
    
    dist.init_process_group(
        backend=backend,
        init_method=ccl_init_method(),
        rank=rank, world_size=world_size,
    )

    rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = rpc_init_method()
    for i in range(world_size):
        rpc_backend_options.set_device_map(f"worker{i}", {rank: i})
    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank, world_size=world_size,
        rpc_backend_options=rpc_backend_options,
    )

    local_rank = os.getenv("LOCAL_RANK") or os.getenv("OMPI_COMM_WORLD_LOCAL_RANK")
    if local_rank is not None:
        local_rank = int(local_rank)
    
    if backend == "nccl" or backend == "mpi":
        device = torch.device(f"cuda:{local_rank or rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    global _COMPUTE_DEVICE
    _COMPUTE_DEVICE = device
    return device

def rank_world_size() -> Tuple[int, int]:
    return dist.get_rank(), dist.get_world_size()

def get_worker_info(rank: Optional[int] = None) -> rpc.WorkerInfo:
    rank = dist.get_rank() if rank is None else rank
    return rpc.get_worker_info(f"worker{rank}")

_COMPUTE_DEVICE = torch.device("cpu")
def get_compute_device() -> torch.device:
    global _COMPUTE_DEVICE
    return _COMPUTE_DEVICE

_TEMP_AG_REMOTE_OBJECT = None
def _remote_object():
    global _TEMP_AG_REMOTE_OBJECT
    return _TEMP_AG_REMOTE_OBJECT

def all_gather_remote_objects(obj: Any) -> List[rpc.RRef]:
    global _TEMP_AG_REMOTE_OBJECT
    _TEMP_AG_REMOTE_OBJECT = rpc.RRef(obj)
    dist.barrier()
    world_size = dist.get_world_size()

    futs: List[torch.futures.Future] = []
    for i in range(world_size):
        info = get_worker_info(i)
        futs.append(rpc.rpc_async(info, _remote_object))
    
    rrefs: List[rpc.RRef] = []
    for f in futs:
        f.wait()
        rrefs.append(f.value())
    dist.barrier()
    _TEMP_AG_REMOTE_OBJECT = None
    return rrefs

def ccl_init_method() -> str:
    master_addr = os.environ["MASTER_ADDR"]
    master_port = int(os.environ["MASTER_PORT"])
    return f"tcp://{master_addr}:{master_port}"

def rpc_init_method() -> str:
    master_addr = os.environ["MASTER_ADDR"]
    master_port = int(os.environ["MASTER_PORT"])
    return f"tcp://{master_addr}:{master_port+1}"