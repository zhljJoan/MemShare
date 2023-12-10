import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.rpc as rpc

import os
from torch import Tensor
from typing import *

import logging

from .cclib import all_to_all_v
from .rpc import rpc_remote_call, rpc_remote_void_call



__all__ = [
    "DistributedContext",
]


class DistributedContext:
    @staticmethod
    def init(
        backend: str,
        use_gpu: Optional[bool] = None,
        rpc_gpu: Optional[bool] = None,
    ) -> 'DistributedContext':
        if DistributedContext.is_initialized():
            raise RuntimeError("not allowed to call init method twice.")

        rank = int(os.getenv("RANK") or os.getenv("OMPI_COMM_WORLD_RANK"))
        world_size = int(os.getenv("WORLD_SIZE") or os.getenv("OMPI_COMM_WORLD_SIZE"))
        local_rank = os.getenv("LOCAL_RANK") or os.getenv("OMPI_COMM_WORLD_LOCAL_RANK")

        if local_rank is None:
            logging.warning(f"LOCAL_RANK has not been set, using the default value 0.")
            os.environ["LOCAL_RANK"] = local_rank = "0"
        local_rank = int(local_rank)

        backend = backend.lower()
        if use_gpu is None:
            use_gpu = False
            if backend == "nccl" or backend == "mpi":
                use_gpu = True
        else:
            use_gpu = bool(use_gpu)
        
        if rpc_gpu is None:
            rpc_gpu = use_gpu
        else:
            rpc_gpu = bool(rpc_gpu)
        
        master_addr = os.environ["MASTER_ADDR"]
        master_port = int(os.environ["MASTER_PORT"])
        ccl_init_url = f"tcp://{master_addr}:{master_port}"
        rpc_init_url = f"tcp://{master_addr}:{master_port + 1}"

        ctx = DistributedContext(
            backend=backend,
            ccl_init_method=ccl_init_url,
            rpc_init_method=rpc_init_url,
            rank=rank, world_size=world_size,
            local_rank=local_rank,
            use_gpu=use_gpu, rpc_gpu=rpc_gpu,
        )
        _set_default_dist_context(ctx)
        
        return ctx
    
    @staticmethod
    def get_default_context() -> 'DistributedContext':
        ctx = _get_default_dist_context()
        if ctx is None:
            raise RuntimeError("please call the init method first.")
        return ctx

    @staticmethod
    def is_initialized() -> bool:
        return _get_default_dist_context() is not None

    def __init__(self,
        backend: str,
        ccl_init_method: str,
        rpc_init_method: str,
        rank: int, world_size: int,
        local_rank: int,
        use_gpu: bool, rpc_gpu: bool,
    ) -> None:
        if use_gpu:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")

        dist.init_process_group(
            backend=backend,
            init_method=ccl_init_method,
            rank=rank, world_size=world_size,
        )

        rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
        rpc_backend_options.init_method = rpc_init_method

        if use_gpu and rpc_gpu:
            device_maps = torch.zeros(world_size, dtype=torch.long, device=device)
            device_maps[rank] = local_rank
            dist.all_reduce(device_maps, op=dist.ReduceOp.SUM)
            for i, dev in enumerate(device_maps.tolist()):
                rpc_backend_options.set_device_map(
                    to=f"worker{i}",
                    device_map={local_rank: dev},
                )

        rpc.init_rpc(
            name=f"worker{rank}",
            rank=rank, world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        self._local_rank = local_rank
        self._compute_device = device

        self.__temp_ag_remote_object: Optional[rpc.RRef] = None
    
    def shutdown(self):
        rpc.shutdown()
    
    @property
    def rank(self) -> int:
        return dist.get_rank()
    
    @property
    def world_size(self) -> int:
        return dist.get_world_size()
    
    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def device(self) -> torch.device:
        return self._compute_device
    
    def get_default_group(self):
        return dist.distributed_c10d._get_default_group()
    
    def get_default_store(self):
        return dist.distributed_c10d._get_default_store()
    
    def get_worker_info(self, rank: Optional[int] = None) -> rpc.WorkerInfo:
        rank = dist.get_rank() if rank is None else rank
        return rpc.get_worker_info(f"worker{rank}")
    
    def remote_call(self, method, rref: rpc.RRef, *args, **kwargs):
        return rpc_remote_call(method, rref, *args, **kwargs)
    
    def remote_void_call(self, method, rref: rpc.RRef, *args, **kwargs):
        return rpc_remote_void_call(method, rref, *args, **kwargs)
    
    def all_to_all_v(self,
        output_tensor_list: List[Tensor],
        input_tensor_list: List[Tensor],
        group: Optional[Any] = None,
        async_op: bool = False,
    ):
        return all_to_all_v(
            output_tensor_list,
            input_tensor_list,
            group=group,
            async_op=async_op,
        )
    
    def all_to_all_g(self,
        input_tensor_list: List[Tensor],
        group: Optional[Any] = None,
        async_op: bool = False,
    ):
        send_sizes = [t.size(0) for t in input_tensor_list]
        recv_sizes = self.get_all_to_all_recv_sizes(send_sizes, group)
        
        output_tensor_list: List[Tensor] = []
        for s, t in zip(recv_sizes, input_tensor_list):
            output_tensor_list.append(
                torch.empty(s, *t.shape[1:], dtype=t.dtype, device=t.device),
            )
        
        work = all_to_all_v(
            output_tensor_list,
            input_tensor_list,
            group=group,
            async_op=async_op,
        )

        if async_op:
            assert work is not None
            return output_tensor_list, work
        else:
            return output_tensor_list
    
    def get_all_to_all_recv_sizes(self,
        send_sizes: List[int],
        group: Optional[Any] = None,
    ) -> List[int]:
        world_size = dist.get_world_size(group)
        assert len(send_sizes) == world_size

        if dist.get_backend(group) == "gloo":
            send_t = torch.tensor(send_sizes, dtype=torch.long)
        else:
            send_t = torch.tensor(send_sizes, dtype=torch.long, device=self.device)
        recv_t = torch.empty_like(send_t)

        dist.all_to_all_single(recv_t, send_t, group=group)
        return recv_t.tolist()
    
    def all_gather_remote_objects(self, obj: Any) -> List[rpc.RRef]:
        if not isinstance(obj, rpc.RRef):
            obj = rpc.RRef(obj)
        self.__temp_ag_remote_object = obj
        
        dist.barrier()

        futs: List[torch.futures.Future] = []
        for i in range(self.world_size):
            info = rpc.get_worker_info(f"worker{i}")
            futs.append(rpc.rpc_async(info, DistributedContext._remote_object))

        rrefs: List[rpc.RRef] = []
        for f in futs:
            f.wait()
            rrefs.append(f.value())
        
        dist.barrier()

        self.__temp_ag_remote_object = None
        return rrefs

    @staticmethod
    def _remote_object():
        ctx = DistributedContext.get_default_context()
        return ctx.__temp_ag_remote_object
    
    def sync_print(self, *args, **kwargs):
        for i in range(self.world_size):
            if i == self.rank:
                print(f"rank {self.rank}:", *args, **kwargs)
            dist.barrier()

    def main_print(self, *args, **kwargs):
        if self.rank == 0:
            print(*args, **kwargs)
        dist.barrier()
    

_DEFAULT_DIST_CONTEXT: Optional['DistributedContext'] = None

def _get_default_dist_context():
    global _DEFAULT_DIST_CONTEXT
    return _DEFAULT_DIST_CONTEXT

def _set_default_dist_context(ctx):
    global _DEFAULT_DIST_CONTEXT
    _DEFAULT_DIST_CONTEXT = ctx
