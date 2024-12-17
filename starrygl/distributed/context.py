from math import floor
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.rpc as rpc

import os
from torch import Tensor
from typing import *

import socket
from contextlib import contextmanager

import logging

from .rpc import *



__all__ = [
    "DistributedContext",
]


class DistributedContext:
    """Global context manager for distributed training
    """
    @staticmethod
    def init(
        backend: str,
        use_rpc: bool = False,
        use_gpu: Optional[bool] = None,
        rpc_gpu: Optional[bool] = None,
        cache_use_rpc: bool =False,
        memory_group_num = 1,
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
            use_rpc=use_rpc,
            use_gpu=use_gpu,
            rpc_gpu=rpc_gpu,
            cache_use_rpc = cache_use_rpc,
            memory_group_num=memory_group_num
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
        use_rpc: bool,
        use_gpu: bool,
        rpc_gpu: bool,
        cache_use_rpc: bool,
        memory_group_num = 1,

    ) -> None:
        if use_gpu:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions()
        rpc_backend_options.init_method = rpc_init_method
        dist.init_process_group(
            backend=backend,
            init_method=ccl_init_method,
            rank=rank, world_size=world_size,
        )
        self._gloo_group = dist.torch.distributed.new_group(backend='gloo')

        if use_gpu and rpc_gpu:
            print('in')
            device_maps = torch.zeros(world_size, dtype=torch.long, device=device)
            device_maps[rank] = local_rank
            dist.all_reduce(device_maps, op=dist.ReduceOp.SUM)
            for i, dev in enumerate(device_maps.tolist()):
                rpc_backend_options.set_device_map(
                    to=f"worker{i}",
                    device_map={local_rank: dev},
                )
        
        if use_rpc or cache_use_rpc :
            rpc.init_rpc(
                name=f"worker{rank}",
                rank=rank, world_size=world_size,
                #rpc_backend_options=rpc_backend_options,
            )
        self._use_rpc = use_rpc

        self._local_rank = local_rank
        self._compute_device = device
        self._hostname = socket.gethostname()
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        rank_to_host = [None] * self.world_size
        dist.all_gather_object(rank_to_host, (self.hostname, self.local_rank))
        self._rank_to_host: Tuple[Tuple[str, int], ...] = tuple(rank_to_host)
        self.memory_group_size = int(dist.get_world_size()/memory_group_num)
        self.memory_group= int(floor(dist.get_rank()/self.memory_group_size))
        self.memory_group_rank = int(dist.get_rank() % self.memory_group_size)
        print('local rank is {} world_size is {} memory group is {} memory rank is {} memory group size is {}\n'.format(self._local_rank,self.world_size,self.memory_group,self.memory_group_rank,self.memory_group_size))
        self.memory_group_num = memory_group_num
        print([i for i in range(self.memory_group*self.memory_group_size,(self.memory_group+1)*self.memory_group_size)])
        self.memory_gloo_group =  dist.torch.distributed.new_group(ranks=[i for i in range(self.memory_group*self.memory_group_size,(self.memory_group+1)*self.memory_group_size)],backend='gloo')
        self.memory_nccl_group =  dist.torch.distributed.new_group(ranks=[i for i in range(self.memory_group*self.memory_group_size,(self.memory_group+1)*self.memory_group_size)],backend='nccl')
        host_index = [h for h, _ in self.rank_to_host]
        host_index.sort()
        self._host_index: Dict[str, int] = {h:i for i, h in enumerate(host_index)}

        self.__temp_ag_remote_object: Optional[rpc.RRef] = None
    
    def shutdown(self):
        if self._use_rpc:
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
    def hostname(self) -> str:
        return self._hostname
    
    @property
    def rank_to_host(self):
        return self._rank_to_host
    
    @property
    def host_index(self):
        return self._host_index

    @property
    def device(self) -> torch.device:
        return self._compute_device
    @property
    def gloo_group(self):
        return self._gloo_group
    
    def get_default_group(self):
        # return dist.distributed_c10d._get_default_group()
        return dist.GroupMember.WORLD
    
    def get_default_store(self):
        return dist.distributed_c10d._get_default_store()

    def get_ranks_by_host(self, hostname: Optional[str] = None) -> Tuple[int,...]:
        if hostname is None:
            hostname = self.hostname
    
        ranks: List[int] = []
        for i, (h, r) in enumerate(self.rank_to_host):
            if h == hostname:
                ranks.append(i)
        ranks.sort()
        return tuple(ranks)
    
    def get_ranks_by_local(self, local_rank: Optional[int] = None) -> Tuple[int,...]:
        if local_rank is None:
            local_rank = self.local_rank
        
        ranks: List[Tuple[int, str]] = []
        for i, (h, r) in enumerate(self.rank_to_host):
            if r == local_rank:
                ranks.append((i, h))
        ranks.sort(key=lambda x: self.host_index[x[1]])
        return tuple(i for i, h in ranks)
    
    def get_hybrid_matrix(self) -> Tensor:
        hosts = sorted(self.host_index.items(), key=lambda x: x[1])
        matrix = []
        for h, _ in hosts:
            rs = self.get_ranks_by_host(h)
            matrix.append(rs)
        return torch.tensor(matrix, dtype=torch.long, device="cpu")
    
    def new_hybrid_subgroups(self,
        matrix: Optional[Tensor] = None,
        backend: Any = None,
    ) -> Tuple[Any, Any]:
        if matrix is None:
            matrix = self.get_hybrid_matrix()

        assert matrix.dim() == 2
        row_group = None
        col_group = None
        for row in matrix.tolist():
            if self.rank in row:
                row_group = dist.new_group(
                    row, backend=backend,
                    use_local_synchronization=True,
                )
                break
        for col in matrix.t().tolist():
            if self.rank in col:
                col_group = dist.new_group(
                    col, backend=backend,
                    use_local_synchronization=True,
                )
                break
        assert row_group is not None
        assert col_group is not None
        return row_group, col_group
    
    def get_worker_info(self, rank: Optional[int] = None) -> rpc.WorkerInfo:
        rank = dist.get_rank() if rank is None else rank
        return rpc.get_worker_info(f"worker{rank}")
    
    def remote_call(self, method, rref: rpc.RRef, *args, **kwargs):
        return rpc_remote_call(method, rref, *args, **kwargs)
    
    def remote_void_call(self, method, rref: rpc.RRef, *args, **kwargs):
        return rpc_remote_void_call(method, rref, *args, **kwargs)
    
    def remote_exec(self, method, rref: rpc.RRef, *args, **kwargs):
        return rpc_remote_exec(method, rref, *args, **kwargs)
    
    @contextmanager
    def use_stream(self, stream: torch.cuda.Stream, with_event: bool = True):
        event = torch.cuda.Event() if with_event else None
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            yield event
            if with_event:
                event.record()
    
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
