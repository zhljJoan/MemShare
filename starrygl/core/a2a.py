import torch
import torch.distributed as dist

from torch import Tensor
from typing import *

def with_nccl() -> bool:
    return dist.get_backend() == "nccl"

def with_gloo() -> bool:
    return dist.get_backend() == "gloo"

class Works:
    def __init__(self, *works) -> None:
        self._works = list(works)
        self._waited = False

    def wait(self) -> None:
        assert not self._waited
        for w in self._works:
            if w is None:
                continue
            w.wait()
        self._waited = True
    
    def push(self, *works) -> None:
        assert not self._waited
        self._works.extend(works)

def all_to_all(
    output_tensor_list: List[Tensor],
    input_tensor_list: List[Tensor],
) -> Works:
    assert len(output_tensor_list) == len(input_tensor_list)
    if with_nccl():
        work = dist.all_to_all(
            output_tensor_list=output_tensor_list,
            input_tensor_list=input_tensor_list,
            async_op=True,
        )
        return Works(work)
    elif with_gloo():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        works = Works()
        for i in range(1, world_size):
            send_i = (rank + i) % world_size
            recv_i = (rank - i + world_size) % world_size

            send_w = dist.isend(input_tensor_list[send_i], send_i)
            recv_w = dist.irecv(output_tensor_list[recv_i], recv_i)

            works.push(recv_w, send_w)
        output_tensor_list[rank][:] = input_tensor_list[rank]
        return works
    else:
        backend = dist.get_backend()
        raise RuntimeError(f"unsupported backend: {backend}")
