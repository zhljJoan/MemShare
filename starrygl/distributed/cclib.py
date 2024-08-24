from starrygl.distributed.context import DistributedContext
import torch
import torch.distributed as dist

from torch import Tensor
from typing import *

__all__ = [
    "all_to_all_v",
    "all_to_all_s",
    "BatchWork",
    "batch_send",
    "batch_recv",
]


class BatchWork:
    def __init__(self,
        works: Optional[List[Any]],
        buffer_tensor_list: Optional[List[Tuple[Tensor, Optional[Tensor]]]],
        step: int = 1,
    ) -> None:
        if works is None:
            self._step = None
            self._works = None
            self._buffer_tensor_list = None
        else:
            if buffer_tensor_list:
                assert len(works) // step == len(buffer_tensor_list)
            self._step = step
            self._works = works
            self._buffer_tensor_list = buffer_tensor_list
    
    def wait(self):
        if self._works is None:
            return
        
        for i, w in enumerate(self._works):
            if w is not None:
                w.wait()

            if (i + 1) % self._step != 0:
                continue

            if self._buffer_tensor_list:
                out, buf = self._buffer_tensor_list[i // self._step]
                if buf is not None:
                    out.copy_(buf)

        self._step = None
        self._works = None
        self._buffer_tensor_list = None


def all_to_all_v(
    output_tensor_list: List[Tensor],
    input_tensor_list: List[Tensor],
    group: Optional[Any] = None,
    async_op: bool = False,
):
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    assert len(output_tensor_list) == world_size
    assert len(input_tensor_list) == world_size

    backend = dist.get_backend(group)

    if backend == "nccl":
        work = dist.all_to_all(
            output_tensor_list=output_tensor_list,
            input_tensor_list=input_tensor_list,
            group=group,
            async_op=async_op,
        )
        return BatchWork([work], None) if async_op else None
    
    elif backend == "mpi":
        work = dist.all_to_all(
            output_tensor_list=output_tensor_list,
            input_tensor_list=input_tensor_list,
            group=group,
            async_op=async_op,
        )
        return BatchWork([work], None) if async_op else None
    
    else:
        assert backend == "gloo", f"backend must be nccl, mpi or gloo"

        p2p_op_works = []
        buffer_tensor_list = []
        for i in range(1, world_size):
            send_i = (rank + i) % world_size
            recv_i = (rank - i + world_size) % world_size

            send_t = input_tensor_list[send_i]
            recv_t = output_tensor_list[recv_i]

            if send_t.is_cuda:
                send_t = send_t.cpu()

            if recv_t.is_cuda:
                recv_b = torch.empty_like(recv_t, device="cpu")
                buffer_tensor_list.append((recv_t, recv_b))
            else:
                recv_b = recv_t
                buffer_tensor_list.append((recv_t, None))

            p2p_op_works.extend([
                dist.isend(send_t, send_i, group=group),
                dist.irecv(recv_b, recv_i, group=group),
            ])
        
        work = BatchWork(p2p_op_works, buffer_tensor_list, 2)
        output_tensor_list[rank].copy_(input_tensor_list[rank])

        if async_op:
            return work
        work.wait()

def all_to_all_s(
    output_tensor: Tensor,
    input_tensor: Tensor,
    output_rowptr: List[int],
    input_rowptr: List[int],
    group: Optional[Any] = None,
    async_op: bool = False,
):
    # rank = dist.get_rank(group)
    
    world_size = dist.get_world_size(group)
    assert len(output_rowptr) == len(input_rowptr)
    assert len(output_rowptr) == world_size + 1
    output_sizes = [t-s for s, t in zip(output_rowptr, output_rowptr[1:])]
    input_sizes = [t-s for s, t in zip(input_rowptr, input_rowptr[1:])]
    """
    return dist.all_to_all(output_tensor_list=,input_tensor_list=,group=group,async_op=True,)
    """
    return dist.all_to_all_single(
        output=output_tensor,
        input=input_tensor,
        output_split_sizes=output_sizes,
        input_split_sizes=input_sizes,
        group=group,
        async_op=async_op,
    )
    

def batch_send(
    *tensors: Tensor,
    dst: int,
    group: Any = None,
    async_op: bool = False,
):
    if len(tensors) == 0:
        return BatchWork(None, None)
    
    if group is None:
        group = dist.GroupMember.WORLD
    # tensors = tuple(t.data for t in tensors)
    backend = dist.get_backend(group)
    dst = dist.get_global_rank(group, dst)

    if async_op:
        works = []
        for t in tensors:
            if backend == "gloo" and t.is_cuda:
                t = t.cpu()
            works.append(dist.isend(t, dst=dst, group=group))
        return BatchWork(works, None)
    else:
        for t in tensors:
            if backend == "gloo" and t.is_cuda:
                t = t.cpu()
            dist.send(t, dst=dst, group=group)

def batch_recv(
    *tensors: Tensor,
    src: int,
    group: Any = None,
    async_op: bool = False,
):
    if len(tensors) == 0:
        return BatchWork(None, None)
    
    if group is None:
        group = dist.GroupMember.WORLD
    # tensors = tuple(t.data for t in tensors)
    backend = dist.get_backend(group)
    src = dist.get_global_rank(group, src)

    if async_op:
        works = []
        output_tensor_list = []
        for t in tensors:
            if backend == "gloo" and t.is_cuda:
                b = torch.empty_like(t, device="cpu")
                works.append(dist.irecv(b, src=src, group=group))
            else:
                b = None
                works.append(dist.irecv(t, src=src, group=group))
            output_tensor_list.append((t, b))
        return BatchWork(works, output_tensor_list, 1)
    else:
        for t in tensors:
            if backend == "gloo" and t.is_cuda:
                b = torch.empty_like(t, device="cpu")
                dist.recv(b, src=src, group=group)
                t.copy_(b)
            else:
                dist.recv(t, src=src, group=group)
