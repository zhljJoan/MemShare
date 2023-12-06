import torch
import torch.distributed as dist

from torch import Tensor
from typing import *


class BatchWork:
    def __init__(self, works, buffer_tensor_list) -> None:
        if isinstance(works, (list, tuple)):
            assert len(works) // 2 == len(buffer_tensor_list)
            self._works = works
            self._buffer_tensor_list = buffer_tensor_list
        else:
            assert self._buffer_tensor_list is None
            self._works = works
            self._buffer_tensor_list = None
    
    def wait(self):
        if self._works is None:
            return
        if isinstance(self._works, (list, tuple)):
            for i, w in enumerate(self._works):
                w.wait()
                if i % 2 == 0:
                    continue
                out, buf = self._buffer_tensor_list[i // 2]
                if buf is not None:
                    out.copy_(buf)
        else:
            self._works.wait()
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

    if group is None:
        group = dist.distributed_c10d._get_default_group()
    backend = dist.get_backend(group)

    if backend == "nccl":
        work = dist.all_to_all(
            output_tensor_list=output_tensor_list,
            input_tensor_list=input_tensor_list,
            group=group,
            async_op=async_op,
        )
        return BatchWork(work, None) if async_op else None
    
    elif backend == "mpi":
        work = dist.all_to_all(
            output_tensor_list=output_tensor_list,
            input_tensor_list=input_tensor_list,
            group=group,
            async_op=async_op,
        )
        return BatchWork(work, None) if async_op else None
    
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
        
        work = BatchWork(p2p_op_works, buffer_tensor_list)
        output_tensor_list[rank].copy_(input_tensor_list[rank])

        if async_op:
            return work
        work.wait()

    # else:
    #     assert backend == "gloo", f"backend must be nccl, mpi or gloo"

    #     rank = dist.get_rank(group)
    #     world_size = dist.get_world_size(group)

    #     p2p_op_list: List[dist.P2POp] = []
    #     for i in range(1, world_size):
    #         send_i = (rank + i) % world_size
    #         recv_i = (rank - i + world_size) % world_size

    #         p2p_op_list.extend([
    #             dist.P2POp(dist.isend, input_tensor_list[send_i], send_i, group=group),
    #             dist.P2POp(dist.irecv, output_tensor_list[recv_i], recv_i, group=group),
    #         ])
    #     dist.batch_isend_irecv(p2p_op_list)
    #     output_tensor_list[rank][:] = input_tensor_list[rank]