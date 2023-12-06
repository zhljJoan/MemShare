import torch
import torch.distributed as dist

from torch import Tensor
from typing import *


def all_to_all(
    output_tensor_list: List[Tensor],
    input_tensor_list: List[Tensor],
    group: Optional[Any] = None,
):
    assert len(output_tensor_list) == len(input_tensor_list)

    if group is None:
        group = dist.distributed_c10d._get_default_group()
    backend = dist.get_backend(group)

    if backend == "nccl":
        dist.all_to_all(
            output_tensor_list=output_tensor_list,
            input_tensor_list=input_tensor_list,
            group=group,
        )
    elif backend == "mpi":
        dist.all_to_all(
            output_tensor_list=output_tensor_list,
            input_tensor_list=input_tensor_list,
            group=group,
        )
    else:
        assert backend == "gloo", f"backend must be nccl, mpi or gloo"
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        p2p_op_list: List[dist.P2POp] = []
        for i in range(1, world_size):
            send_i = (rank + i) % world_size
            recv_i = (rank - i + world_size) % world_size

            p2p_op_list.extend([
                dist.P2POp(dist.isend, input_tensor_list[send_i], send_i, group=group),
                dist.P2POp(dist.irecv, output_tensor_list[recv_i], recv_i, group=group),
            ])
        dist.batch_isend_irecv(p2p_op_list)
        output_tensor_list[rank][:] = input_tensor_list[rank]
        