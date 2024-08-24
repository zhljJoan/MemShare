from starrygl.distributed.context import DistributedContext
from typing import *
import torch.distributed as dist
import torch

from starrygl.distributed.utils import DistIndex, DistributedTensor
def build_mapper(nids):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dst_len = nids.size(0)
    ikw = dict(dtype=torch.long, device=torch.device('cpu'))
    num_nodes = torch.zeros(1, **ikw)
    num_nodes[0] = dst_len
    _ctx = DistributedContext.get_default_context()
    dist.all_reduce(num_nodes, op=dist.ReduceOp.SUM,group=_ctx.gloo_group)
    all_ids: List[torch.Tensor] = [None] * world_size
    dist.all_gather_object(all_ids,nids,group=_ctx.gloo_group)
    part_mp = torch.empty(num_nodes,**ikw)
    ind_mp = torch.empty(num_nodes,**ikw)
    for i in range(world_size):
        iid = all_ids[i]
        part_mp[iid] = i
        ind_mp[iid] = torch.arange(all_ids[i].shape[0],**ikw)
    return DistIndex(ind_mp,part_mp)

def _get_pin(cache: dict, layer: int, rows: int, dims: List[int]) -> torch.Tensor:
    """
    Creates/reshapes pinned buffer and returns it.
    :param dict cache: The dictionary of the buffer with layer numbers as keys.
    :param int layer: Which layer pinned data belongs to.
    :param int rows: Number of rows of pinned data.
    :param List[int] dims: Size of each pinned data.
    :return: Pinned buffer.
    :rtype: Tensor
    """
    if layer not in cache:
        shape = tuple((rows, *dims))
        pin = torch.zeros(shape, pin_memory=True)
        cache[layer] = pin
        return pin
    pin = cache[layer]
    if pin.shape[0] < rows or pin.shape[1:] != dims:
        shape = tuple((rows, *dims))
        pin.resize_(shape)
    return pin[:rows]


