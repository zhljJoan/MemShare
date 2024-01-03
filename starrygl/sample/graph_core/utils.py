from starrygl.distributed.context import DistributedContext
from typing import *
import torch.distributed as dist
import torch

from starrygl.distributed.utils import DistIndex
def build_mapper(nids):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dst_len = nids.size(0)
    ikw = dict(dtype=torch.long, device=nids.device)
    num_nodes = torch.zeros(1, **ikw)
    num_nodes[0] = dst_len
    dist.all_reduce(num_nodes, op=dist.ReduceOp.SUM)
    all_ids: List[torch.Tensor] = [None] * world_size
    dist.all_gather_object(all_ids,nids)
    part_mp = torch.empty(num_nodes,**ikw)
    ind_mp = torch.empty(num_nodes,**ikw)
    for i in range(world_size):
        iid = all_ids[i]
        part_mp[iid] = i
        ind_mp[iid] = torch.arange(all_ids[i].shape[0],**ikw)
    return DistIndex(ind_mp,part_mp)


def get_validate_graph(self,graph):
    pass