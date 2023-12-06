import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc

import os

from torch import Tensor
from typing import *

from starrygl.parallel import init_process_group, all_gather_remote_objects
from starrygl.utils.printer import main_print, sync_print



class A:
    def __init__(self, device) -> None:
        self.id = rpc.get_worker_info().id
        self.data = torch.tensor([self.id], device=device)
    
    @staticmethod
    def get_data(rref: rpc.RRef) -> Tensor:
        self: A = rref.local_value()
        return self.data
    
    def set_remote_a(self, *rrefs: rpc.RRef):
        self.rrefs = tuple(rrefs)
    
    def display(self):
        sync_print(f"id = {self.id}")

    def gather_data(self, device) -> List[Tensor]:
        world_size = dist.get_world_size()
        futs: List[torch.futures.Future] = []
        for i in range(world_size):
            f = rpc.rpc_async(self.rrefs[i].owner(), A.get_data, args=(self.rrefs[i],))
            futs.append(f)
        
        results: List[Tensor] = []
        for f in futs:
            f.wait()
            results.append(f.value())
        return results


if __name__ == "__main__":
    device = init_process_group("nccl")

    a = A(device)
    a.set_remote_a(*all_gather_remote_objects(a))
    a.display()

    for t in a.gather_data(device):
        sync_print(t, t.device)

    rpc.shutdown()