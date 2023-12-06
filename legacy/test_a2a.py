import torch
import torch.distributed as dist

from starrygl.parallel import init_process_group
from typing import *

if __name__ == "__main__":
    device = init_process_group(backend="gloo")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    assert world_size == 4
    
    xs = [torch.randn(100+i*10, device=device) for i in range(world_size)]
    send_sizes = torch.tensor([x.size(0) for x in xs], dtype=torch.long, device=device)
    recv_sizes = torch.zeros_like(send_sizes)
    dist.all_to_all_single(recv_sizes, send_sizes)
    
    