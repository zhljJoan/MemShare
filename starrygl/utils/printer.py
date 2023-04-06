import torch.distributed as dist

def sync_print(*args, **kwargs):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    for i in range(world_size):
        if i == rank:
            print(f"rank {rank}:", *args, **kwargs)
        dist.barrier()

def main_print(*args, **kwargs):
    rank = dist.get_rank()
    if rank == 0:
        print(*args, **kwargs)
    dist.barrier()