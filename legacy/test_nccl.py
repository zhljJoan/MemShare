import torch
import torch.distributed as dist

from torch import Tensor
from typing import *

from starrygl.parallel import init_process_group
from starrygl.utils.printer import main_print, sync_print

from multiprocessing.pool import ThreadPool

num_threads = 1
num_tasks = 5
pool = ThreadPool(num_threads)

if __name__ == "__main__":
    import os
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = init_process_group(backend="nccl")

    group = dist.new_group()
    stream = torch.cuda.Stream(device)

    handles = []
    for i in range(num_tasks):
        x = torch.ones(10, dtype=torch.float32, device=device)
        def run(x):
            # stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                group.allreduce(x)
            return x
        stream.wait_stream(torch.cuda.current_stream())
        handles.append(pool.apply_async(run, (x,)))
    
    x = torch.ones(3, dtype=torch.float32, device=device)
    dist.all_reduce(x)
    sync_print(x.tolist())

    for i in range(num_tasks):
        main_print("="*64)
        x = handles[i].get()
        assert (handles[i].get() == x).all()
        sync_print(x.tolist())
