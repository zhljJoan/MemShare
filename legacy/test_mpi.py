import torch
import torch.distributed as dist

from torch import Tensor
from typing import *

from starrygl.parallel import init_process_group
from starrygl.utils.printer import main_print, sync_print

if __name__ == "__main__":
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = init_process_group(backend="mpi")
    sync_print(dist.get_backend())