import torch
import torch.nn as nn
import torch.distributed as dist

from torch import Tensor
from typing import *

from starrygl.core.gather import Gather
from starrygl.parallel import init_process_group
from starrygl.parallel.sync_bn import SyncBatchNorm
from starrygl.graph import DistGraph, EID, SID, DID
from starrygl.utils.printer import main_print, sync_print

if __name__ == "__main__":
    device = init_process_group(backend="nccl")
    bn = SyncBatchNorm(100).to(device)
    bn.train()

    x = torch.randn(10, 100, dtype=torch.float64).to(device).requires_grad_()
    sync_print(x)
    from torch.autograd import gradcheck
    sync_print(gradcheck(lambda x: bn(x), x))
    # sync_print(bn(x))
