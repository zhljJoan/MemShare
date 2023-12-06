import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.rpc as rpc

import os
from torch import Tensor
from typing import *

from .context import DistributedContext
from .utils import (
    TensorAccessor,
    DistributedTensor,
    DistIndex,
)


def init_distributed_context(backend: str = "gloo") -> DistributedContext:
    return DistributedContext.init(backend)
