import torch
import starrygl

from torch import Tensor
from enum import Enum
from typing import *

__all__ = [
    "uvm_empty",
    "uvm_share",
    "uvm_advise",
    "uvm_prefetch",
    "cudaMemoryAdvise",
]

def uvm_empty(*sizes: int, dtype: torch.dtype, device: Any):
    sizes = torch.Size(sizes)
    device = torch.device(device)

    assert device.type == "cuda" \
        and device.index is not None, "device must be cuda:x"
    
    size_bytes = torch.Size(sizes).numel() * dtype.itemsize

    # default strides
    dims = len(sizes)
    strides = [1] * dims
    for i in range(1, dims):
        strides[dims-i-1] = strides[dims-i] * sizes[dims-i]
    strides = torch.Size(strides)

    storage = starrygl.ops.uvm_storage_new(size_bytes, device.index)
    return torch.empty(0, dtype=dtype, device=device).set_(storage, 0, sizes, strides)

def uvm_share(x: Tensor, device: Any):
    device = torch.device(device)
    if device.type == "cpu":
        storage = starrygl.ops.uvm_storage_to_cpu(x.untyped_storage())
    else:
        assert device.type == "cuda" \
            and device.index is not None, "device must be cuda:x or cpu"
        storage = starrygl.ops.uvm_storage_to_cuda(x.untyped_storage(), device.index)

    return torch.empty(0, dtype=x.dtype, device=device) \
        .set_(storage, x.storage_offset(), x.size(), x.stride())

class cudaMemoryAdvise(Enum):
    cudaMemAdviseSetAccessedBy = starrygl.ops.cudaMemoryAdvise.cudaMemAdviseSetAccessedBy
    cudaMemAdviseUnsetAccessedBy = starrygl.ops.cudaMemoryAdvise.cudaMemAdviseUnsetAccessedBy
    cudaMemAdviseSetPreferredLocation = starrygl.ops.cudaMemoryAdvise.cudaMemAdviseSetPreferredLocation
    cudaMemAdviseUnsetPreferredLocation = starrygl.ops.cudaMemoryAdvise.cudaMemAdviseUnsetPreferredLocation
    cudaMemAdviseSetReadMostly = starrygl.ops.cudaMemoryAdvise.cudaMemAdviseSetReadMostly
    cudaMemAdviseUnsetReadMostly = starrygl.ops.cudaMemoryAdvise.cudaMemAdviseUnsetReadMostly

def uvm_advise(x: Tensor, advise: cudaMemoryAdvise):
    assert isinstance(advise, cudaMemoryAdvise)
    advise = starrygl.ops.cudaMemoryAdvise(advise.value)
    starrygl.ops.uvm_storage_advise(x.untyped_storage(), advise)

def uvm_prefetch(x: Tensor):
    starrygl.ops.uvm_storage_prefetch(x.untyped_storage())