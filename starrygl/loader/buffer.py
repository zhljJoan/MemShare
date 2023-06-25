import torch
import torch.nn as nn

from torch import Tensor
from typing import *



class TensorBuffers(nn.Module):
    def __init__(self, src_size: int, channels: Tuple[Union[int, Tuple[int]]]) -> None:
        super().__init__()

        self.src_size = src_size
        self.num_layers = len(channels)
        for i, s in enumerate(channels):
            s = (s,) if isinstance(s, int) else s
            self.register_buffer(f"data_{i}", torch.zeros(src_size, *s), persistent=False)
    
    def get(self, i: int, index: Optional[Tensor]) -> Tensor:
        i = self._idx(i)
        if index is None:
            return self.get_buffer(f"data_{i}")
        else:
            return self.get_buffer(f"data_{i}")[index]
    
    def set(self, i: int, index: Optional[Tensor], value: Tensor):
        i = self._idx(i)
        if index is None:
            self.get_buffer(f"data_{i}")[:,...] = value
        else:
            self.get_buffer(f"data_{i}")[index] = value
    
    def add(self, i: int, index: Optional[Tensor], value: Tensor):
        i = self._idx(i)
        if index is None:
            self.get_buffer(f"data_{i}")[:,...] += value
        else:
            self.get_buffer(f"data_{i}")[index] += value
    
    def zero_grad(self):
        for name, grad in self.named_buffers("data_", recurse=False):
            grad.zero_()
    
    def _idx(self, i: int) -> int:
        assert -self.num_layers < i and i < self.num_layers
        return (self.num_layers + i) % self.num_layers
    