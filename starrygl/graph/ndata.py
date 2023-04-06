import torch

from torch import Tensor
from typing import *

NID: str = "__ndata_nid_key__"

class NData:
    def __init__(self, src_size: int, dst_size: int, p = None) -> None:
        if p is None:
            self.src_size = src_size
            self.dst_size = dst_size
        else:
            assert src_size is None
            assert dst_size is None
            self.src_size = p.src_size
            self.dst_size = p.dst_size

        self.prev_data = p
        self.data: Dict[str, Tensor] = {}
    
    def __getitem__(self, name: str) -> Tensor:
        t = self.get(name)
        if t is None:
            raise ValueError(f"not found '{name}' in data")
        return t
    
    def __setitem__(self, name: str, tensor: Tensor):
        if not isinstance(tensor, Tensor):
            raise ValueError(f"the second parameter's type must be Tensor")
        if tensor.size(0) == self.src_size:
            self.data[name] = tensor
        elif tensor.size(0) == self.dst_size:
            self.data[name] = tensor
        else:
            raise ValueError(f"tensor's shape must match the src_size or dst_size")
    
    def __delitem__(self, name: str) -> None:
        self.pop(name)
    
    def get(self, name: str) -> Optional[Tensor]:
        p, t = self, None
        while p is not None:
            t = p.data.get(name)
            if t is not None:
                break
            p = self.prev_data
        return t
    
    def pop(self, name: str) -> Tensor:
        return self.data.pop(name)
    
    def get_type(self, name: str):
        t = self.get(name)
        if t.size(0) == self.src_size:
            return "src"
        elif t.size(0) == self.dst_size:
            return "dst"
        else:
            raise RuntimeError