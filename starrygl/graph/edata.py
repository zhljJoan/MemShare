import torch

from torch import Tensor
from typing import *



class EData:
    def __init__(self,
        edge_size: Optional[int] = None,
        p = None,
    ) -> None:
        if p is None:
            assert edge_size is not None
            self.edge_size = edge_size
        else:
            assert edge_size is None
            self.edge_size = p.edge_size

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
        
        if tensor.size(0) == self.edge_size:
            self.data[name] = tensor
        else:
            raise ValueError(f"tensor's shape must match the edge_size")
    
    def __delitem__(self, name: str) -> None:
        self.pop(name)
    
    def get(self, name: str) -> Optional[Tensor]:
        p, t = self, None
        while p is not None:
            t = p.data.get(name)
            if t is not None:
                break
            p = p.prev_data
        return t
       
    def pop(self, name: str) -> Tensor:
        return self.data.pop(name)
    
    def permute_(self, perm: Tensor):
        p = self
        while p is not None:
            for key in list(p.data.keys()):
                val = p.data.get(key)
                p.data[key] = val[perm]
            p = p.prev_data
            