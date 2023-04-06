import torch

from torch import Tensor
from typing import *

EID: str = "__edata_eid_key__"
SID: str = "__edata_sid_key__"
DID: str = "__edata_did_key__"

class EData:
    def __init__(self, edge_size: int, p = None) -> None:
        if p is None:
            self.edge_size = edge_size
        else:
            assert edge_size is None
            self.edge_size = p.edge_size

        self.prev_data = p
        self.data: Dict[str, Tensor] = {}
    
    def __getitem__(self, name: str) -> Tensor:
        t = self.get(name)
        if t is None:
            if name in (EID, SID, DID):
                raise ValueError(f"not found EID in data")
            else:
                raise ValueError(f"not found '{name}' in data")
        return t
    
    def __setitem__(self, name: str, tensor: Tensor):
        if not isinstance(tensor, Tensor):
            raise ValueError(f"the second parameter's type must be Tensor")
        if name in (SID, DID):
            raise ValueError(f"please try EID")
        if name == EID:
            if tensor.dim() == 2 \
                and tensor.size(0) == 2 \
                and tensor.size(1) == self.edge_size:
                self.data[name] = tensor
            else:
                raise ValueError(f"EID's shape must be 2 x edge_size")
        else:
            if tensor.size(0) == self.edge_size:
                self.data[name] = tensor
            else:
                raise ValueError(f"tensor's shape must match the edge_size")
    
    def __delitem__(self, name: str) -> None:
        self.pop(name)
    
    def _get_impl(self, name: str) -> Optional[Tensor]:
        p, t = self, None
        while p is not None:
            t = p.data.get(name)
            if t is not None:
                break
            p = self.prev_data
        return t
    
    def get(self, name: str) -> Optional[Tensor]:
        if name in (EID, SID, DID):
            t = self._get_impl(EID)
            if t is None:
                raise ValueError(f"no edge info in data")
            assert t.dim() == 2 and t.size(0) == 2
            if name == SID:
                t = t[0]
            elif name == DID:
                t = t[1]
            return t
        else:
            return self._get_impl(name)
    
    def pop(self, name: str) -> Tensor:
        if name in (SID, DID):
            raise ValueError(f"please try EID")
        if name == EID:
            return self.data.pop(EID)
        else:
            return self.data.pop(name)
    