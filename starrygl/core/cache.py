import torch

from torch import Tensor
from typing import *

class EmbeddingCache:
    def __init__(self) -> None:
        self.values: Optional[Tensor] = None
        self.active: Optional[Tensor] = None
    
    def get_values(self) -> Optional[Tensor]:
        return self.values
    
    def get_active(self) -> Optional[Tensor]:
        return self.active
    
    def get(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        return self.get_values(), self.get_active()
    
    def set(self, values: Tensor, active: Optional[Tensor] = None):
        values = values.detach()
        active = active.detach()
        if active is None:
            if self.values is None:
                self.values = values.clone()
            else:
                self.values[:] = values
            self.active = None
        else:
            if self.values is None:
                self.values = torch.zeros(
                    size=active.shape[:1] + values.shape[1:],
                    dtype=values.dtype,
                    device=values.device,
                )

            if values.size(0) == active.size(0):
                self.values[active] = values[active]
            else:
                self.values[active] = values

            if self.active is None:
                self.active = active.clone()
            else:
                self.active[:] = active

    def fuse(self, values: Tensor, active: Optional[Tensor] = None) -> Tensor:
        if active is None:
            return values
        else:
            if self.values is None:
                x = torch.zeros(
                    size=active.shape[:1] + values.shape[1:],
                    dtype=values.dtype,
                    device=values.device,
                )
            else:
                x = self.values.clone()
            
            if values.size(0) == active.size(0):
                x[active] = values[active]
            else:
                x[active] = values
            return x
        