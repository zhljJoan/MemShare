import torch

from contextlib import contextmanager

from torch import Tensor
from typing import *



class RouteContext:
    def __init__(self) -> None:
        self._futs: List[torch.futures.Future] = []
    

    def synchronize(self):
        for fut in self._futs:
            fut.wait()
        self._futs = []
    
    def add_futures(self, *futs):
        for fut in futs:
            assert isinstance(fut, torch.futures.Future)
            self._futs.append(fut)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            raise exc_type(exc_value)
        self.synchronize()
