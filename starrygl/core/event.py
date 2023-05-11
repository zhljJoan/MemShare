import torch

import time

from torch import Tensor
from typing import *



class Event:
    def __init__(self, use_cuda: bool = True) -> None:
        self._use_cuda = use_cuda
        if use_cuda:
            self._event = torch.cuda.Event(enable_timing=True)
        else:
            self._time: Optional[float] = None
    
    def record(self):
        if self._use_cuda:
            self._event.record()
        else:
            self._time = time.time()
    
    def wait(self, stream: Optional[Tensor] = None):
        if self._use_cuda:
            return
        self._event.wait(stream)
    
    def elapsed_time(self, other) -> float:
        if self._use_cuda:
            return self._event.elapsed_time(other._event)
        else:
            return (other._time - self._time) * 1000.0
