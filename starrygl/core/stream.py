import torch

from torch import Tensor
from contextlib import contextmanager
from typing import *

__all__ = [
    "ABCStream",
    "ABCEvent",
    "phony_tensor",
    "new_stream",
    "current_stream",
    "default_stream",
    "use_stream",
    "use_device",
    "wait_stream",
    "wait_event",
    "record_stream",
]

class CPUStreamType:
    def __init__(self) -> None:
        self._device = torch.device("cpu")
    
    @property
    def device(self):
        return self._device
    
    def __call__(self):
        return self

class CPUEventType:
    def __init__(self) -> None:
        self._device = torch.device("cpu")
    
    @property
    def device(self):
        return self._device
    
    def __call__(self):
        return self

    
CPUStream = CPUStreamType()
ABCStream = Union[torch.cuda.Stream, CPUStreamType]

CPUEvent = CPUEventType()
ABCEvent = Union[torch.cuda.Event, CPUEventType]

def new_stream(device: Any) -> ABCStream:
    device = torch.device(device)
    if device.type != "cuda":
        return CPUStream()
    return torch.cuda.Stream(device)

_phonies: Dict[Tuple[torch.device, bool], Tensor] = {}
def phony_tensor(device: Any, requires_grad: bool = True):
    device = torch.device(device)
    key = (device, requires_grad)
    if key not in _phonies:
        with use_stream(default_stream(device)):
            _phonies[key] = torch.empty(
                0, device=device,
                requires_grad=requires_grad,
            )
    return _phonies[key]

def current_stream(device: Any) -> ABCStream:
    device = torch.device(device)
    if device.type != "cuda":
        return CPUStream()
    return torch.cuda.current_stream(device)

def default_stream(device: Any) -> ABCStream:
    device = torch.device(device)
    if device.type != "cuda":
        return CPUStream()
    return torch.cuda.default_stream(device)

@contextmanager
def use_stream(stream: ABCStream, fence_event: bool = False):
    if isinstance(stream, CPUStreamType):
        if fence_event:
            event = CPUEvent()
            yield event
        else:
            yield
        return
    with torch.cuda.stream(stream):
        if fence_event:
            event = torch.cuda.Event()
            yield event
            event.record()
        else:
            yield

@contextmanager
def use_device(device: Any):
    device = torch.device(device)
    if device.type != "cuda":
        yield
        return    
    with torch.cuda.device(device):
        yield

def wait_stream(source: ABCStream, target: ABCStream):
    if isinstance(target, CPUStreamType):
        return
    if isinstance(source, CPUStreamType):
        target.synchronize()
    else:
        source.wait_stream(target)

def wait_event(source: ABCStream, target: ABCEvent):
    if isinstance(target, CPUEventType):
        return
    if isinstance(source, CPUStreamType):
        target.synchronize()
    else:
        source.wait_event(target)

def record_stream(tensor: Tensor, stream: ABCStream):
    if isinstance(stream, CPUStreamType):
        return
    storage = tensor.untyped_storage()
    tensor = tensor.new_empty(0).set_(storage)
    tensor.record_stream(stream)
