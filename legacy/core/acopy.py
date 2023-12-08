# import torch

# from torch import Tensor
# from typing import *

# from multiprocessing.pool import ThreadPool
# from .event import Event


# class AsyncCopyWorkBase:
#     def __init__(self) -> None:
#         self._events: Optional[Tuple[Event, Event]] = None

#     def wait(self):
#         raise NotImplementedError
    
#     def get(self) -> Tensor:
#         raise NotImplementedError
    
#     def has_events(self) -> bool:
#         return self._events is not None
    
#     def set_events(self, start, end):
#         self._events = (start, end)
#         return self
    
#     def time_used(self) -> float:
#         if self._events is None:
#             raise RuntimeError("not found events")
#         start, end = self._events
#         return start.elapsed_time(end)

# class AsyncPushWork(AsyncCopyWorkBase):
#     def __init__(self, data: Tensor, index: Tensor, values: Tensor) -> None:
#         super().__init__()
#         assert data.device == index.device
#         self.set_events(
#             Event(use_cuda=index.is_cuda),
#             Event(use_cuda=index.is_cuda),
#         )
#         self._events[0].record()
#         data.index_copy_(0, index, values)
#         self._events[1].record()
    
#     def wait(self):
#         pass
    
#     def get(self):
#         pass

# class AsyncPullWork(AsyncCopyWorkBase):
#     def __init__(self, data: Tensor, index: Tensor) -> None:
#         super().__init__()
#         assert data.device == index.device
#         self.set_events(
#             Event(use_cuda=index.is_cuda),
#             Event(use_cuda=index.is_cuda),
#         )
#         self._events[0].record()
#         self._val = data.index_select(0, index)
#         self._events[1].record()
    
#     def wait(self):
#         pass

#     def get(self) -> Tensor:
#         return self._val

# class AsyncOffloadWork(AsyncCopyWorkBase):
#     def __init__(self, handle) -> None:
#         super().__init__()
#         self._handle = handle
    
#     def wait(self):
#         if self._events is not None:
#             self._events[1].wait(torch.cuda.current_stream())
#         self._handle.wait()
    
#     def get(self) -> Tensor:
#         if self._events is not None:
#             self._events[1].wait(torch.cuda.current_stream())
#         return self._handle.get()

# class AsyncCopyExecutor:
#     def __init__(self) -> None:
#         self._stream = torch.cuda.Stream()
#         self._executor = ThreadPool(processes=1)
    
#     @torch.no_grad()
#     def async_pull(self, data: Tensor, index: Tensor) -> AsyncCopyWorkBase:
#         # 这边的代码最好全部放到一个独立的线程里，一方面方便计时，也便于异步
#         if data.device != index.device:
#             assert not data.is_cuda
#             assert index.is_cuda

#             start = Event(index.is_cuda)
#             end = Event(index.is_cuda)
#             stream: Optional[torch.cuda.Stream] = self._stream
#             def run():
#                 start.wait(stream)
#                 with torch.cuda.stream(stream):
#                     idx = index.to(data.device)
#                     dst = torch.zeros(
#                         size=(index.size(0),) + data.shape[1:],
#                         dtype=data.dtype,
#                         device=index.device,
#                     )
#                     dst.copy_(data[idx])
#                     end.record()
#                     return dst
#             start.record()
#             handle = self._executor.apply_async(run)
#             return AsyncOffloadWork(handle).set_events(start, end)
#         else:
#             # data and index in the same device
#             return AsyncPullWork(data, index)
    
#     @torch.no_grad()
#     def async_push(self, data: Tensor, index: Tensor, values: Tensor) -> AsyncCopyWorkBase:
#         assert index.device == values.device
#         if data.device != index.device:
#             assert not data.is_cuda
#             assert index.is_cuda
#             start = Event(index.is_cuda)
#             end = Event(index.is_cuda)
#             stream: Optional[torch.cuda.Stream] = self._stream
#             def run():
#                 start.wait(stream)
#                 with torch.cuda.stream(stream):
#                     idx = index.to(data.device)
#                     val = values.to(data.device)
#                     data[idx] = val
#                     end.record()
#             start.record()
#             handle = self._executor.apply_async(run)
#             return AsyncOffloadWork(handle).set_events(start, end)
#         else:
#             return AsyncPushWork(data, index, values)


# _THREAD_EXEC: Optional[AsyncCopyExecutor] = None
# def get_executor() -> AsyncCopyExecutor:
#     global _THREAD_EXEC
#     if _THREAD_EXEC is None:
#         _THREAD_EXEC = AsyncCopyExecutor()
#     return _THREAD_EXEC
