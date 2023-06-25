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


# class Lache:
#     def __init__(self,
#         cache_size: int,
#         data: Tensor,
#     ) -> None:
#         assert not data.is_cuda, "data must be in CPU Memory"

#         cache_size = (cache_size,) + data.shape[1:]
#         self.fdata = data
#         self.cache = torch.zeros(cache_size, dtype=data.dtype)

#         self.no_idx: int = (2**62-1)*2+1
#         self.cached_idx = torch.empty(cache_size, dtype=torch.long).fill_(self.no_idx)
#         self.read_count = torch.zeros(data.size(0), dtype=torch.long)
    
#     def to(self, device):
#         self.cache = self.cache.to(device)
#         self.cached_idx = self.cached_idx.to(device)
#         self.read_count = self.read_count.to(device)
#         return self
    
#     def _push_impl(self, value: Tensor, index: Tensor):
#         # self.read_count[index] += 1
#         imp = torch.zeros_like(self.read_count)


#     def _pull_impl(self, index: Tensor):
#         assert index.device == self.cache.device

#         self.read_count[index] += 1

#         s = index.shape[:1] + self.cache.shape[1:]
#         x = torch.empty(s, dtype=self.cache.dtype, device=self.cache.device)

#         cache_mask = torch.zeros_like(self.read_count, dtype=torch.bool).index_fill_(0, self.cached_idx, 1)[index]
#         cache_index = index[cache_mask]
#         x[cache_mask] = 
#         no_cache_index = index[~cache_mask]
#         rt_index = index[~lc_mask].to(self.fdata.device)
#         rt_data = self.fdata[rt_index].to(index.device)


# _THREAD_EXEC: Optional[AsyncCopyExecutor] = None
# def get_executor() -> AsyncCopyExecutor:
#     global _THREAD_EXEC
#     if _THREAD_EXEC is None:
#         _THREAD_EXEC = AsyncCopyExecutor()
#     return _THREAD_EXEC
