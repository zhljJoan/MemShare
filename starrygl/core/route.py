import torch
import torch.autograd as autograd
import torch.distributed as dist

from multiprocessing.pool import ThreadPool

from torch import Tensor
from typing import *

from contextlib import contextmanager
from .a2a import all_to_all, Works, with_nccl
from .event import Event



# class GatherWork:
#     def __init__(self,
#         values_buffer: Tensor,
#         active_buffer: Optional[Tensor],
#         recv_val_idx: List[Tensor],
#         recv_val_dat: List[Tensor],
#         works_list: List[Works],
#     ) -> None:
#         self._waited = False
#         self._values_buffer = values_buffer
#         self._active_buffer = active_buffer
#         self._recv_val_idx = recv_val_idx
#         self._recv_val_dat = recv_val_dat
#         self._works_list = works_list
        
#         self._cached_idx: Optional[Tensor] = None
    
#     def _wait(self) -> None:
#         assert not self._waited
#         for w in self._works_list:
#             if w is None:
#                 continue
#             w.wait()
        
#         if self._active_buffer is None:
#             for idx, dat in zip(self._recv_val_idx, self._recv_val_dat):
#                 self._values_buffer[idx] += dat
#         else:
#             for idx, dat in zip(self._recv_val_idx, self._recv_val_dat):
#                 self._values_buffer[idx] += dat
#                 self._active_buffer[idx] = True
#             self._active_buffer = torch.where(self._active_buffer)[0]
        
#         self._works_list = []
#         self._recv_val_dat = []
#         self._recv_val_idx = []
#         self._waited = True

#     def get_val(self) -> Tensor:
#         if not self._waited:
#             self._wait()
#         return self._values_buffer
    
#     def get_idx(self) -> Optional[Tensor]:
#         if not self._waited:
#             self._wait()
#         return self._active_buffer
    
#     def get(self) -> Tuple[Tensor, Optional[Tensor]]:
#         return self.get_val(), self.get_idx()

class RouteWorkBase:
    def __init__(self) -> None:
        self._events: Optional[Tuple[Event, Event]] = None

    def wait(self) -> None:
        raise NotImplementedError
    
    def get(self) -> Tuple[Tensor, Optional[Tensor]]:
        raise NotImplementedError
    
    def has_events(self) -> bool:
        return self._events is not None
    
    def set_events(self, start, end):
        self._events = (start, end)
        return self
    
    def time_used(self) -> float:
        if self._events is None:
            raise RuntimeError("not found events")
        start, end = self._events
        return start.elapsed_time(end)

class RouteWork(RouteWorkBase):
    def __init__(self,
        buf: Union[Tensor, int],
        recv_val_idx: List[Tensor],
        recv_val_dat: List[Tensor],
        works_list: List[Works],
    ) -> None:
        super().__init__()
        assert len(recv_val_idx) != 0
        assert len(recv_val_idx) == len(recv_val_dat)

        self._buf = buf
        self._recv_val_idx = recv_val_idx
        self._recv_val_dat = recv_val_dat
        self._works_list = works_list
        
        self._val: Optional[Tensor] = None
        self._idx: Optional[Tensor] = None
    
    def wait(self) -> None:
        if self._val is not None:
            return
        
        for w in self._works_list:
            if w is None:
                continue
            w.wait()
        
        ikw = dict(dtype=self._recv_val_idx[0].dtype, device=self._recv_val_idx[0].device)
        fkw = dict(dtype=self._recv_val_dat[0].dtype, device=self._recv_val_dat[0].device)
        
        if isinstance(self._buf, Tensor):
            for idx in self._recv_val_idx:
                self._buf[idx] = True
            self._idx = torch.where(self._buf)[0]
            
            imp = torch.empty(self._buf.size(0), **ikw).fill_((2**62-1)*2+1)
            imp[self._idx] = torch.arange(self._idx.size(0), **ikw)
            
            s = self._idx.shape[:1] + self._recv_val_dat[0].shape[1:]
            self._val = torch.zeros(s, **fkw)
            
            for idx, dat in zip(self._recv_val_idx, self._recv_val_dat):
                idx = imp[idx]
                self._val[idx] += dat
        else:
            s = (self._buf,) + self._recv_val_dat[0].shape[1:]
            self._val = torch.zeros(s, **fkw)
            for idx, dat in zip(self._recv_val_idx, self._recv_val_dat):
                self._val[idx] += dat
        
        self._buf = None
        self._recv_val_dat = None
        self._recv_val_idx = None
        self._works_list = None

        if self._events is not None:
            self._events[1].record()
    
    def get(self) -> Tuple[Tensor, Optional[Tensor]]:
        if self._val is None:
            self.wait()
        return self._val, self._idx

class RouteExecutorWork(RouteWorkBase):
    def __init__(self, handle) -> None:
        super().__init__()
        self._handle = handle
        self._events: Optional[Tuple[Event, Event]] = None
    
    def wait(self) -> None:
        if self._events is not None:
            self._events[1].wait(torch.cuda.current_stream())
        return self._handle.wait()
    
    def get(self) -> Tuple[Tensor, Optional[Tensor]]:
        if self._events is not None:
            self._events[1].wait(torch.cuda.current_stream())
        return self._handle.get()

class RouteExecutor:
    def __init__(self) -> None:
        self._stream = torch.cuda.Stream()
        self._executor = ThreadPool(processes=1)
        self._group = dist.new_group()
        
    def async_forward_a2a(self, val: Tensor, idx: Optional[Tensor], route) -> RouteWorkBase:
        return self._async_a2a_impl(route.forward_a2a, val, idx)
    
    def async_backward_a2a(self, val: Tensor, idx: Optional[Tensor], route) -> RouteWorkBase:
        return self._async_a2a_impl(route.backward_a2a, val, idx)
    
    def _async_a2a_impl(self, func, val: Tensor, idx: Optional[Tensor]) -> RouteWorkBase:
        start = Event(use_cuda=val.is_cuda)
        end = Event(use_cuda=val.is_cuda)

        stream: Optional[torch.cuda.Stream] = self._stream
        def run():
            start.wait(stream)
            with torch.cuda.stream(stream):
                ret = func(val, idx, group=self._group).get()
                end.record()
                return ret
        start.record()
        handle = self._executor.apply_async(run)
        return RouteExecutorWork(handle).set_events(start, end)
    
    # def apply_async(self, func, val, idx, async_op) -> RouteWorkBase:
    #     start = Event(val.is_cuda)
    #     end = Event(val.is_cuda)

    #     if not async_op:
    #         start.record()
    #         return func(val, idx).set_events(start, end)

    #     stream: Optional[torch.cuda.Stream] = self._stream
    #     def run():
    #         start.wait(stream)
    #         with torch.cuda.stream(stream):
    #             ret = func(val, idx, group=self._group).get()
    #             end.record()
    #             return ret
    #     start.record()
    #     handle = self._executor.apply_async(run)
    #     return RouteExecutorWork(handle).set_events(start, end)

class RouteCtx:
    def __init__(self, route, state, async_op) -> None:
        self.route = route
        self.state = state
        self.async_op = async_op

class Route:
    def __init__(self,
        src_ids: Tensor,
        dst_ids: Tensor,
        bipartite: bool = False,
    ):
        assert src_ids.dtype == torch.long
        assert dst_ids.dtype == torch.long
        assert src_ids.dim() == 1
        assert dst_ids.dim() == 1
        
        if not bipartite:
            # 要求数据为点分割，即所有分区的src_ids正交
            assert src_ids.size(0) <= dst_ids.size(0)
            assert (dst_ids[:src_ids.size(0)] == src_ids).all()
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # 注意这里的src_ids和dst_ids与PGraph中的二分图含义不一样
        # src_ids表示发送方，实际是PGraph.dst_ids
        # dst_ids表示接收方，实际是PGraph.src_ids
        self.src_size: int = src_ids.size(0)
        self.dst_size: int = dst_ids.size(0)

        # 索引的格式
        ikw = dict(dtype=torch.long, device=src_ids.device)

        # 获得每个分区发送方的节点个数
        all_src_sizes: List[torch.Size] = [None] * world_size
        dist.all_gather_object(all_src_sizes, src_ids.size())

        # 获得总节点个数
        num_nodes = _all_reduce_num_nodes(src_ids, dst_ids)

        # dst_ids节点到局部编号的映射
        imp = torch.empty(num_nodes, **ikw).fill_((2**62-1)*2+1)
        imp[dst_ids] = torch.arange(self.dst_size, **ikw)
        
        # 这部分代码主要是异步获取其他分区的点，重叠一部分计算和IO
        all_src_ids: List[Tensor] = [None] * world_size
        all_src_get = [None] * world_size
        def fetch_src_ids(i: int):
            if i == rank:
                all_src_ids[i] = src_ids
            else:
                s = all_src_sizes[i]
                all_src_ids[i] = torch.empty(s, **ikw)
            all_src_get[i] = dist.broadcast(
                all_src_ids[i], src=i, async_op=True)
        
        self.forward_routes: List[Tensor] = []
        self.backward_routes: List[Tensor] = []

        # xmp有两个作用，一个是计算节点交集，构建索引
        # 另一个是映射src_ids到局部编号
        xmp = torch.empty_like(imp)
        for i in range(world_size):
            # 预取数据
            if i == 0:
                fetch_src_ids(i)
            if i + 1 < world_size:
                fetch_src_ids(i + 1)
            all_src_get[i].wait()
            ids = all_src_ids[i]

            # 计算交集，判断需要传输的部分
            xmp.fill_(0)
            xmp[ids] += 1
            xmp[dst_ids] += 1
            ind = torch.where(xmp > 1)[0]

            # src_ids节点到局部编码的映射
            xmp.fill_((2**62-1)*2+1)
            xmp[ids] = torch.arange(ids.size(0), **ikw)
            
            # 局部编码
            src_ind = xmp[ind]
            dst_ind = imp[ind]

            # 此时只有bw_route是正常的，fw_route需要发送给src_ids所在分区
            fw_route = torch.vstack([src_ind, dst_ind])
            bw_route = torch.vstack([dst_ind, src_ind])

            self.forward_routes.append(fw_route)
            self.backward_routes.append(bw_route)
        
        # 把fw_route发送给src_ids所在分区，构建最终的路由表
        self.forward_routes = _fix_fw_routes(self.forward_routes)
        
        # 满足同构图条件，则每个点添加自环
        if not bipartite:
            rank_ind = torch.arange(src_ids.size(0), **ikw)
            rank_route = torch.vstack([rank_ind, rank_ind])
            self.forward_routes[rank] = rank_route
            self.backward_routes[rank] = rank_route
        
        self.total_time_used: float = 0.0
        dist.barrier()
    
    def to(self, device):
        self.forward_routes = [ro.to(device) for ro in self.forward_routes]
        self.backward_routes = [ro.to(device) for ro in self.backward_routes]
        return self
    
    def _a2a_impl(self,
        val: Tensor,
        idx: Optional[Tensor],
        send_buf_size: int,
        recv_buf_size: int,
        send_routes: List[Tensor],
        recv_routes: List[Tensor],
        group: Optional[Any],
    ) -> RouteWork:
        bkw = dict(dtype=torch.bool, device=val.device)
        ikw = dict(dtype=torch.long, device=val.device)
        fkw = dict(dtype=val.dtype, device=val.device)
        
        if idx is not None:
            msk = torch.zeros(send_buf_size, **bkw).index_fill_(0, idx, 1)
            send_routes = [ro[:,msk[ro[0]]] for ro in send_routes]

            send_sizes = torch.tensor([ro.size(1) for ro in send_routes], **ikw)            
            recv_sizes = torch.zeros_like(send_sizes)
            
            dist.all_to_all_single(recv_sizes, send_sizes, group=group)
            recv_sizes = recv_sizes.tolist()
        
        if idx is None:
            send_val_dat, recv_val_dat = [], []
            recv_val_idx = [ro[0] for ro in recv_routes]
            for i, ro in enumerate(recv_routes):
                s = ro.size(1)
                c = (s,) + val.shape[1:]
                send_val_dat.append(val[send_routes[i][0]])
                recv_val_dat.append(torch.zeros(c, **fkw))
            
            works_list = [
                all_to_all(recv_val_dat, send_val_dat, group=group),
            ]
            
            return RouteWork(
                buf=recv_buf_size,
                recv_val_idx=recv_val_idx,
                recv_val_dat=recv_val_dat,
                works_list=works_list,
            )
        else:
            imp = torch.empty(send_buf_size, **ikw).fill_((2**62-1)*2+1)
            imp[idx] = torch.arange(idx.size(0), **ikw)
            
            send_val_idx, recv_val_idx = [], []
            send_val_dat, recv_val_dat = [], []
            for i, s in enumerate(recv_sizes):
                c = (s,) + val.shape[1:]
                
                send_val_idx.append(send_routes[i][1])
                recv_val_idx.append(torch.zeros(s, **ikw))
                
                send_index = imp[send_routes[i][0]]
                send_val_dat.append(val[send_index])
                recv_val_dat.append(torch.zeros(c, **fkw))
            
            works_list = [
                all_to_all(recv_val_idx, send_val_idx, group=group),
                all_to_all(recv_val_dat, send_val_dat, group=group),
            ]
            
            recv_buf = torch.zeros(recv_buf_size, **bkw)
            return RouteWork(
                buf=recv_buf,
                recv_val_idx=recv_val_idx,
                recv_val_dat=recv_val_dat,
                works_list=works_list,
            )
    
    def forward_a2a(self,
        val: Tensor,
        idx: Optional[Tensor] = None,
        group: Optional[Any] = None,
    ) -> RouteWork:
        if idx is None:
            assert val.size(0) == self.src_size
        else:
            assert val.size(0) == idx.size(0)
        
        if group is None:
            start = Event(use_cuda=val.is_cuda)
            end = Event(use_cuda=val.is_cuda)
            start.record()
        work = self._a2a_impl(
            val=val, idx=idx,
            send_buf_size=self.src_size,
            recv_buf_size=self.dst_size,
            send_routes=self.forward_routes,
            recv_routes=self.backward_routes,
            group=group,
        )
        if group is None:
            return work.set_events(start, end)
        return work
    
    def backward_a2a(self,
        val: Tensor,
        idx: Optional[Tensor] = None,
        group: Optional[Any] = None,
    ) -> RouteWork:
        if idx is None:
            assert val.size(0) == self.dst_size
        else:
            assert val.size(0) == idx.size(0)

        if group is None:
            start = Event(use_cuda=val.is_cuda)
            end = Event(use_cuda=val.is_cuda)
            start.record()
        work = self._a2a_impl(
            val=val, idx=idx,
            send_buf_size=self.dst_size,
            recv_buf_size=self.src_size,
            send_routes=self.backward_routes,
            recv_routes=self.forward_routes,
            group=group,
        )
        if group is None:
            return work.set_events(start, end)
        return work
    
    def apply(self,
        val: Tensor,
        idx: Optional[Tensor],
        state: Dict[str, Any],
        async_op: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        with self._a2a_manager(state, async_op):
            return RouteFunction.apply(val, idx)
    
    @contextmanager
    def _a2a_manager(self, state, async_op) -> RouteCtx:
        global _global_route_context
        stacked_ctx = _global_route_context
        try:
            _global_route_context = RouteCtx(self, state, async_op)
            yield _global_route_context
        finally:
            _global_route_context = stacked_ctx
    
    # def _gather_impl(self,
    #     values_buffer: Tensor,
    #     active_buffer: Optional[Tensor],
    #     values: Tensor,
    #     active: Optional[Tensor],
    #     send_routes: List[Tensor],
    #     recv_routes: List[Tensor],
    #     async_op: bool,
    # ):
    #     ikw = dict(dtype=torch.long, device=values.device)
    #     fkw = dict(dtype=values.dtype, device=values.device)

    #     # 当定义active时，只传输active标记的values到邻居分区，否则传输所有values
    #     if active is not None:
    #         # 计算新的路由表
    #         active_routes = [ro[:,active[ro[0]]] for ro in send_routes]

    #         # 计算接收缓冲区大小
    #         scatter_sizes = [ro.size(1) for ro in active_routes]
    #         scatter_sizes = torch.tensor(scatter_sizes, **ikw)
    #         gather_sizes = torch.zeros_like(scatter_sizes)
    #         dist.all_to_all_single(gather_sizes, scatter_sizes)
    #         gather_sizes = gather_sizes.tolist()
        
    #     def async_run():
    #         if active is not None:
    #             send_val_idx, recv_val_idx = [], []
    #             send_val_dat, recv_val_dat = [], []
    #             for i, s in enumerate(gather_sizes):
    #                 c = (s,) + values.shape[1:]
    #                 send_val_idx.append(active_routes[i][1])
    #                 recv_val_idx.append(torch.zeros(s, **ikw))
    #                 send_val_dat.append(values[active_routes[i][0]])
    #                 recv_val_dat.append(torch.zeros(c, **fkw))
    #             works_list = [
    #                 all_to_all(recv_val_idx, send_val_idx),
    #                 all_to_all(recv_val_dat, send_val_dat),
    #             ]

    #             return GatherWork(
    #                 values_buffer=values_buffer,
    #                 active_buffer=active_buffer,
    #                 recv_val_idx=recv_val_idx,
    #                 recv_val_dat=recv_val_dat,
    #                 works_list=works_list,
    #             )
    #         else:
    #             send_val_dat, recv_val_dat = [], []
    #             recv_val_idx = [ro[0] for ro in recv_routes]
    #             for i, r in enumerate(recv_routes):
    #                 s = r.size(1)
    #                 c = (s,) + values.shape[1:]
    #                 send_val_dat.append(values[send_routes[i][0]])
    #                 recv_val_dat.append(torch.zeros(c, **fkw))
    #             works_list = [
    #                 all_to_all(recv_val_dat, send_val_dat),
    #             ]
    #             return GatherWork(
    #                 values_buffer=values_buffer,
    #                 active_buffer=active_buffer,
    #                 recv_val_idx=recv_val_idx,
    #                 recv_val_dat=recv_val_dat,
    #                 works_list=works_list,
    #             )
            
    #     if async_op and with_nccl():
    #         stream = get_stream()
    #         stream.wait_stream(torch.cuda.current_stream())
    #         with torch.cuda.stream(stream):
    #             return async_run()
    #     else:
    #         return async_run()

    # def gather_forward(self,
    #     val: Tensor,
    #     idx: Optional[Tensor],
    #     async_op: bool = False,
    #     return_idx: bool = True,
    # ):
    #     val, idx = _spread_val_idx(val, idx, self.src_size)
        
    #     bkw = dict(dtype=torch.bool, device=val.device)
    #     fkw = dict(dtype=val.dtype, device=val.device)
        
    #     val_buf = torch.zeros((self.dst_size,) + val.shape[1:], **fkw)
    #     idx_buf = torch.zeros(self.dst_size, **bkw) if return_idx else None

    #     return self._gather_impl(
    #         values_buffer=val_buf,
    #         active_buffer=idx_buf,
    #         values=val,
    #         active=idx,
    #         send_routes=self.forward_routes,
    #         recv_routes=self.backward_routes,
    #         async_op=async_op,
    #     )
    
    # def gather_backward(self,
    #     val: Tensor,
    #     idx: Optional[Tensor],
    #     async_op: bool = False,
    #     return_idx: bool = True,
    # ):
    #     val, idx = _spread_val_idx(val, idx, self.dst_size)
        
    #     bkw = dict(dtype=torch.bool, device=val.device)
    #     fkw = dict(dtype=val.dtype, device=val.device)
        
    #     val_buf = torch.zeros((self.src_size,) + val.shape[1:], **fkw)
    #     idx_buf = torch.zeros(self.src_size, **bkw) if return_idx else None

    #     return self._gather_impl(
    #         values_buffer=val_buf,
    #         active_buffer=idx_buf,
    #         values=val,
    #         active=idx,
    #         send_routes=self.backward_routes,
    #         recv_routes=self.forward_routes,
    #         async_op=async_op,
    #     )

class RouteFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        val: Tensor,
        idx: Optional[Tensor],
    ):
        route_ctx = get_global_route_context()
        route: Route = route_ctx.route
        state: Dict[str, Any] = route_ctx.state
        async_op: bool = route_ctx.async_op

        # cur_work = get_executor().apply_async(route.forward_a2a, val, idx, async_op)
        cur_work = get_executor().async_forward_a2a(val, idx, route)
        if async_op:
            # cur_work = get_executor().async_forward_a2a(val, idx, route)
            work = state.get("__route_fw_work__") or cur_work
            state["__route_fw_work__"] = cur_work
        else:
            # work = route.forward_a2a(val, idx)
            work = cur_work
        
        val, idx = work.get()
        if work.has_events():
            route.total_time_used += work.time_used()
        
        ctx.route = route
        ctx.state = state
        ctx.async_op = async_op
        ctx.save_for_backward(idx)
        
        return val, idx

    @staticmethod
    def backward(
        ctx: autograd.function.FunctionCtx,
        grad: Tensor,
        _: None,
    ):
        route: Route = ctx.route
        state: Dict[str, Any] = ctx.state
        async_op: bool = ctx.async_op
        
        with torch.no_grad():
            idx, = ctx.saved_tensors
            
            # cur_work = get_executor().apply_async(route.backward_a2a, grad, idx, async_op)
            cur_work = get_executor().async_backward_a2a(grad, idx, route)
            if async_op:
                # cur_work = get_executor().async_backward_a2a(grad, idx, route)
                work = state.get("__route_bw_work__") or cur_work
                state["__route_bw_work__"] = cur_work
            else:
                # work = route.backward_a2a(grad, idx)
                work = cur_work
            
            val, idx = work.get()
            if work.has_events():
                route.total_time_used += work.time_used()

            return val, idx

#### private functions

def _all_reduce_num_nodes(
    src_ids: Tensor,
    dst_ids: Tensor,
) -> int:
    max_ids = torch.zeros(1, dtype=src_ids.dtype, device=src_ids.device)
    max_ids = max_ids.max(src_ids.max()) if src_ids.numel() > 0 else max_ids
    max_ids = max_ids.max(dst_ids.max()) if dst_ids.numel() > 0 else max_ids
    dist.all_reduce(max_ids, op=dist.ReduceOp.MAX)
    return max_ids.item() + 1
    
def _fix_fw_routes(tensors: List[Tensor]) -> List[Tensor]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tensor_sizes: List[torch.Size] = [t.size() for t in tensors]
    all_tensor_sizes: List[List[torch.Size]] = [None] * world_size
    dist.all_gather_object(all_tensor_sizes, tensor_sizes)

    new_tensors: List[Tensor] = []
    for i in range(world_size):
        s = all_tensor_sizes[i][rank]
        t = torch.zeros(s).type_as(tensors[i])
        new_tensors.append(t)
    
    all_to_all(new_tensors, tensors).wait()
    return new_tensors

# def _spread_val_idx(val: Tensor, idx: Optional[Tensor], size: int) -> Tuple[Tensor, Optional[Tensor]]:
#     if idx is None:
#         assert val.size(0) == size
#         return val, None
#     else:
#         assert val.size(0) == idx.size(0)
#         x = torch.zeros(
#             size=(size,) + val.shape[1:],
#             dtype=val.dtype,
#             device=val.device,
#         )
#         x[idx] = val
#         y = torch.zeros(
#             size=(size,),
#             dtype=torch.bool,
#             device=idx.device,
#         ).index_fill_(0, idx, 1)
#         return x, y

_global_route_context: Optional[RouteCtx] = None
def get_global_route_context() -> RouteCtx:
    global _global_route_context
    assert _global_route_context is not None
    return _global_route_context

# 创建一个新的stream，专用于Gather异步通信
# _STREAM: Optional[torch.cuda.Stream] = None
# def get_stream() -> torch.cuda.Stream:
#     global _STREAM
#     if _STREAM is None:
#         _STREAM = torch.cuda.Stream()
#     return _STREAM

# @contextmanager
# def stream_manager(enable: bool = True) -> Optional[torch.cuda.Stream]:
#     if enable and with_nccl():
#         stream = get_stream()
#         stream.wait_stream(torch.cuda.current_stream())
#         with torch.cuda.stream(stream):
#             yield stream
#     else:
#         yield

_THREAD_EXEC: Optional[RouteExecutor] = None
def get_executor() -> RouteExecutor:
    global _THREAD_EXEC
    if _THREAD_EXEC is None:
        _THREAD_EXEC = RouteExecutor()
    return _THREAD_EXEC
