import torch
import torch.autograd as autograd
import torch.distributed as dist

from multiprocessing.pool import ThreadPool

from torch import Tensor
from typing import *

from contextlib import contextmanager
from .a2a import all_to_all
from .event import Event



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
        if torch.cuda.is_available():
            self._stream = torch.cuda.Stream()
        else:
            self._stream = None
        self._executor = ThreadPool(processes=1)
        self._group = dist.new_group()
    
    def async_forward_a2a(self, val: Tensor, idx: Optional[Tensor], route) -> RouteWorkBase:
        return self._async_a2a_impl(route.forward_a2a, val, idx)
    
    def async_backward_a2a(self, val: Tensor, idx: Optional[Tensor], route) -> RouteWorkBase:
        return self._async_a2a_impl(route.backward_a2a, val, idx)
    
    @torch.no_grad()
    def _async_a2a_impl(self, func, val: Tensor, idx: Optional[Tensor]) -> RouteWorkBase:
        start = Event(use_cuda=val.is_cuda)
        end = Event(use_cuda=val.is_cuda)

        stream: Optional[torch.cuda.Stream] = self._stream
        def run():
            start.wait(stream)
            with torch.cuda.stream(stream):
                ret = func(val, idx, group=self._group)
                end.record()
                return ret
        start.record()
        handle = self._executor.apply_async(run)
        return RouteExecutorWork(handle).set_events(start, end)


class Route:
    def __init__(self,
        src_ids: Tensor,
        dst_ids: Tensor,
        bipartite: bool = False,
        group: Any = None,
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
        dist.all_gather_object(all_src_sizes, src_ids.size(), group=group)

        # 获得总节点个数
        num_nodes = _all_reduce_num_nodes(src_ids, dst_ids, group=group)

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
                all_src_ids[i], src=i, async_op=True, group=group)
        
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
        self.forward_routes = _fix_fw_routes(self.forward_routes, group=group)
        
        # 满足同构图条件，则每个点添加自环
        if not bipartite:
            rank_ind = torch.arange(src_ids.size(0), **ikw)
            rank_route = torch.vstack([rank_ind, rank_ind])
            self.forward_routes[rank] = rank_route
            self.backward_routes[rank] = rank_route
        dist.barrier()
    
    def to(self, device):
        self.forward_routes = [ro.to(device) for ro in self.forward_routes]
        self.backward_routes = [ro.to(device) for ro in self.backward_routes]
        return self
    
    def _a2a_impl(self,
        value: Tensor,
        index: Optional[Tensor],
        smask: Union[Tensor, int],
        rmask: Union[Tensor, int],
        send_routes: List[Tensor],
        recv_routes: List[Tensor],
        group: Optional[Any],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        bkw = dict(dtype=torch.bool, device=value.device)
        ikw = dict(dtype=torch.long, device=value.device)
        fkw = dict(dtype=value.dtype, device=value.device)

        send_buf_size = smask.size(0) if isinstance(smask, Tensor) else smask
        recv_buf_size = rmask.size(0) if isinstance(rmask, Tensor) else rmask
        
        if index is None:
            send_val_idx = [ro[0] for ro in send_routes]
            recv_val_idx = [ro[0] for ro in recv_routes]
            
            send_val_dat, recv_val_dat = [], []
            for sidx, ridx in zip(send_val_idx, recv_val_idx):
                s = (ridx.size(0),) + value.shape[1:]
                send_val_dat.append(value[sidx])
                recv_val_dat.append(torch.zeros(s, **fkw))

            all_to_all(recv_val_dat, send_val_dat, group=group)

            s = (recv_buf_size,) + value.shape[1:]
            recv_value = torch.zeros(s, **fkw)

            for idx, dat in zip(recv_val_idx, recv_val_dat):
                recv_value[idx] += dat
            return recv_value, None
        else:
            assert value.size(0) == index.size(0)
            if not isinstance(smask, Tensor):
                smask = torch.zeros(send_buf_size, **bkw).index_fill_(0, index, 1)
            send_routes = [ro[:,smask[ro[0]]] for ro in send_routes]

            send_sizes = torch.tensor([ro.size(1) for ro in send_routes], **ikw)
            recv_sizes = torch.zeros_like(send_sizes)
            dist.all_to_all_single(recv_sizes, send_sizes, group=group)
            recv_sizes = recv_sizes.tolist()

            imp = torch.empty(send_buf_size, **ikw).fill_((2**62-1)*2+1)
            imp[index] = torch.arange(index.size(0), **ikw)
            
            send_val_idx, recv_val_idx = [], []
            send_val_dat, recv_val_dat = [], []
            for i, s in enumerate(recv_sizes):
                c = (s,) + value.shape[1:]
                
                send_val_idx.append(send_routes[i][1])
                recv_val_idx.append(torch.zeros(s, **ikw))
                
                send_index = imp[send_routes[i][0]]
                send_val_dat.append(value[send_index])
                recv_val_dat.append(torch.zeros(c, **fkw))
            
            all_to_all(recv_val_idx, send_val_idx, group=group)
            all_to_all(recv_val_dat, send_val_dat, group=group)
            
            rmask = rmask if isinstance(rmask, Tensor) else torch.zeros(recv_buf_size, **bkw)
            for idx in recv_val_idx:
                rmask[idx] = True
            recv_index = torch.where(rmask)[0]

            imp = torch.empty(recv_buf_size, **ikw).fill_((2**62-1)*2+1)
            imp[recv_index] = torch.arange(recv_index.size(0), **ikw)

            s = recv_index.shape[:1] + value.shape[1:]
            recv_value = torch.zeros(s, **fkw)

            for idx, dat in zip(recv_val_idx, recv_val_dat):
                recv_value[imp[idx]] += dat
            return recv_value, recv_index
    
    def forward_a2a(self,
        value: Tensor,
        index: Optional[Tensor] = None,
        group: Optional[Any] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if index is None or index.dtype == torch.long:
            return self._a2a_impl(
                value=value,
                index=index,
                smask=self.src_size,
                rmask=self.dst_size,
                send_routes=self.forward_routes,
                recv_routes=self.backward_routes,
                group=group,
            )
        else:
            smask, index = index, torch.where(index)[0]
            assert smask.size(0) == self.src_size
            rmask = torch.zeros(self.dst_size, dtype=smask.dtype, device=smask.device)
            value, index = self._a2a_impl(
                value=value,
                index=index,
                smask=smask,
                rmask=rmask,
                send_routes=self.forward_routes,
                recv_routes=self.backward_routes,
                group=group,
            )
            return value, rmask
    
    def backward_a2a(self,
        value: Tensor,
        index: Optional[Tensor] = None,
        group: Optional[Any] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if index is None or index.dtype == torch.long:
            return self._a2a_impl(
                value=value,
                index=index,
                smask=self.dst_size,
                rmask=self.src_size,
                send_routes=self.backward_routes,
                recv_routes=self.forward_routes,
                group=group,
            )
        else:
            smask, index = index, torch.where(index)[0]
            assert smask.size(0) == self.dst_size
            rmask = torch.zeros(self.src_size, dtype=smask.dtype, device=smask.device)
            value, index = self._a2a_impl(
                value=value,
                index=index,
                smask=smask,
                rmask=rmask,
                send_routes=self.backward_routes,
                recv_routes=self.forward_routes,
                group=group,
            )
            return value, rmask
    
    def apply(self,
        value: Tensor,
        index: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        with self._a2a_manager():
            return RouteFunction.apply(value, index)
    
    @contextmanager
    def _a2a_manager(self):
        global _global_route_context
        stacked_ctx = _global_route_context
        try:
            _global_route_context = RouteCtx(self)
            yield _global_route_context
        finally:
            _global_route_context = stacked_ctx


class RouteCtx:
    def __init__(self, route) -> None:
        self.route = route


class RouteFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        value: Tensor,
        index: Optional[Tensor],
    ):
        route_ctx = get_global_route_context()
        route: Route = route_ctx.route
        
        value = value.detach()
        if index is not None:
            index = index.detach()
        
        work = get_route_executor().async_forward_a2a(value, index, route)
        recv_value, recv_index = work.get()
        # if work.has_events():
        #     route.total_time_used += work.time_used()
        
        ctx.route = route
        ctx.save_for_backward(recv_index)
        return recv_value, recv_index

    @staticmethod
    def backward(
        ctx: autograd.function.FunctionCtx,
        grad: Tensor,
        _: None,
    ):
        route: Route = ctx.route
        index, = ctx.saved_tensors

        grad = grad.detach()
        
        work = get_route_executor().async_backward_a2a(grad, index, route)
        recv_grad, recv_index = work.get()
        # if work.has_events():
        #     route.total_time_used += work.time_used()

        return recv_grad, recv_index


#### private functions

def _all_reduce_num_nodes(
    src_ids: Tensor,
    dst_ids: Tensor,
    group: Any = None,
) -> int:
    max_ids = torch.zeros(1, dtype=src_ids.dtype, device=src_ids.device)
    max_ids = max_ids.max(src_ids.max()) if src_ids.numel() > 0 else max_ids
    max_ids = max_ids.max(dst_ids.max()) if dst_ids.numel() > 0 else max_ids
    dist.all_reduce(max_ids, op=dist.ReduceOp.MAX, group=group)
    return max_ids.item() + 1
    
def _fix_fw_routes(tensors: List[Tensor], group: Any = None) -> List[Tensor]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tensor_sizes: List[torch.Size] = [t.size() for t in tensors]
    all_tensor_sizes: List[List[torch.Size]] = [None] * world_size
    dist.all_gather_object(all_tensor_sizes, tensor_sizes, group=group)

    new_tensors: List[Tensor] = []
    for i in range(world_size):
        s = all_tensor_sizes[i][rank]
        t = torch.zeros(s).type_as(tensors[i])
        new_tensors.append(t)
    
    all_to_all(new_tensors, tensors, group=group)
    return new_tensors


_global_route_context: Optional[RouteCtx] = None
def get_global_route_context() -> RouteCtx:
    global _global_route_context
    assert _global_route_context is not None
    return _global_route_context


_THREAD_EXEC: Optional[RouteExecutor] = None
def get_route_executor() -> RouteExecutor:
    global _THREAD_EXEC
    if _THREAD_EXEC is None:
        _THREAD_EXEC = RouteExecutor()
    return _THREAD_EXEC
