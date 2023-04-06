import torch
import torch.distributed as dist

from torch import Tensor
from typing import *

from .a2a import all_to_all, Works, with_nccl



class GatherWork:
    def __init__(self,
        values_buffer: Tensor,
        active_buffer: Optional[Tensor],
        recv_val_idx: List[Tensor],
        recv_val_dat: List[Tensor],
        works_list: List[Works],
    ) -> None:
        self._waited = False
        self._values_buffer = values_buffer
        self._active_buffer = active_buffer
        self._recv_val_idx = recv_val_idx
        self._recv_val_dat = recv_val_dat
        self._works_list = works_list
    
    def _wait(self) -> None:
        assert not self._waited
        for w in self._works_list:
            if w is None:
                continue
            w.wait()
        
        if self._active_buffer is None:
            for idx, dat in zip(self._recv_val_idx, self._recv_val_dat):
                self._values_buffer[idx] += dat
        else:
            for idx, dat in zip(self._recv_val_idx, self._recv_val_dat):
                self._values_buffer[idx] += dat
                self._active_buffer[idx] = True
        self._waited = True

    def get_values(self) -> Tensor:
        if not self._waited:
            self._wait()
        return self._values_buffer
    
    def get_active(self) -> Optional[Tensor]:
        if not self._waited:
            self._wait()
        return self._active_buffer


class Route:
    def __init__(self,
        src_ids: Tensor,
        dst_ids: Tensor,
    ):
        assert src_ids.dtype == torch.long
        assert dst_ids.dtype == torch.long
        assert src_ids.dim() == 1
        assert dst_ids.dim() == 1

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
        self.forward_routes = _p2p_recv(self.forward_routes)
        
        # 满足同构图条件，则每个点添加自环
        if src_ids.size(0) <= dst_ids.size(0):
            t = dst_ids[:src_ids.size(0)] == src_ids
            if t.all():
                rank_ind = torch.arange(src_ids.size(0), **ikw)
                rank_route = torch.vstack([rank_ind, rank_ind])
                self.forward_routes[rank] = rank_route
                self.backward_routes[rank] = rank_route
        dist.barrier()
    
    def _gather_impl(self,
        values_buffer: Tensor,
        active_buffer: Optional[Tensor],
        values: Tensor,
        active: Optional[Tensor],
        send_routes: List[Tensor],
        recv_routes: List[Tensor],
        async_op: bool,
    ):
        ikw = dict(dtype=torch.long, device=values.device)
        fkw = dict(dtype=values.dtype, device=values.device)

        # 当定义active时，只传输active标记的values到邻居分区，否则传输所有values
        if active is not None:
            if values.size(0) < active.size(0):
                values = _spread_values(values, active)
            assert values.size(0) == active.size(0)

            # 计算新的路由表
            active_routes = [ro[:,active[ro[0]]] for ro in send_routes]

            # 计算接收缓冲区大小
            scatter_sizes = [ro.size(1) for ro in active_routes]
            scatter_sizes = torch.tensor(scatter_sizes, **ikw)
            gather_sizes = torch.zeros_like(scatter_sizes)
            dist.all_to_all_single(gather_sizes, scatter_sizes)
            gather_sizes = gather_sizes.tolist()
        
        def async_run():
            if active is not None:
                send_val_idx, recv_val_idx = [], []
                send_val_dat, recv_val_dat = [], []
                for i, s in enumerate(gather_sizes):
                    c = (s,) + values.shape[1:]
                    send_val_idx.append(active_routes[i][1])
                    recv_val_idx.append(torch.zeros(s, **ikw))
                    send_val_dat.append(values[active_routes[i][0]])
                    recv_val_dat.append(torch.zeros(c, **fkw))
                works_list = [
                    all_to_all(recv_val_idx, send_val_idx),
                    all_to_all(recv_val_dat, send_val_dat),
                ]

                return GatherWork(
                    values_buffer=values_buffer,
                    active_buffer=active_buffer,
                    recv_val_idx=recv_val_idx,
                    recv_val_dat=recv_val_dat,
                    works_list=works_list,
                )
            else:
                send_val_dat, recv_val_dat = [], []
                recv_val_idx = [ro[0] for ro in recv_routes]
                for i, r in enumerate(recv_routes):
                    s = r.size(1)
                    c = (s,) + values.shape[1:]
                    send_val_dat.append(values[send_routes[i][0]])
                    recv_val_dat.append(torch.zeros(c, **fkw))
                works_list = [
                    all_to_all(recv_val_dat, send_val_dat),
                ]
                return GatherWork(
                    values_buffer=values_buffer,
                    active_buffer=active_buffer,
                    recv_val_idx=recv_val_idx,
                    recv_val_dat=recv_val_dat,
                    works_list=works_list,
                )
            
        if async_op and with_nccl():
            _stream = _get_stream()
            _stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(_stream):
                return async_run()
        else:
            return async_run()

    def gather_forward(self,
        values: Tensor,
        active: Optional[Tensor],
        async_op: bool = False,
    ):
        bkw = dict(dtype=torch.bool, device=values.device)
        fkw = dict(dtype=values.dtype, device=values.device)

        s = (self.dst_size,) + values.shape[1:]
        values_buffer = torch.zeros(s, **fkw)
        active_buffer = torch.zeros(self.dst_size, **bkw)

        return self._gather_impl(
            values_buffer=values_buffer,
            active_buffer=active_buffer,
            values=values,
            active=active,
            send_routes=self.forward_routes,
            recv_routes=self.backward_routes,
            async_op=async_op,
        )
    
    def gather_backward(self,
        values: Tensor,
        active: Optional[Tensor],
        async_op: bool = False,
    ):
        bkw = dict(dtype=torch.bool, device=values.device)
        fkw = dict(dtype=values.dtype, device=values.device)
        
        s = (self.src_size,) + values.shape[1:]
        values_buffer = torch.zeros(s, **fkw)
        active_buffer = torch.zeros(self.src_size, **bkw)

        return self._gather_impl(
            values_buffer=values_buffer,
            active_buffer=active_buffer,
            values=values,
            active=active,
            send_routes=self.backward_routes,
            recv_routes=self.forward_routes,
            async_op=async_op,
        )

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
    
def _p2p_recv(tensors: List[Tensor]) -> List[Tensor]:
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

def _spread_values(values: Tensor, active: Tensor) -> Tensor:
    new_values = torch.zeros(
        size=active.shape[:1] + values.shape[1:],
        dtype=values.dtype,
        device=values.device,
    )
    new_values[active] = values
    return new_values

# 创建一个新的stream，专用于Gather异步通信
_STREAM: Optional[torch.cuda.Stream] = None
def _get_stream() -> torch.cuda.Stream:
    global _STREAM
    if _STREAM is None:
        _STREAM = torch.cuda.Stream()
    return _STREAM