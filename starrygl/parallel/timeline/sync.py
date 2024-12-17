import torch
import torch.distributed as dist

from torch import Tensor
from typing import *


class BatchSync:
    def __init__(self,
        *state: Tensor,
        seq_index: int,
        seq_ranks: Optional[List[int]] = None,
        group: Any = None,
        device: Any = None,
    ) -> None:
        self._state = state
        self._grads = [None] * len(self._state)
        self._rgrad = torch.tensor(
            [t.requires_grad for t in self._state],
            dtype=torch.bool, device=device,
        )

        self._seq_index = int(seq_index)

        if group is None:
            group = dist.GroupMember.WORLD
        self._group = group
        self._device = torch.device(device)

        if seq_ranks is None:
            group_size = dist.get_world_size(group)
            seq_ranks = range(group_size)
        self._seq_ranks: Tuple[int,...] = tuple(seq_ranks)
        self._works = []
    
    def zip_for_backward(self, *grads: Optional[Tensor]):
        assert len(grads) == len(self._state)
        self._grads = grads
        vec_loss, vec_grad = [], []
        for s, g in zip(self._state, self._grads):
            if s.requires_grad:
                vec_loss.append(s)
                vec_grad.append(g)
        return vec_loss, vec_grad
    
    def wait_state(self) -> Sequence[Tensor]:
        for w in self._works:
            w.wait()
        self._works.clear()

        rgrad = self._rgrad.tolist()
        for r, t in zip(rgrad, self._state):
            assert t.is_leaf
            t.requires_grad_(r)
        return self._state
    
    def wait_grads(self) -> Sequence[Tensor]:
        for w in self._works:
            w.wait()
        self._works.clear()
        
        assert self._grads is not None
        return self._grads
    
    def send_state(self):
        if not self._state:
            return
        if self._seq_index + 1 >= len(self._seq_ranks):
            return
        dst = self._seq_ranks[self._seq_index + 1]
        dst = dist.get_global_rank(self._group, dst)
        dist.isend(self._rgrad, dst=dst, group=self._group)
        for t in self._state:
            dist.isend(t, dst=dst, group=self._group)
    
    def send_grads(self):
        if not self._state:
            return
        if self._seq_index <= 0:
            return
        
        rgrad = self._rgrad.tolist()

        dst = self._seq_ranks[self._seq_index - 1]
        dst = dist.get_global_rank(self._group, dst)
        for r, t in zip(rgrad, self._state):
            if not r:
                continue
            g, t.grad = t.grad, None
            if g is None:
                g = torch.zeros_like(t)
            dist.isend(g, dst=dst, group=self._group)
    
    def recv_state(self):
        if not self._state:
            return
        if self._seq_index <= 0:
            return
        
        src = self._seq_ranks[self._seq_index - 1]
        src = dist.get_global_rank(self._group, src)
        
        self._works.append(
            dist.irecv(self._rgrad, src=src, group=self._group)
        )
        for t in self._state:
            self._works.append(
                dist.irecv(t, src=src, group=self._group)
            )

    def recv_grads(self):
        if not self._state:
            return
        if self._seq_index + 1 >= len(self._seq_ranks):
            return
        
        rgrad = self._rgrad.tolist()

        src = self._seq_ranks[self._seq_index + 1]
        src = dist.get_global_rank(self._group, src)
        for i, (r, t) in enumerate(zip(rgrad, self._state)):
            if not r:
                self._grads[i] = None
                continue
            if self._grads[i] is None:
                self._grads[i] = torch.empty_like(t)
            self._works.append(
                dist.irecv(self._grads[i], src=src, group=self._group)
            )


class VirtualForward:
    def __init__(self,
        index: int,
        num_ranks: int,
        max_count: int,
        batch_vsz: int = 2,
    ) -> None:
        assert batch_vsz > 0

        self._max_count = max_count
        self._bs = batch_vsz

        vmax_count = (max_count + batch_vsz - 1) // batch_vsz

        self._motions: List[ForwardGenerator] = []
        for _ in range(batch_vsz):
            self._motions.append(
                ForwardGenerator(index, num_ranks, vmax_count)
            )
        self._step_count = 0
    
    @property
    def finished(self):
        return self._motions[self._step_count].finished
    
    def step_comp(self):
        for op, i in self._motions[self._step_count].step_comp():
            k = i * self._bs + self._step_count
            if k < self._max_count:
                yield op, k

    def step_sync(self):
        for op, d, i in self._motions[self._step_count].step_sync():
            k = i * self._bs + self._step_count
            if k < self._max_count:
                yield op, d, k
        self._step_count += 1
        self._step_count %= self._bs
        

class ForwardGenerator:
    def __init__(self,
        index: int,
        num_ranks: int,
        max_count: int,
    ) -> None:
        self._index = index
        self._num_ranks = num_ranks

        self._dst_fp = ForwardFootprint(index+1, num_ranks, max_count)
        self._fp = ForwardFootprint(index, num_ranks, max_count)

        self._dst_fp.step()
        _, op, i = self._fp.step()
        self._last_action = op, i
        
        self._finished = False
    
    @property
    def finished(self):
        t = self._dst_fp.finished
        k = self._fp.finished
        op, _ = self._last_action
        return t and k and not op
    
    def step_comp(self):
        if self.finished:
            return
        
        op, i = self._last_action
        self._last_action = None, -1

        if op == "forward":
            yield "forward", i
    
    def step_sync(self):
        if self.finished:
            return
        
        _, dst_op, dst_i = self._dst_fp.step()

        _, op, i = self._fp.step()
        self._last_action = op, i

        if dst_op == "forward":
            yield "send", "state", dst_i
        
        if op == "forward" and self._index > 0:
            yield "recv", "state", i 

class ForwardFootprint:
    def __init__(self,
        index: int,
        num_ranks: int,
        max_count: int,
    ) -> None:
        self._index = index
        self._num_ranks = num_ranks
        self._max_count = max_count

        self._fw_batch_id = 0
        
        self._count = 0
        if index < 0 or index >= num_ranks:
            self._finished = True
        else:
            self._finished = False
    
    @property
    def finished(self):
        return self._finished
    
    def step(self) -> Tuple[int, Optional[str], int]:
        if self._finished:
            return (self._count, None, -1)

        ret = (self._count, "nop", -1)
        if self._count == self._index + self._fw_batch_id:
            ret = (self._count, "forward", self._fw_batch_id)
            self._fw_batch_id += 1
        
        if self._fw_batch_id >= self._max_count:
            self._finished = True

        self._count += 1
        return ret

class VirtualMotions:
    def __init__(self,
        index: int,
        num_ranks: int,
        max_count: int,
        batch_vsz: int = 2,
    ) -> None:
        assert batch_vsz > 0

        self._max_count = max_count
        self._bs = batch_vsz

        vmax_count = (max_count + batch_vsz - 1) // batch_vsz

        self._motions: List[MotionGenerator] = []
        for _ in range(batch_vsz):
            self._motions.append(
                MotionGenerator(index, num_ranks, vmax_count)
            )
        self._step_count = 0
    
    @property
    def finished(self):
        return self._motions[self._step_count].finished
    
    def step_comp(self):
        for op, i in self._motions[self._step_count].step_comp():
            k = i * self._bs + self._step_count
            if k < self._max_count:
                yield op, k

    def step_sync(self):
        for op, d, i in self._motions[self._step_count].step_sync():
            k = i * self._bs + self._step_count
            if k < self._max_count:
                yield op, d, k
        self._step_count += 1
        self._step_count %= self._bs

class MotionGenerator:
    def __init__(self,
        index: int,
        num_ranks: int,
        max_count: int,
    ) -> None:
        self._index = index
        self._num_ranks = num_ranks

        self._src_fp = F1B1Footprint(index-1, num_ranks, max_count)
        self._dst_fp = F1B1Footprint(index+1, num_ranks, max_count)
        self._fp = F1B1Footprint(index, num_ranks, max_count)

        self._src_fp.step()
        self._dst_fp.step()

        _, op, i = self._fp.step()
        self._last_action = op, i
        
        self._finished = False
    
    @property
    def finished(self):
        s = self._src_fp.finished
        t = self._dst_fp.finished
        k = self._fp.finished
        op, _ = self._last_action
        return s and t and k and not op
    
    def step_comp(self):
        if self.finished:
            return
        
        op, i = self._last_action
        self._last_action = None, -1

        if op == "forward":
            yield "forward", i
        elif op == "backward":
            yield "backward", i
    
    def step_sync(self):
        if self.finished:
            return
        
        _, src_op, src_i = self._src_fp.step()
        _, dst_op, dst_i = self._dst_fp.step()

        _, op, i = self._fp.step()
        self._last_action = op, i

        if op == "backward" and \
            self._index + 1 < self._num_ranks:
            yield "recv", "grads", i

        if src_op == "backward":
            assert dst_op != "forward"
            yield "send", "grads", src_i
        elif dst_op == "forward":
            assert src_op != "backward"
            yield "send", "state", dst_i
        
        if op == "forward" and self._index > 0:
            yield "recv", "state", i

class F1B1Footprint:
    def __init__(self,
        index: int,
        num_ranks: int,
        max_count: int,
    ) -> None:
        self._index = index
        self._num_ranks = num_ranks
        self._max_count = max_count

        self._bw_offset = 2 * self._num_ranks - self._index - 1

        self._fw_batch_id = 0
        self._bw_batch_id = 0
        
        self._count = 0
        if index < 0 or index >= num_ranks:
            self._finished = True
        else:
            self._finished = False
    
    @property
    def finished(self):
        return self._finished
    
    def step(self) -> Tuple[int, Optional[str], int]:
        if self._finished:
            return (self._count, None, -1)

        ret = (self._count, "nop", -1)
        if self._count >= self._bw_offset + 2 * self._bw_batch_id:
            ret = (self._count, "backward", self._bw_batch_id)
            self._bw_batch_id += 1
        elif self._fw_batch_id < self._max_count:
            if self._count >= self._index + 2 * self._fw_batch_id:
                ret = (self._count, "forward", self._fw_batch_id)
                self._fw_batch_id += 1
        
        if self._bw_batch_id >= self._max_count:
            self._finished = True

        self._count += 1
        return ret
