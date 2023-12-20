import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.distributed as dist

from torch import Tensor
from typing import *

from starrygl.distributed.cclib import batch_send, batch_recv, BatchWork


__all__ = [
    "SequencePipe",
]


class SequencePipe:
    def __init__(self,
        batch_size: int,
        seq_ranks: List[int],
        group: Any,
    ) -> None:
        self._batch_size = int(batch_size)
        self._seq_ranks = tuple(seq_ranks)
        self._group = group

        rank = dist.get_rank(group)
        self._index = self._seq_ranks.index(rank)
        
        self._init_states: Optional[Tuple[Tensor,...]] = None
    
    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @property
    def seq_ranks(self):
        return self._seq_ranks
    
    @property
    def num_ranks(self) -> int:
        return len(self._seq_ranks)
    
    @property
    def index(self) -> int:
        return self._index
    
    @property
    def group(self):
        return self._group
    
    def init_states(self, *states: Tensor):
        for s in states:
            assert isinstance(s, Tensor), f"states must be tuple of tensors"
        self._init_states = tuple(states)
        return self
    
    def state_forward(self,
        batch_id: int,
        inputs: Tuple[Tensor,...],
        states: Tuple[Tensor,...],
    ) -> Tuple[Tuple[Tensor,...], Tuple[Tensor,...]]:
        raise NotImplementedError()
    
    def loss_fn(self,
        batch_id: int,
        outputs: Tuple[Tensor,...],
    ) -> Tensor:
        raise NotImplementedError()
    
    @torch.inference_mode()
    def forward(self, *inputs: Tensor) -> Tuple[Tensor,...]:
        inputs = self._detach_inputs(inputs)

        B = inputs[0].size(0)
        num_batchs = (B + self.batch_size - 1) // self.batch_size
        
        last_work = None

        outputs = []
        for batch_id in range(num_batchs):
            start = batch_id * self.batch_size
            end = min(B, start + self.batch_size)
            
            batch_inputs = self._get_batch_inputs(start, end, inputs)
            batch_states = self._get_batch_states(end - start)
            batch_outputs, _, work = self._forward_inner(batch_id, batch_inputs, batch_states)
            outputs.append(batch_outputs)

            if last_work is not None:
                last_work.wait()
            last_work = work

        concat_outputs = []
        for ts in zip(*outputs):
            concat_outputs.append(torch.cat(ts, dim=0))
        
        if last_work is not None:
            last_work.wait()

        return tuple(concat_outputs)
    
    def backward(self, *inputs: Tensor, scale: float = 1.0) -> Tensor:
        # inputs = self._detach_inputs(inputs)

        B = inputs[0].size(0)
        num_batchs = (B + self.batch_size - 1) // self.batch_size

        fw_batch_id = 0
        bw_batch_id = 0

        bw_offset = 2 * self.num_ranks - self.index - 1
        bw_ready: Dict[int, Any] = {}
        last_bw_work = None

        input_batch_grads = []
        total_loss = None

        # hist = []

        count = 0
        while fw_batch_id < num_batchs or bw_batch_id < num_batchs:
            if count >= bw_offset + 2 * bw_batch_id:
                if bw_batch_id < num_batchs:
                    bs, work, *fw_graph = bw_ready.pop(bw_batch_id)
                    work.wait()

                    scale_factor = scale * self.batch_size / bs
                    grads, loss, work = self._backward_inner(bw_batch_id, *fw_graph, scale_factor=scale_factor)
                    # hist.append(f"{count+1}bw{bw_batch_id + 1}")

                    if last_bw_work is not None:
                        last_bw_work.wait()
                    last_bw_work = work

                    total_loss = loss if total_loss is None else total_loss + loss
                    input_batch_grads.append(grads)

                    bw_batch_id += 1
            else:
                fw_starting = (fw_batch_id < self.num_ranks and count >= fw_batch_id + self.index)
                fw_occupying = (count >= self.index + 2 * fw_batch_id)
                if fw_batch_id < num_batchs and (fw_starting or fw_occupying):
                    start = fw_batch_id * self.batch_size
                    end = min(B, start + self.batch_size)
                    batch_inputs = self._get_batch_inputs(start, end, inputs, requires_grad=True)
                    batch_states = self._get_batch_states(end - start, requires_grad=True)

                    batch_outputs, batch_next_states, work = self._forward_inner(fw_batch_id, batch_inputs, batch_states)
                    # hist.append(f"{count+1}fw{fw_batch_id + 1}")
                    bw_ready[fw_batch_id] = [
                        end - start,
                        work,
                        batch_inputs,
                        batch_outputs,
                        batch_states,
                        batch_next_states,
                    ]

                    fw_batch_id += 1
            count += 1
        
        input_grads = []
        for ts in zip(*input_batch_grads):
            if None in ts:
                input_grads.append(None)
            else:
                input_grads.append(torch.cat(ts, dim=0))
        
        if last_bw_work is not None:
            last_bw_work.wait()
            
        self._vector_backward(inputs, input_grads)
        
        # print(f"{self.index+1}: {hist}")
        return total_loss
    
    
    def _load_states(self, states: Tuple[Tensor,...], grad: bool = False):
        for s in states:
            s.grad = None

        if grad: # from target to source
            if self.index + 1 < self.num_ranks:
                s_grads = []
                for s in states:
                    if s.dtype.is_floating_point:
                        s.grad = torch.zeros_like(s.data)
                        s_grads.append(s.grad)
                        
                batch_recv(
                    *s_grads,
                    src=self.seq_ranks[self.index + 1],
                    group=self.group,
                    async_op=True,
                ).wait()
        else:    # from source to target
            if self.index > 0:
                batch_recv(
                    *[s.data for s in states],
                    src=self.seq_ranks[self.index - 1],
                    group=self.group,
                    async_op=True,
                ).wait()

    def _save_states(self, states: Tuple[Tensor,...], grad: bool = False) -> BatchWork:
        if grad: # from target to source
            if self.index > 0:
                s_grads = []
                for s in states:
                    g, s.grad = s.grad, None
                    if s.dtype.is_floating_point:
                        s_grads.append(torch.zeros_like(s) if g is None else g)

                return batch_send(
                    *s_grads,
                    dst=self.seq_ranks[self.index - 1],
                    group=self.group,
                    async_op=True,
                )
        else:    # from source to target
            if self.index + 1 < self.num_ranks:
                return batch_send(
                    *[s.data for s in states],
                    dst=self.seq_ranks[self.index + 1],
                    group=self.group,
                    async_op=True
                )
        return BatchWork(None, None)
    
    def _vector_backward(self, vec_loss: List[Tensor], vec_grad: List[Optional[Tensor]]):
        loss = []
        grad = []
        for x, g in zip(vec_loss, vec_grad):
            if g is None:
                continue
            if not x.requires_grad:
                continue
            # if not x.dtype.is_floating_point:
            #     continue
            loss.append(x.flatten())
            grad.append(g.flatten())

        if len(loss) != 0:
            loss = torch.cat(loss, dim=0)
            grad = torch.cat(grad, dim=0)
            loss.backward(grad)
    
    def _get_batch_inputs(self,
        start: int, end: int,
        inputs: Tuple[Tensor,...],
        requires_grad: bool = False
    ) -> Tuple[Tensor,...]:
        _inputs: List[Tensor] = []
        for t in inputs:
            if requires_grad and t.requires_grad:
                t = t.detach()[start:end]
                t.requires_grad_()
                t.retain_grad()
            else:
                t = t.detach()[start:end]
            _inputs.append(t)
        return tuple(_inputs)

    def _get_batch_states(self, bs: int, requires_grad: bool = False) -> Tuple[Tensor,...]:
        assert self._init_states is not None, "please call init_states()."
        states: List[Tensor] = []
        for s in self._init_states:
            s = s.unsqueeze(0).broadcast_to(bs, *s.size()).contiguous()
            if requires_grad and self.index > 0 and s.dtype.is_floating_point:
                s.requires_grad_()
                s.retain_grad()
            states.append(s)
        return tuple(states)
    
    def _detach_inputs(self, inputs: Tuple[Tensor,...]) -> Tuple[Tensor,...]:
        assert inputs[0].size(0) > 0, "input tensors' batch dimension must be greater than one."
        detach_inputs: List[Tensor] = []
        for t in inputs:
            assert t.size(0) == inputs[0].size(0), "all tensors' batch dimension must be the exact same."
            detach_inputs.append(t.detach())
        return tuple(detach_inputs)
    
    def _forward_inner(self,
        batch_id: int,
        inputs: Tuple[Tensor,...],
        states: Tuple[Tensor,...],
    ) -> Tuple[Tuple[Tensor,...], Tuple[Tensor,...], BatchWork]:
        
        self._load_states(states)
        outputs, next_states = self.state_forward(batch_id, inputs, states)
        work = self._save_states(next_states)
        return outputs, next_states, work
    
    def _backward_inner(self,
        batch_id: int,
        inputs: Tuple[Tensor,...],
        outputs: Tuple[Tensor,...],
        input_states: Tuple[Tensor,...],
        output_states: Tuple[Tensor,...],
        scale_factor: float = 1.0,
    ) -> Tuple[Tuple[Tensor,...], Tensor, BatchWork]:
        loss = self.loss_fn(batch_id, outputs)
        if scale_factor != 1.0:
            loss = loss * scale_factor

        vec_loss = [loss]
        vec_grad = [torch.ones_like(loss)]

        self._load_states(output_states, grad=True)
        for s in output_states:
            if s.requires_grad:
                s.retain_grad()
            g, s.grad = s.grad, None
            vec_loss.append(s)
            vec_grad.append(g)
        self._vector_backward(vec_loss, vec_grad)

        work = self._save_states(input_states, grad=True)

        input_grads = []
        for t in inputs:
            g, t.grad = t.grad, None
            input_grads.append(g)
        return tuple(input_grads), loss.detach(), work
    
