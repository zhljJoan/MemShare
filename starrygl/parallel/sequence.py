import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.distributed as dist

from torch import Tensor
from typing import *

from starrygl.distributed.cclib import *
from abc import ABC, abstractmethod
from contextlib import contextmanager


__all__ = [
    "SequencePipe",
]

OptTensor = Optional[Tensor]
SInteger = Sequence[int]
STensor = Sequence[Tensor]
SOptTensor = Sequence[OptTensor]
PairSTensor = Tuple[STensor, STensor]
OptSTensor = Optional[STensor]

class SequencePipe(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._pos_begin = 0
        self._pos_end = 0
        self._pos_start = True
    
    @abstractmethod
    def get_group(self) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def get_init_states(self) -> STensor:
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, inputs: STensor, states: STensor) -> PairSTensor:
        raise NotImplementedError
    
    def loss_fn(self, inputs: STensor, labels: STensor) -> Tensor:
        raise NotImplementedError
    
    def get_ranks(self) -> SInteger:
        world_size = dist.get_world_size(self.get_group())
        return tuple(range(world_size))
    
    def apply(self, bs: int, *inputs: Tensor):
        return SequencePipeFunction.apply(bs, self, *inputs)
    
    def fast_backward(self, bs: int, inputs: STensor, labels: STensor) -> Tensor:
        runtime = SequencePipeRuntime(bs, self, last_layer=True)
        inputs_grads = runtime.backward(inputs, labels)
        runtime.vector_backward(inputs, inputs_grads)
        return runtime.acc_loss
    
    @property
    def begin(self) -> int:
        return self._pos_begin
    
    @property
    def end(self) -> int:
        return self._pos_end
    
    @property
    def start(self) -> bool:
        return self._pos_start
    
    @property
    def batch_size(self) -> int:
        return self._pos_end - self._pos_begin
    
    @contextmanager
    def _switch(self,
        begin: int, end: int, start: bool,
    ):
        saved_begin = self._pos_begin
        saved_end   = self._pos_end
        saved_start = self._pos_start

        self._pos_begin = begin
        self._pos_end   = end
        self._pos_start = start
        yield
        
        self._pos_begin = saved_begin
        self._pos_end   = saved_end
        self._pos_start = saved_start


class SequencePipeRuntime:
    def __init__(self,
        micro_batch_size: int,
        program: SequencePipe,
        last_layer: bool = False,
    ) -> None:
        self.micro_batch_size = micro_batch_size
        self.program = program
        self.last_layer = last_layer
        self.acc_loss = None

        self.group = program.get_group()
        self.ranks = program.get_ranks()
        self.index = self.ranks.index(dist.get_rank(self.group))
        self._last_work = None
    
    def forward(self, inputs: STensor) -> STensor:
        detach = self.detach_inputs(inputs)

        N = inputs[0].size(0)
        S = (N + self.micro_batch_size - 1) // self.micro_batch_size

        outputs = None
        for i in range(S):
            begin = i * self.micro_batch_size
            end = min(N, begin + self.micro_batch_size)
            start = (self.index == 0)
            with self.program._switch(begin, end, start):
                batch_inputs = self.get_batch_inputs(begin, end, detach)
                batch_states = self.get_batch_states(begin, end)
                batch_outputs, batch_states = self.forward_inner(batch_inputs, batch_states)

                if outputs is None:
                    outputs = []
                    for t in batch_outputs:
                        t = torch.empty(N, *t.shape[1:], dtype=t.dtype, device=t.device)
                        outputs.append(t)
                    outputs = tuple(outputs)
                
                for t, b in zip(outputs, batch_outputs):
                    t[begin:end] = b.data
                
                self.wait_last_work(
                    next_one=self.save_states(batch_states),
                )
        self.wait_last_work()
        return outputs

    def backward(self, inputs: STensor, output_grads: OptSTensor = None) -> STensor:
        detach = self.detach_inputs(inputs)
        detach_grads = self.detach_inputs(output_grads)

        N = inputs[0].size(0)
        S = (N + self.micro_batch_size - 1) // self.micro_batch_size

        footprint = F1B1Footprint(self.index, len(self.ranks), S)
        source_footprint = F1B1Footprint(self.index-1, len(self.ranks), S)
        target_footprint = F1B1Footprint(self.index+1, len(self.ranks), S)

        fw_ready = {}
        bw_ready = {}

        input_grads = None
        while True:
            _, op, i = footprint.step()
            _, source_op, source_i = source_footprint.step()
            _, target_op, target_i = target_footprint.step()
            if (not op) and (not source_op) and (not target_op):
                break

            self.wait_last_work()

            if source_op == "backward":
                batch_input_state_grads, = bw_ready.pop(source_i)
                self.wait_last_work(
                    next_one=self.save_grads(batch_input_state_grads),
                )
            elif target_op == "forward":
                *_, batch_output_states = fw_ready[target_i]
                self.wait_last_work(
                    next_one=self.save_states(batch_output_states),
                )
                del _
            
            begin = i * self.micro_batch_size
            end = min(N, begin + self.micro_batch_size)
            start = (self.index == 0)
            with self.program._switch(begin, end, start):
                if op == "forward":
                    batch_inputs = self.get_batch_inputs(begin, end, detach, inputs)
                    batch_input_states = self.get_batch_states(begin, end, requires_grad=True)

                    batch_outputs, batch_output_states = \
                        self.forward_inner(batch_inputs, batch_input_states)
                    fw_ready[i] = [
                        batch_inputs, batch_outputs,
                        batch_input_states, batch_output_states,
                    ]
                elif op == "backward":
                    batch_inputs, batch_outputs,\
                    batch_input_states, batch_output_states = fw_ready.pop(i)
                    batch_output_grads = self.get_batch_inputs(begin, end, detach_grads)

                    batch_input_grads, batch_input_state_grads = \
                        self.backward_inner(
                            batch_inputs, batch_outputs,
                            batch_input_states, batch_output_states,
                            batch_output_grads,
                        )
                    bw_ready[i] = [
                        batch_input_state_grads,
                    ]
                    
                    if input_grads is None:
                        input_grads = []
                        for t in batch_input_grads:
                            if t is not None:
                                t = torch.empty(N, *t.shape[1:], dtype=t.dtype, device=t.device)
                            input_grads.append(t)
                        input_grads = tuple(input_grads)
                    
                    for g, t in zip(input_grads, batch_input_grads):
                        if g is not None:
                            g[begin:end] = t.data
        self.wait_last_work()
        return input_grads

    def wait_last_work(self, next_one = None):
        if self._last_work is not None:
            self._last_work.wait()
        self._last_work = next_one

    def detach_inputs(self, inputs: STensor) -> STensor:
        detach: STensor = []
        for t in inputs:
            assert t.size(0) == inputs[0].size(0), "The first dimension of all tensors must be the same."
            detach.append(t.detach())
        return detach
    
    def get_batch_inputs(self,
        begin: int, end: int,
        detach: STensor, inputs: OptSTensor = None,
    ) -> STensor:
        batch: STensor = []
        for i, t in enumerate(detach):
            assert not t.requires_grad
            if inputs and inputs[i].requires_grad:
                t = t[begin:end]
                t.requires_grad_()
                t.retain_grad()
                batch.append(t)
            else:
                batch.append(t[begin:end])
        return batch
    
    def get_batch_states(self,
        begin: int, end: int,
        requires_grad: bool = False,
    ) -> STensor:
        states = []
        for s in self.program.get_init_states():
            s = s.unsqueeze(0).broadcast_to(
                end - begin, *s.size()).contiguous()
            if requires_grad and self.index > 0 and s.is_floating_point():
                s.requires_grad_()
                s.retain_grad()
            states.append(s)
        return states

    def forward_inner(self,
        inputs: STensor,
        states: STensor,
    ):
        states = self.load_states(states)
        return self.program.forward(inputs, states)

    def backward_inner(self,
        inputs: STensor,
        outputs: STensor,
        input_states: STensor,
        output_states: STensor,
        output_grads: STensor,
    ) -> PairSTensor:
        vloss = []
        vgrad = []
        
        if self.last_layer:
            vloss.append(self.program.loss_fn(outputs, output_grads))
            vgrad.append(torch.ones_like(vloss[-1]))

            if self.acc_loss is None:
                self.acc_loss = vloss[-1].detach()
            else:
                self.acc_loss += vloss[-1].detach()
        else:
            vloss.extend(outputs)
            vgrad.extend(output_grads)
        
        vloss.extend(output_states)
        vgrad.extend(self.load_grads(output_states))

        self.vector_backward(vloss, vgrad)

        input_grads = []
        for t in inputs:
            g, t.grad = t.grad, None
            input_grads.append(g)
        
        input_state_grads = []
        for s in input_states:
            g, s.grad = s.grad, None
            if s.is_floating_point():
                g = torch.zeros_like(s) if g is None else g
            else:
                g = None
            input_state_grads.append(g)

        return input_grads, input_state_grads

    def load_states(self, states: STensor):
        if self.index > 0:
            batch_recv(
                *[s.data for s in states],
                src=self.ranks[self.index - 1],
                group=self.group,
                async_op=True,
            ).wait()
        return states
    
    def load_grads(self, states: STensor):
        grads: SOptTensor = []
        if self.index + 1 < len(self.ranks):
            for s in states:
                if s.is_floating_point():
                    g = torch.zeros_like(s)
                    grads.append(g)
                else:
                    grads.append(None)

            batch_recv(
                *[g.data for g in grads if g is not None],
                src=self.ranks[self.index + 1],
                group=self.group,
                async_op=True,
            ).wait()
        else:
            for s in states:
                grads.append(None)
        return grads

    def save_states(self, states: STensor):
        if self.index + 1 < len(self.ranks):
            return batch_send(
                *[s.data for s in states],
                dst=self.ranks[self.index + 1],
                group=self.group,
                async_op=True,
            )
        else:
            return BatchWork(None, None)
    
    def save_grads(self, grads: STensor):
        if self.index > 0:
            return batch_send(
                *[g.data for g in grads if g is not None],
                dst=self.ranks[self.index - 1],
                group=self.group,
                async_op=True,
            )
        else:
            return BatchWork(None, None)
    
    def vector_backward(self, vloss: STensor, vgrad: STensor):
        loss: List[Tensor] = []
        grad: List[Tensor] = []
        for x, g in zip(vloss, vgrad):
            if g is None:
                continue
            if not x.requires_grad:
                continue
            loss.append(x.flatten())
            grad.append(g.flatten())
        
        if loss:
            loss = torch.cat(loss, dim=0)
            grad = torch.cat(grad, dim=0)
            loss.backward(grad)


class SequencePipeFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        micro_batch_size: int,
        program: SequencePipe,
        *inputs: Tensor,
    ):
        runtime = SequencePipeRuntime(
            micro_batch_size, program
        )
        ctx.save_for_backward(*inputs)
        ctx.saved_runtime = runtime
        return runtime.forward(inputs)

    @staticmethod
    def backward(
        ctx: autograd.function.FunctionCtx,
        *grads: Tensor,
    ):
        inputs: STensor = ctx.saved_tensors
        runtime: SequencePipeRuntime = ctx.saved_runtime
        with torch.enable_grad():
            input_grads = runtime.backward(inputs, grads)
        return None, None, *input_grads
    

# class SequencePipe(nn.Module,ABC):
#     def __init__(self,
#         batch_size: int,
#         seq_ranks: Optional[List[int]] = None,
#         group: Any = None,
#     ) -> None:
#         super().__init__()
#         self._batch_size = int(batch_size)

#         if seq_ranks is None:
#             seq_ranks = list(range(dist.get_world_size(group)))
#         self._seq_ranks = tuple(seq_ranks)
#         self._group = group

#         rank = dist.get_rank(group)
#         self._index = self._seq_ranks.index(rank)
        
#         self._init_states: Optional[Tuple[Tensor,...]] = None

#         self._all_reduce_grads_op = None
#         self._all_reduce_buffers_op = None
#         self._all_reduce_group = None
    
#     @property
#     def batch_size(self) -> int:
#         return self._batch_size
    
#     @property
#     def seq_ranks(self):
#         return self._seq_ranks
    
#     @property
#     def num_ranks(self) -> int:
#         return len(self._seq_ranks)
    
#     @property
#     def index(self) -> int:
#         return self._index
    
#     @property
#     def group(self):
#         return self._group
    
#     def init_states(self, *states: Tensor):
#         for s in states:
#             assert isinstance(s, Tensor), f"states must be tuple of tensors"
#         self._init_states = tuple(states)
#         return self
    
#     def enable_all_reduce(self,
#         grads_op = dist.ReduceOp.SUM,
#         buffers_op = dist.ReduceOp.AVG,
#         group: Any = None,
#     ):
#         self._all_reduce_grads_op = grads_op
#         self._all_reduce_buffers_op = buffers_op
#         self._all_reduce_group = group
#         return self
    
#     def disable_all_reduce(self):
#         self._all_reduce_grads_op = None
#         self._all_reduce_buffers_op = None
#         self._all_reduce_group = None
#         return self
    
#     @abstractmethod
#     def state_forward(self,
#         batch_id: int,
#         inputs: Tuple[Tensor,...],
#         states: Tuple[Tensor,...],
#     ) -> Tuple[Tuple[Tensor,...], Tuple[Tensor,...]]:
#         raise NotImplementedError()
    
#     @abstractmethod
#     def loss_fn(self,
#         batch_id: int,
#         outputs: Tuple[Tensor,...],
#     ) -> Tensor:
#         raise NotImplementedError()
    
#     @torch.inference_mode()
#     def forward(self, *inputs: Tensor) -> Tuple[Tensor,...]:
#         detach = self._detach_inputs(inputs)

#         B = inputs[0].size(0)
#         num_batchs = (B + self.batch_size - 1) // self.batch_size
        
#         last_work = None

#         outputs = None
#         for batch_id in range(num_batchs):
#             start = batch_id * self.batch_size
#             end = min(B, start + self.batch_size)
            
#             batch_inputs = self._get_batch_inputs(start, end, detach)
#             batch_states = self._get_batch_states(end - start)
#             batch_outputs, batch_states = self._forward_inner(batch_id, batch_inputs, batch_states)
            
#             if outputs is None:
#                 outputs = []
#                 for t in batch_outputs:
#                     t = torch.empty(B, *t.shape[1:], dtype=t.dtype, device=t.device)
#                     outputs.append(t)
#                 outputs = tuple(outputs)
            
#             for o, t in zip(outputs, batch_outputs):
#                 o[start:end] = t.data

#             if last_work is not None:
#                 last_work.wait()
#             last_work = self._save_states(batch_states)
        
#         if last_work is not None:
#             last_work.wait()

#         return outputs
    
#     def backward(self, *inputs: Tensor, scale: float = 1.0) -> Tensor:
#         detach = self._detach_inputs(inputs)

#         B = inputs[0].size(0)
#         num_batchs = (B + self.batch_size - 1) // self.batch_size

#         footprint = F1B1Footprint(self.index, self.num_ranks, num_batchs)
#         source_footprint = F1B1Footprint(self.index-1, self.num_ranks, num_batchs)
#         target_footprint = F1B1Footprint(self.index+1, self.num_ranks, num_batchs)

#         fw_ready: Dict[int, Any] = {}
#         bw_ready: Dict[int, Any] = {}

#         last_work = None

#         # input_batch_grads = []
#         input_grads = None
#         total_loss = None

#         while True:
#             _, op, batch_id = footprint.step()
#             _, source_op, source_batch_id = source_footprint.step()
#             _, target_op, target_batch_id = target_footprint.step()
#             if op is None and source_op is None and target_op is None:
#                 break

#             if last_work is not None:
#                 last_work.wait()
#             last_work = None

#             if source_op == "backward":
#                 input_states, = bw_ready.pop(source_batch_id)
#                 last_work = self._save_states(input_states, grad=True)
#                 del input_states
#             elif target_op == "forward":
#                 *_, output_states = fw_ready[target_batch_id]
#                 last_work = self._save_states(output_states)
#                 del _, output_states

#             if op == "forward":
#                 start = batch_id * self.batch_size
#                 end = min(B, start + self.batch_size)

#                 batch_inputs = self._get_batch_inputs(start, end, detach, inputs)
#                 batch_input_states = self._get_batch_states(end - start, requires_grad=True)

#                 batch_outputs, batch_output_states = self._forward_inner(
#                     batch_id,
#                     batch_inputs, batch_input_states,
#                 )
#                 fw_ready[batch_id] = [
#                     batch_inputs,
#                     batch_outputs,
#                     batch_input_states,
#                     batch_output_states,
#                 ]
#             elif op == "backward":
#                 start = batch_id * self.batch_size
#                 end = min(B, start + self.batch_size)

#                 batch_inputs, batch_outputs, batch_input_states, batch_output_states = fw_ready.pop(batch_id)

#                 scale_factor = scale * self.batch_size / (end - start)
#                 grads, loss = self._backward_inner(
#                     batch_id,
#                     batch_inputs,
#                     batch_outputs,
#                     batch_input_states,
#                     batch_output_states,
#                     scale_factor=scale_factor,
#                 )
#                 bw_ready[batch_id] = [
#                     batch_input_states,
#                 ]

#                 total_loss = loss if total_loss is None else total_loss + loss

#                 if input_grads is None:
#                     input_grads = []
#                     for t in grads:
#                         if t is not None:
#                             t = torch.empty(B, *t.shape[1:], dtype=t.dtype, device=t.device)
#                         input_grads.append(t)
#                     input_grads = tuple(input_grads)
                
#                 for g, t in zip(input_grads, grads):
#                     if g is not None:
#                         g[start:end] = t.data
        
#         if last_work is not None:
#             last_work.wait()
        
#         prev_works = self._prev_inputs_backward()
#         self._vector_backward(inputs, input_grads)
#         self._post_inputs_backward(prev_works)
        
#         return total_loss
    
#     def _prev_inputs_backward(self):
#         works = []

#         works.extend(all_reduce_gradients(
#             self, op=self._all_reduce_grads_op,
#             group=self._all_reduce_group, async_op=True,
#         ))
#         works.extend(all_reduce_buffers(
#             self, op=self._all_reduce_buffers_op,
#             group=self._all_reduce_group, async_op=True,
#         ))

#         return works
    
#     def _post_inputs_backward(self, works):
#         for w in works:
#             w.wait()
#         works.clear()
    
#     def _load_states(self, states: Tuple[Tensor,...], grad: bool = False):
#         for s in states:
#             s.grad = None

#         if grad: # from target to source
#             if self.index + 1 < self.num_ranks:
#                 s_grads = []
#                 for s in states:
#                     if s.dtype.is_floating_point:
#                         s.grad = torch.zeros_like(s.data)
#                         s_grads.append(s.grad)
                        
#                 batch_recv(
#                     *s_grads,
#                     src=self.seq_ranks[self.index + 1],
#                     group=self.group,
#                     async_op=True,
#                 ).wait()
#         else:    # from source to target
#             if self.index > 0:
#                 batch_recv(
#                     *[s.data for s in states],
#                     src=self.seq_ranks[self.index - 1],
#                     group=self.group,
#                     async_op=True,
#                 ).wait()

#     def _save_states(self, states: Tuple[Tensor,...], grad: bool = False) -> BatchWork:
#         if grad: # from target to source
#             if self.index > 0:
#                 s_grads = []
#                 for s in states:
#                     g, s.grad = s.grad, None
#                     if s.dtype.is_floating_point:
#                         s_grads.append(torch.zeros_like(s) if g is None else g)

#                 return batch_send(
#                     *s_grads,
#                     dst=self.seq_ranks[self.index - 1],
#                     group=self.group,
#                     async_op=True,
#                 )
#         else:    # from source to target
#             if self.index + 1 < self.num_ranks:
#                 return batch_send(
#                     *[s.data for s in states],
#                     dst=self.seq_ranks[self.index + 1],
#                     group=self.group,
#                     async_op=True
#                 )
#         return BatchWork(None, None)
    
#     def _vector_backward(self, vec_loss: List[Tensor], vec_grad: List[Optional[Tensor]]):
#         loss = []
#         grad = []
#         for x, g in zip(vec_loss, vec_grad):
#             if g is None:
#                 continue
#             if not x.requires_grad:
#                 continue
#             # if not x.dtype.is_floating_point:
#             #     continue
#             loss.append(x.flatten())
#             grad.append(g.flatten())

#         if len(loss) != 0:
#             loss = torch.cat(loss, dim=0)
#             grad = torch.cat(grad, dim=0)
#             loss.backward(grad)
    
#     def _get_batch_inputs(self,
#         start: int, end: int,
#         detach: Tuple[Tensor,...],
#         inputs: Optional[Tuple[Tensor,...]] = None,
#     ) -> Tuple[Tensor,...]:
#         outs: List[Tensor] = []
#         for i, t in enumerate(detach):
#             assert not t.requires_grad
#             if inputs is None:
#                 outs.append(t[start:end])
#             elif inputs[i].requires_grad:
#                 t = t[start:end]
#                 t.requires_grad_()
#                 t.retain_grad()
#                 outs.append(t)
#             else:
#                 outs.append(t[start:end])
#         return tuple(outs)

#     def _get_batch_states(self, bs: int, requires_grad: bool = False) -> Tuple[Tensor,...]:
#         assert self._init_states is not None, "please call init_states()."
#         states: List[Tensor] = []
#         for s in self._init_states:
#             s = s.unsqueeze(0).broadcast_to(bs, *s.size()).contiguous()
#             if requires_grad and self.index > 0 and s.dtype.is_floating_point:
#                 s.requires_grad_()
#                 s.retain_grad()
#             states.append(s)
#         return tuple(states)
    
#     def _detach_inputs(self, inputs: Tuple[Tensor,...]) -> Tuple[Tensor,...]:
#         assert inputs[0].size(0) > 0, "input tensors' batch dimension must be greater than one."
#         detach_inputs: List[Tensor] = []
#         for t in inputs:
#             assert t.size(0) == inputs[0].size(0), "all tensors' batch dimension must be the exact same."
#             detach_inputs.append(t.detach())
#         return tuple(detach_inputs)
    
#     def _forward_inner(self,
#         batch_id: int,
#         inputs: Tuple[Tensor,...],
#         states: Tuple[Tensor,...],
#     ) -> Tuple[Tuple[Tensor,...], Tuple[Tensor,...]]:
        
#         self._load_states(states)
#         outputs, next_states = self.state_forward(batch_id, inputs, states)
#         return outputs, next_states
    
#     def _backward_inner(self,
#         batch_id: int,
#         inputs: Tuple[Tensor,...],
#         outputs: Tuple[Tensor,...],
#         input_states: Tuple[Tensor,...],
#         output_states: Tuple[Tensor,...],
#         scale_factor: float = 1.0,
#     ) -> Tuple[Tuple[Tensor,...], Tensor]:
#         loss = self.loss_fn(batch_id, outputs)
#         if scale_factor != 1.0:
#             loss = loss * scale_factor

#         vec_loss = [loss]
#         vec_grad = [torch.ones_like(loss)]

#         self._load_states(output_states, grad=True)
#         for s in output_states:
#             if s.requires_grad:
#                 s.retain_grad()
#             g, s.grad = s.grad, None
#             vec_loss.append(s)
#             vec_grad.append(g)
#         self._vector_backward(vec_loss, vec_grad)

#         input_grads = []
#         for t in inputs:
#             g, t.grad = t.grad, None
#             input_grads.append(g)
#         return tuple(input_grads), loss.detach()
    

class F1B1Footprint:
    def __init__(self,
        index: int,
        num_ranks: int,
        num_batchs: int,
    ) -> None:
        self._index = index
        self._num_ranks = num_ranks
        self._num_batchs = num_batchs

        self._bw_offset = 2 * self._num_ranks - self._index - 1

        self._fw_batch_id = 0
        self._bw_batch_id = 0
        
        self._count = 0
        if index < 0 or index > num_ranks:
            self._finish = True
        else:
            self._finish = False
    
    def step(self) -> Tuple[int, Optional[str], int]:
        if self._finish:
            return (self._count, None, -1)

        ret = (self._count, "nop", -1)
        if self._count >= self._bw_offset + 2 * self._bw_batch_id:
            ret = (self._count, "backward", self._bw_batch_id)
            self._bw_batch_id += 1
        elif self._fw_batch_id < self._num_batchs:
            if self._count >= self._index + 2 * self._fw_batch_id:
                ret = (self._count, "forward", self._fw_batch_id)
                self._fw_batch_id += 1
        
        if self._bw_batch_id >= self._num_batchs:
            self._finish = True

        self._count += 1
        return ret