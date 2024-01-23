import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.distributed as dist

from torch import Tensor
from typing import *

from abc import ABC, abstractmethod
from contextlib import contextmanager

from .sync import VirtualMotions, VirtualForward, BatchSync
from .utils import vector_backward
from starrygl.parallel.utils import *

class SequencePipe(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._pos_begin = 0
        self._pos_end = 0
    
    @abstractmethod
    def get_group(self) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def get_init_states(self) -> Union[Tensor, Sequence[Tensor]]:
        raise NotImplementedError
    
    @abstractmethod
    def forward(self,
        inputs: Sequence[Tensor],
        states: Sequence[Tensor],
    ) -> Tuple[
        Sequence[Tensor],
        Sequence[Tensor],
    ]:
        raise NotImplementedError
    
    def loss_fn(self,
        inputs: Sequence[Tensor],
        labels: Sequence[Tensor],
    ) -> Tensor:
        raise NotImplementedError
    
    def get_ranks(self) -> Sequence[int]:
        world_size = dist.get_world_size(self.get_group())
        return tuple(range(world_size))
    
    def get_model(self) -> Sequence[Tuple[str, nn.Module]]:
        models = []
        for key in dir(self):
            val = getattr(self, key)
            if isinstance(val, nn.Module):
                models.append((key, val))
        return tuple(models)
    
    def parameters(self):
        params: List[nn.Parameter] = []
        for name, m in self.get_model():
            params.extend(m.parameters())
        return params
    
    def to(self, device: Any):
        for _, net in self.get_model():
            net.to(device)
        return self
    
    def all_reduce(self, async_op: bool = False):
        works = []
        for name, net in self.get_model():
            ws = all_reduce_gradients(net, async_op=async_op)
            if async_op:
                works.extend(ws)
            
            ws = all_reduce_buffers(net, async_op=async_op)
            if async_op:
                works.extend(ws)
        if async_op:
            return ws
    
    def apply(self, bs: int, *inputs: Tensor) -> Sequence[Tensor]:
        runtime = SequencePipeRuntime(bs, self)
        return SequencePipeFunction.apply(runtime, *inputs)
    
    def fast_backward(self,
        bs: int,
        inputs: Sequence[Tensor],
        labels: Sequence[Tensor],
    ) -> Optional[Tensor]:
        runtime = SequencePipeRuntime(bs, self, use_fast_backward=True)
        inputs_grads = runtime.backward(inputs, labels)
        vector_backward(inputs, inputs_grads)
        return runtime.acc_loss
    
    @property
    def begin(self) -> int:
        return self._pos_begin
    
    @property
    def end(self) -> int:
        return self._pos_end
    
    @property
    def batch_size(self) -> int:
        return self._pos_end - self._pos_begin
    
    @contextmanager
    def _switch_batch(self, begin: int, end: int):
        saved_begin = self._pos_begin
        saved_end   = self._pos_end

        self._pos_begin = begin
        self._pos_end   = end
        try:
            yield
        finally:
            self._pos_begin = saved_begin
            self._pos_end   = saved_end


class SequencePipeRuntime:
    def __init__(self,
        micro_batch_size: int,
        program: SequencePipe,
        use_fast_backward: bool = False,
    ) -> None:
        self.micro_batch_size = micro_batch_size
        self.program = program
        self.use_fast_backward = use_fast_backward
        
        self.acc_loss = None

        self.group = program.get_group()
        self.ranks = program.get_ranks()
        self.index = self.ranks.index(dist.get_rank(self.group))
        self._last_work = None
        
    def detach_inputs(self, inputs: Sequence[Tensor]) -> Sequence[Tensor]:
        detach = []
        for t in inputs:
            assert t.size(0) == inputs[0].size(0), "The first dimension of all tensors must be the same."
            detach.append(t.detach())
        return detach
    
    def get_begin_end(self, i: int, n: int) -> Tuple[int, int]:
        begin = i * self.micro_batch_size
        end   = min(n, begin + self.micro_batch_size)
        return begin, end
    
    def get_batch_sync(self, tensors: Sequence[Tensor], device: Any) -> BatchSync:
        return BatchSync(
            *tensors,
            seq_index=self.index,
            seq_ranks=self.ranks,
            group=self.group, device=device,
        )
    
    def forward(self, inputs: Sequence[Tensor]) -> Sequence[Tensor]:
        detach = self.detach_inputs(inputs)

        N = inputs[0].size(0)
        S = (N + self.micro_batch_size - 1) // self.micro_batch_size

        motion = VirtualForward(self.index, len(self.ranks), S, batch_vsz=3)

        outputs = None
        ready_recv: Dict[int, BatchSync] = {}
        ready_send: Dict[int, BatchSync] = {}
        while not motion.finished:
            for op, i in motion.step_comp():
                begin, end = self.get_begin_end(i, N)

                if op == "forward":
                    batch_inputs = self.get_batch_inputs(begin, end, detach)
                    
                    if self.index > 0:
                        batch_states = ready_recv.pop(i).wait_state()
                    else:
                        batch_states = self.get_batch_states(begin, end)

                    with self.program._switch_batch(begin, end):
                        batch_outputs, batch_states = \
                            self.program.forward(batch_inputs, batch_states)
                        
                    if self.index + 1 < len(self.ranks):
                        ready_send[i] = self.get_batch_sync(
                            batch_states,
                            device=detach[0].device,
                        )
                    del batch_inputs, batch_states
                    
                    if outputs is None:
                        outputs = []
                        for t in batch_outputs:
                            t = torch.empty(N, *t.shape[1:], dtype=t.dtype, device=t.device)
                            outputs.append(t)
                        outputs = tuple(outputs)

                    for t, b in zip(outputs, batch_outputs):
                        t[begin:end] = b.detach()
                    del batch_outputs

            for op, type, i in motion.step_sync():
                assert type == "state"
                begin, end = self.get_begin_end(i, N)
                if op == "send":
                    ready_send.pop(i).send_state()
                elif op == "recv":
                    ready_recv[i] = self.get_batch_sync(
                        self.get_batch_states(begin, end),
                        device=detach[0].device,
                    )
                    ready_recv[i].recv_state()

        assert not ready_recv
        assert not ready_send
        return outputs

    def backward(self,
        inputs: Sequence[Tensor],
        gradients: Sequence[Tensor],
    ) -> Sequence[Tensor]:
        detach = self.detach_inputs(inputs)
        detach_grads = self.detach_inputs(gradients)

        N = inputs[0].size(0)
        S = (N + self.micro_batch_size - 1) // self.micro_batch_size

        motions = VirtualMotions(self.index, len(self.ranks), S, batch_vsz=3)
        
        ready_recv_s: Dict[int, BatchSync] = {}
        ready_recv_g: Dict[int, BatchSync] = {}
        ready_send_s: Dict[int, BatchSync] = {}
        ready_send_g: Dict[int, BatchSync] = {}
        
        ready_bw_cmp = {}
        
        input_grads = [None] * len(detach)
        while not motions.finished:
            for op, i in motions.step_comp():
                begin, end = self.get_begin_end(i, N)
                if op == "forward":
                    batch_inputs = self.get_batch_inputs(begin, end, detach, inputs)
                    
                    if self.index > 0:
                        ready_send_g[i] = ready_recv_s.pop(i)
                        batch_states = ready_send_g[i].wait_state()
                    else:
                        batch_states = self.get_batch_states(begin, end)
                    
                    with self.program._switch_batch(begin, end):
                        batch_outputs, batch_states = self.program.forward(batch_inputs, batch_states)
                    
                    if self.index + 1 < len(self.ranks):
                        ready_send_s[i] = self.get_batch_sync(
                            batch_states,
                            device=detach[0].device,
                        )
                        ready_recv_g[i] = ready_send_s[i]

                    ready_bw_cmp[i] = (batch_inputs, batch_outputs, batch_states)
                    del batch_inputs, batch_outputs, batch_states

                elif op == "backward":
                    batch_output_grads = self.get_batch_inputs(begin, end, detach_grads)
                    batch_inputs, batch_outputs, batch_states = ready_bw_cmp.pop(i)

                    if self.use_fast_backward:
                        with self.program._switch_batch(begin, end):
                            vec_loss = [self.program.loss_fn(batch_outputs, batch_output_grads)]
                            vec_grad = [torch.ones_like(vec_loss[0])]
                        
                        if self.acc_loss is None:
                            self.acc_loss = vec_loss[0].detach()
                        else:
                            self.acc_loss += vec_loss[0].detach()
                    else:
                        vec_loss = list(batch_outputs)
                        vec_grad = list(batch_output_grads)

                    del batch_outputs, batch_output_grads

                    if self.index + 1 < len(self.ranks):
                        batch_state_grads = ready_recv_g.pop(i).wait_grads()
                        vec_loss.extend(batch_states)
                        vec_grad.extend(batch_state_grads)
                        del batch_state_grads
                    del batch_states

                    vector_backward(vec_loss, vec_grad)

                    for i, t in enumerate(batch_inputs):
                        g, t.grad = t.grad, None
                        if g is None:
                            continue
                        if input_grads[i] is None:
                            input_grads[i] = torch.zeros(N, *g.shape[1:], dtype=g.dtype, device=g.device)
                        input_grads[i][begin:end] = g
                    del batch_inputs

            for op, type, i in motions.step_sync():
                begin, end = self.get_begin_end(i, N)
                if op == "send":
                    if type == "state":
                        ready_send_s.pop(i).send_state()
                    elif type == "grads":
                        ready_send_g.pop(i).send_grads()
                elif op == "recv":
                    if type == "state":
                        ready_recv_s[i] = self.get_batch_sync(
                            self.get_batch_states(begin, end),
                            device=detach[0].device,
                        )
                        ready_recv_s[i].recv_state()
                    elif type == "grads":
                        ready_recv_g[i].recv_grads()
        
        assert not ready_recv_s
        assert not ready_recv_g
        assert not ready_send_s
        assert not ready_send_g
        assert not ready_bw_cmp

        return input_grads

    def get_batch_inputs(self,
        begin: int, end: int,
        detach: Sequence[Tensor],
        inputs: Sequence[Tensor] = None,
    ) -> Sequence[Tensor]:
        batch = []
        for i, t in enumerate(detach):
            assert not t.requires_grad
            t = t[begin:end]
            if inputs and inputs[i].requires_grad:
                t.requires_grad_()
                t.retain_grad()
            batch.append(t)
        return batch
    
    def get_batch_states(self,
        begin: int, end: int,
    ) -> Sequence[Tensor]:
        states = []
        for s in self.program.get_init_states():
            s = s.unsqueeze(0).broadcast_to(
                end - begin, *s.size(),
            ).contiguous()
            states.append(s)
        return states

class SequencePipeFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx: autograd.function.FunctionCtx,
        runtime: SequencePipeRuntime,
        *inputs: Tensor,
    ):
        ctx.save_for_backward(*inputs)
        ctx.saved_runtime = runtime
        return runtime.forward(inputs)

    @staticmethod
    def backward(
        ctx: autograd.function.FunctionCtx,
        *grads: Tensor,
    ):
        inputs: Sequence[Tensor] = ctx.saved_tensors
        runtime: SequencePipeRuntime = ctx.saved_runtime
        with torch.enable_grad():
            input_grads = runtime.backward(inputs, grads)
        return None, *input_grads
    