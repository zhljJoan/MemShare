import torch
import torch.nn as nn
import torch.autograd as autograd

from torch import Tensor
from typing import *

from abc import ABC, abstractmethod
from contextlib import contextmanager
from .route import Route, RouteWork
from .timeline.utils import vector_backward
from .utils import *


__all__ = [
    "LayerPipe",
    "LayerDetach",
]


class LayerPipe(ABC):
    def __init__(self) -> None:
        self._layer_id: Optional[int] = None
        self._snapshot_id: Optional[int] = None
        self._rts: List[LayerPipeRuntime] = []
    
    @property
    def layer_id(self) -> int:
        assert self._layer_id is not None
        return self._layer_id
    
    @property
    def snapshot_id(self) -> int:
        assert self._snapshot_id is not None
        return self._snapshot_id
    
    def apply(self,
        num_layers: int,
        num_snapshots: int,
    ) -> Sequence[Sequence[Tensor]]:
        runtime = LayerPipeRuntime(num_layers, num_snapshots, self)
        self._rts.append(runtime)
        return runtime.forward()
    
    def backward(self):
        for runtime in self._rts:
            runtime.backward()
        self._rts.clear()
    
    def all_reduce(self, async_op: bool = False):
        works = []
        for _, net in self.get_model():
            ws = all_reduce_gradients(net, async_op=async_op)
            if async_op:
                works.extend(ws)
            
            ws = all_reduce_buffers(net, async_op=async_op)
            if async_op:
                works.extend(ws)
        if async_op:
            return ws
    
    def to(self, device: Any):
        for _, net in self.get_model():
            net.to(device)
        return self

    def get_model(self) -> Sequence[Tuple[str, nn.Module]]:
        models = []
        for key in dir(self):
            if key in {"layer_id", "snapshot_id"}:
                continue
            val = getattr(self, key)
            if isinstance(val, nn.Module):
                models.append((key, val))
        return tuple(models)
    
    def parameters(self):
        params: List[nn.Parameter] = []
        for name, m in self.get_model():
            params.extend(m.parameters())
        return params
    
    def register_route(self, *xs: Tensor):
        for t in xs:
            t.requires_route = True

    @abstractmethod
    def get_route(self) -> Route:
        raise NotImplementedError
    
    @abstractmethod
    def layer_inputs(self,
        inputs: Optional[Sequence[Tensor]] = None,
    ) -> Sequence[Tensor]:
        raise NotImplementedError
    
    @abstractmethod
    def layer_forward(self,
        inputs: Sequence[Tensor],
    ) -> Sequence[Tensor]:
        raise NotImplementedError
    
    @contextmanager
    def _switch_layer(self,
        layer_id: int,
        snapshot_id: int,
    ):
        saved_layer_id = self._layer_id
        saved_snapshot_id = self._snapshot_id

        self._layer_id = layer_id
        self._snapshot_id = snapshot_id
        try:
            yield
        finally:
            self._layer_id = saved_layer_id
            self._snapshot_id = saved_snapshot_id


class LayerPipeRuntime:
    def __init__(self,
        num_layers: int,
        num_snapshots: int,
        program: LayerPipe,
    ) -> None:
        self.num_layers = num_layers
        self.num_snapshots = num_snapshots
        self.program = program
        self.ready_bw: Dict[Any, Union[LayerDetach, LayerRoute]] = {}
    
    def forward(self) -> Sequence[Sequence[Tensor]]:
        for op, layer_i, snap_i in ForwardFootprint(self.num_layers, self.num_snapshots):
            if op == "sync":
                xs = self.ready_bw[(layer_i - 1, snap_i, 1)].values() if layer_i > 0 else None
                with self.program._switch_layer(layer_i, snap_i):
                    xs = self.program.layer_inputs(xs)
                    route = self.program.get_route()
                self.ready_bw[(layer_i, snap_i, 0)] = LayerRoute(route, *xs)
            elif op == "comp":
                xs = self.ready_bw[(layer_i, snap_i, 0)].values()
                with self.program._switch_layer(layer_i, snap_i):
                    xs = self.program.layer_forward(xs)
                self.ready_bw[(layer_i, snap_i, 1)] = LayerDetach(*xs)
        
        xs = []
        for snap_i in range(self.num_snapshots):
            layer_i = self.num_layers - 1
            xs.append(self.ready_bw[(layer_i, snap_i, 1)].values())
        return xs

    def backward(self):
        for op, layer_i, snap_i in BackwardFootprint(self.num_layers, self.num_snapshots):
            if op == "sync":
                self.ready_bw[(layer_i, snap_i, 0)].backward()
            elif op == "comp":
                if layer_i + 1 < self.num_layers:
                    self.ready_bw.pop((layer_i + 1, snap_i, 0)).wait_gradients()
                self.ready_bw.pop((layer_i, snap_i, 1)).backward()
        for snap_i in range(self.num_snapshots):
            self.ready_bw.pop((0, snap_i, 0)).wait_gradients()
        assert len(self.ready_bw) == 0

class LayerDetach:
    def __init__(self,
        *inputs: Tensor,
    ) -> None:
        outputs = tuple(t.detach() for t in inputs)
        for s, t in zip(inputs, outputs):
            t.requires_grad_(s.requires_grad)

        self._inputs = inputs
        self._outputs = outputs

    def values(self) -> Sequence[Tensor]:
        return tuple(self._outputs)

    def backward(self) -> None:
        vec_loss, vec_grad = [], []
        for s, t in zip(self._inputs, self._outputs):
            g, t.grad = t.grad, None
            if not s.requires_grad:
                continue
            vec_loss.append(s)
            vec_grad.append(g)
        vector_backward(vec_loss, vec_grad)

class LayerRoute:
    def __init__(self,
        route: Route,
        *inputs: Tensor,
    ) -> None:
        self._route = route
        self._works: Optional[List[Union[Tensor, RouteWork]]] = []
        for t in inputs:
            r = t.requires_route if hasattr(t, "requires_route") else False
            if r:
                self._works.append(self._route.fw_tensor(t, async_op=True))
            else:
                self._works.append(t.detach())

        self._inputs = inputs
        self._outputs: Optional[List[Tensor]] = None
    
    def values(self) -> Sequence[Tensor]:
        if self._outputs is None:
            works, self._works = self._works, None
            assert works is not None

            outputs = []
            for s, t in zip(self._inputs, works):
                if isinstance(t, RouteWork):
                    t = t.wait()
                t = t.requires_grad_(s.requires_grad)
                outputs.append(t)
            self._outputs = outputs
        return self._outputs

    def backward(self):
        assert self._works is None
        assert self._outputs is not None
        works = []
        for s, t in zip(self._inputs, self._outputs):
            g, t.grad = t.grad, None
            rs = s.requires_route if hasattr(s, "requires_route") else False
            rg = s.requires_grad
            if rg and rs:
                works.append(self._route.bw_tensor(g, async_op=True))
            elif rg:
                works.append(g)
            else:
                works.append(None)
        self._works = works
        self._outputs = None

    def wait_gradients(self):
        if self._works is None:
            return
        works, self._works = self._works, None

        vec_loss, vec_grad = [], []
        for t, g in zip(self._inputs, works):
            if isinstance(g, RouteWork):
                g = g.wait()
            if not t.requires_grad:
                continue
            vec_loss.append(t)
            vec_grad.append(g)
        vector_backward(vec_loss, vec_grad)


class ForwardFootprint:
    def __init__(self,
        num_layers: int,
        num_snapshots: int,
    ) -> None:
        self._num_layers = num_layers
        self._num_snapshots = num_snapshots
    
    def __iter__(self):
        if self._num_layers <= 0 or self._num_snapshots <= 0:
            return
        
        # starting
        if self._num_snapshots > 1:
            yield "sync", 0, 0
            yield "sync", 0, 1
        elif self._num_snapshots > 0:
            yield "sync", 0, 0

        for i in range(0, self._num_snapshots, 2):
            for l in range(self._num_layers):
                # snapshot i
                yield "comp", l, i

                if l + 1 < self._num_layers:
                    yield "sync", l + 1, i
                elif i + 2 < self._num_snapshots:
                    yield "sync", 0, i + 2

                # snapshot i + 1
                if i + 1 >= self._num_snapshots:
                    continue
                
                yield "comp", l, i + 1
                
                if l + 1 < self._num_layers:
                    yield "sync", l + 1, i + 1
                elif i + 3 < self._num_snapshots:
                    yield "sync", 0, i + 3


class BackwardFootprint:
    def __init__(self,
        num_layers: int,
        num_snapshots: int,
    ) -> None:
        self._num_layers = num_layers
        self._num_snapshots = num_snapshots

    def __iter__(self):
        if self._num_layers <= 0 or self._num_snapshots <= 0:
            return

        for i in range(0, self._num_snapshots, 2):
            for j in range(self._num_layers):
                l = self._num_layers - j - 1

                # snapshot i
                yield "comp", l, i
                yield "sync", l, i
                
                # snapshot i + 1
                if i + 1 >= self._num_snapshots:
                    continue
                
                yield "comp", l, i + 1
                yield "sync", l, i + 1

