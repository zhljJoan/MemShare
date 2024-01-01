import torch

from torch import Tensor
from typing import *

from .route import Route
from abc import ABC, abstractmethod

class LayerPipe(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_group(self):
        raise NotImplemented

    @abstractmethod
    def get_route(self, tag: int) -> Route:
        raise NotImplemented
    
    @abstractmethod
    def get_graph(self, tag: int) -> Any:
        raise NotImplemented
    
    @abstractmethod
    def forward(self,
        tag: int,
        dist_inputs: Sequence[Tensor],
        self_inputs: Sequence[Tensor],
        time_states: Sequence[Tensor],
    ) -> Tuple[
        Sequence[Tensor],
        Sequence[Tensor],
        Sequence[Tensor],
    ]:
        raise NotImplemented
    
    def load_node(self) -> Sequence[Tensor]:
        pass

    def load_edge(self) -> Sequence[Tensor]:
        pass

    def load_conv_state(self) -> Sequence[Tensor]:
        pass

    def load_time_state(self) -> Sequence[Tensor]:
        pass
    
    def save_as_node(self, *outputs: Tensor):
        pass

    def save_as_edge(self, *outputs: Tensor):
        pass

    def save_as_conv_state(self, *states: Tensor):
        pass

    def save_as_time_state(self, *states: Tensor):
        pass

class SparseLayerPipe:
    def __init__(self) -> None:
        pass