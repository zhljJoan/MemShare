import torch

from torch import Tensor
from typing import *

from contextlib import contextmanager

from .ndata import NID, NData
from .edata import EID, SID, DID, EData
from .utils import init_local_edge_index
from ..core import Route


class DistGraph:
    def __init__(self,
        ids: Tensor,
        edge_index: Tensor,
    ):
        # build local_edge_index
        dst_ids = ids
        src_ids, local_edge_index = init_local_edge_index(
            dst_ids=dst_ids,
            edge_index=edge_index,
        )

        # node's attributes
        self.ndata = NData(
            src_size=src_ids.size(0),
            dst_size=dst_ids.size(0),
        )
        self.ndata[NID] = src_ids
        
        # edge's attributes
        self.edata = EData(
            edge_size=local_edge_index.size(1),
        )
        self.edata[EID] = local_edge_index

        # graph's attributes
        self.data = {}
        self.args = {}

        self.route = Route(dst_ids, src_ids)
    
    # @property
    # def edge_index(self) -> Tensor:
    #     return self.edata[EID]
    
    # @property
    # def src_size(self) -> int:
    #     self.ndata.src_size
    
    # @property
    # def dst_size(self) -> int:
    #     self.ndata.dst_size
    
    @contextmanager
    def scoped_manager(self):
        last_ndata = self.ndata
        last_edata = self.edata
        try:
            self.ndata = NData(
                src_size=None,
                dst_size=None,
                p=last_ndata,
            )
            self.edata = EData(
                edge_size=None,
                p=last_edata,
            )
            yield self
        finally:
            self.ndata = last_ndata
            self.edata = last_edata