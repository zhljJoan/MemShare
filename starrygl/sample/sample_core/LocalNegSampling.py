import random
import sys
from os.path import abspath, join, dirname

from starrygl.distributed.context import DistributedContext

sys.path.insert(0, join(abspath(dirname(__file__))))
from torch import Tensor
import torch
from base import NegativeSampling
from base import NegativeSamplingMode
from typing import Any, List, Optional, Tuple, Union

class LocalNegativeSampling(NegativeSampling):

    def __init__(
        self,
        mode: Union[NegativeSamplingMode, str],
        amount: Union[int, float] = 1,
        unique: bool = False,
        src_node_list: torch.Tensor = None,
        dst_node_list: torch.Tensor = None,
        local_mask = None,
        seed = None,
        prob = None
    ):
        super(LocalNegativeSampling,self).__init__(mode,amount,unique=unique)
        self.src_node_list = src_node_list.to('cpu') if src_node_list is not None else None
        self.dst_node_list = dst_node_list.to('cpu') if dst_node_list is not None else None
        self.rdm = torch.Generator()
        if seed is not None:
            random.seed(seed)
        seed = random.randint(0,100000)
        print('seed is',seed)
        ctx = DistributedContext.get_default_context()
        self.rdm.manual_seed(seed^ctx.rank)
        self.rdm = torch.Generator()
        self.local_mask = local_mask
        if self.local_mask is not None:
            self.local_dst = dst_node_list[local_mask]
        self.prob = prob
        #self.rdm.manual_seed(42)
        #print('dst_nde_list {}\n'.format(dst_node_list))
    def is_binary(self) -> bool:
        return self.mode == NegativeSamplingMode.binary

    def is_triplet(self) -> bool:
        return self.mode == NegativeSamplingMode.triplet

    def sample(self, num_samples: int,
               num_nodes: Optional[int] = None) -> Tensor:
        r"""Generates :obj:`num_samples` negative samples."""
        if self.is_binary():
            if self.src_node_list is None or self.dst_node_list is None:
                return torch.randint(num_nodes, (num_samples, )),torch.randint(num_nodes, (num_samples, ))
            else:
                self.src_node_list[torch.randint(len(self.src_node_list), (num_samples, ))],
                self.dst_node_list[torch.randint(len(self.dst_node_list), (num_samples, ))]
        else:   
            if self.dst_node_list is None:
                return torch.randint(num_nodes, (num_samples, ),generator=self.rdm)
            elif self.local_mask is not None:
                p = torch.rand(size=(num_samples,))
                sr = self.dst_node_list[torch.randint(len(self.dst_node_list), (num_samples, ),generator=self.rdm)]
                sl = self.local_dst[torch.randint(len(self.local_dst), (num_samples, ),generator=self.rdm)]
                s=torch.where(p<=self.prob,sr,sl)
                return s
            else:
                s = torch.randint(len(self.dst_node_list), (num_samples, ),generator=self.rdm)
                return self.dst_node_list[s]
        
