import sys
from os.path import abspath, join, dirname

sys.path.insert(0, join(abspath(dirname(__file__))))
from torch import Tensor
import torch
from base import NegativeSampling
from base import NegativeSamplingMode
from typing import Any, List, Optional, Tuple, Union

    

class PreNegativeSampling(NegativeSampling):
    r"""The negative sampling configuration of a
    :class:`~torch_geometric.sampler.BaseSampler` when calling
    :meth:`~torch_geometric.sampler.BaseSampler.sample_from_edges`.
    Args:
        mode (str): The negative sampling mode
            (:obj:`"binary"` or :obj:`"triplet"`).
            If set to :obj:`"binary"`, will randomly sample negative links
            from the graph.
            If set to :obj:`"triplet"`, will randomly sample negative
            destination nodes for each positive source node.
        amount (int or float, optional): The ratio of sampled negative edges to
            the number of positive edges. (default: :obj:`1`)
        weight (torch.Tensor, optional): A node-level vector determining the
            sampling of nodes. Does not necessariyl need to sum up to one.
            If not given, negative nodes will be sampled uniformly.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        mode: Union[NegativeSamplingMode, str],
        neg_sample_list: torch.Tensor
    ):
        super(PreNegativeSampling,self).__init__(mode)
        self.neg_sample_list = neg_sample_list
        self.next_pos = 0
    def set_next_pos(self,pos):
        self.next_pos = pos
    def sample(self, num_samples: int,
               num_nodes: Optional[int] = None) -> Tensor:
        r"""Generates :obj:`num_samples` negative samples."""
        if num_nodes is None:
            raise ValueError(
                f"Cannot sample negatives in '{self.__class__.__name__}' "
                f"without passing the 'num_nodes' argument")
        neg_sample_out = self.neg_sample_list[
            self.next_pos:self.next_pos+num_samples,:].reshape(-1)
        self.next_pos = self.next_pos + num_samples
        return neg_sample_out
        #return torch.from_numpy(np.random.randint(num_nodes, size=num_samples))
