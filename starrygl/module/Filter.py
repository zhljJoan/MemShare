import torch
from torch import nn

from starrygl.distributed.context import DistributedContext
from starrygl.distributed.utils import DistIndex, DistributedTensor

class Filter(nn.Module):

  def __init__(self, n_nodes, memory_dimension,
               device=torch.device('cuda')):
    super(Filter, self).__init__()
    self.n_nodes = n_nodes
    self.memory_dimension = memory_dimension
    self.device = device

    self.__init_filter__()

  def __init_filter__(self):
    """
    Initializes the filter to all zeros. It should be called at the start of each epoch.
    """
    # Treat filter as parameter so that it is saved and loaded together with the model
    self.count = torch.zeros((self.n_nodes),1).to(self.device)
    self.incretment = torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device)


  def get_count(self, node_idxs):
    return self.count[node_idxs, :]
  
  def get_incretment(self, node_idxs):
    #print(self.incretment[node_idxs,:].shape,self.count[node_idxs].shape)
    
    return self.incretment[node_idxs,:]/torch.clamp(self.count[node_idxs,:],1)
  
  def detach_filter(self):
    self.incretment.detach_()
  
  def update(self, node_idxs, incret):
    self.count[node_idxs, :] = self.count[node_idxs, :] + 1
    self.incretment[node_idxs, :] = self.incretment[node_idxs, :] + incret
  
  def clear(self):
    self.count.zero_()
    self.incretment.zero_()
