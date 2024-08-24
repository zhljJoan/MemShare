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
    self.incretment_sqr = torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device)


  def get_count(self, node_idxs):
    return self.count[node_idxs, :]
  
  def get_incretment(self, node_idxs):
    #print(self.incretment[node_idxs,:].shape,self.count[node_idxs].shape)
    
    return self.incretment[node_idxs,:]/torch.clamp(self.count[node_idxs,:],1)
  
  def get_incretment_remote(self, idx):
    remote_tensor = DistributedTensor(self.incretment)
    remote_count  = DistributedTensor(self.count)
    idx,pos = idx.sort()
    xpos = torch.empty_like(pos)
    xpos[pos] = torch.arange(pos.shape[0],device = pos.device,dtype=pos.dtype)
    return remote_tensor.all_to_all_get(idx)[xpos]/torch.clamp(remote_count.all_to_all_get(idx)[xpos],1)
  

  def get_incretment_sqr(self, node_idxs):
    return self.incretment_sqr[node_idxs, :]
  
  def detach_filter(self):
    self.incretment.detach_()
  
  def update(self, node_idxs, incret):
    self.count[node_idxs, :] = self.count[node_idxs, :] + 1
    self.incretment[node_idxs, :] = self.incretment[node_idxs, :] + incret
    self.incretment_sqr[node_idxs, :] = self.incretment_sqr[node_idxs, :] + incret * incret
  def clear(self):
    self.incretment.zero_()
  def compute_prediction(self, node_idxs):
    mu    = self.incretment[node_idxs, :]/self.count[node_idxs, :]
    sigma = self.incretment_sqr[node_idxs, :]/self.count[node_idxs, :]
    return mu, sigma
  