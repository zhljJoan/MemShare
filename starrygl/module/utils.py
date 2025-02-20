import yaml
import numpy as np
import torch
import math

from starrygl.distributed.context import DistributedContext
def parse_config(f):
    conf = yaml.safe_load(open(f, 'r'))
    sample_param = conf['sampling'][0]
    memory_param = conf['memory'][0]
    gnn_param = conf['gnn'][0]
    train_param = conf['train'][0]
    return sample_param, memory_param, gnn_param, train_param

class EarlyStopMonitor(object):
  def __init__(self, max_round=10, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round
  
class AdaParameter:
  def __init__(self, wait_threshold=0.1, init_beta = 0.1 ,init_alpha = 0.1, min_beta = 0.01, max_beta = 0.8, min_alpha = 0.1, max_alpha = 1):
    self.wait_threshold = wait_threshold
    self.beta = init_beta
    self.alpha = init_alpha
    self.average_fetch = 0
    self.count_fetch = 0
    self.average_memory_sync = 0
    self.count_memory_sync = 0
    self.average_memory_update= 0
    self.count_memory_update = 0
    self.average_gnn_aggregate = 0
    self.count_gnn_aggregate = 0
    self.counts = 0

    self.last_start_event_fetch = None
    self.last_start_event_memory_sync = None
    self.last_start_event_memory_update = None
    self.last_start_event_gnn_aggregate = None

    self.end_event_fetch = None
    self.end_event_memory_sync = None
    self.end_event_memory_update = None
    self.end_event_gnn_aggregate = None

    self.min_beta = min_beta
    self.max_beta = max_beta
    self.max_alpha = max_alpha
    self.min_alpha = min_alpha

    self.training = False
  def train(self):
    self.training = True
  def eval(self):
    self.training = False
  def start_event(self):
    start_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    return start_event
  def update_fetch_time(self,start_event):
    if start_event is None:
      return
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()
    #end_event.synchronize()
    #elapsed_time_ms = start_event.elapsed_time(end_event)
    #self.average_fetch += elapsed_time_ms
    
    #self.count_fetch += 1
    self.end_event_fetch = (self.last_start_event_fetch,end_event)


  def update_memory_sync_time(self,start_event):
    if start_event is None:
      return
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()
    #end_event.synchronize()
    #elapsed_time_ms = start_event.elapsed_time(end_event)
    #self.average_memory_sync += elapsed_time_ms
    #self.count_memory_sync += 1
    self.end_event_memory_sync = (self.last_start_event_memory_sync,end_event)

  def update_memory_update_time(self,start_event):
    if start_event is None:
      return
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()
    #end_event.synchronize()
    #elapsed_time_ms = start_event.elapsed_time(end_event)
    #self.average_memory_update += elapsed_time_ms
    #self.count_memory_update += 1
    self.end_event_memory_update = (self.last_start_event_memory_update,end_event)
  
  def update_gnn_aggregate_time(self,start_event):

    if start_event is None:
      return
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()
    #end_event.synchronize()
    #elapsed_time_ms = start_event.elapsed_time(end_event)
    #self.average_gnn_aggregate += elapsed_time_ms
    #self.count_gnn_aggregate += 1
    self.end_event_gnn_aggregate = (self.last_start_event_gnn_aggregate,end_event)

  def reset_time(self):
    self.average_fetch = 0
    self.count_fetch = 0
    self.average_memory_sync = 0
    self.count_memory_sync = 0
    self.average_memory_update= 0
    self.count_memory_update = 0
    self.average_gnn_aggregate = 0
    self.count_gnn_aggregate = 0
    
  def update_parameter(self):
    #print('beta is {} alpha is {}\n'.format(self.beta,self.alpha))
    #if self.count_fetch == 0 or self.count_memory_sync == 0 or self.count_memory_update == 0 or self.count_gnn_aggregate == 0:
    #  return
    if self.end_event_fetch is None or self.end_event_memory_sync is None or self.end_event_memory_update is None or self.end_event_gnn_aggregate is None:
      return
    self.end_event_fetch[1].synchronize()
    #self.end_event_memory_sync[1].synchronize()
    #self.end_event_memory_update[1].synchronize()
    #self.end_event_gnn_aggregate[1].synchronize()
    self.average_fetch += self.end_event_fetch[0].elapsed_time(self.end_event_fetch[1])
    self.count_fetch += 1
    self.average_memory_sync += self.end_event_memory_sync[0].elapsed_time(self.end_event_memory_sync[1])
    self.count_memory_sync += 1
    self.average_memory_update += self.end_event_memory_update[0].elapsed_time(self.end_event_memory_update[1])
    self.count_memory_update += 1
    self.average_gnn_aggregate += self.end_event_gnn_aggregate[0].elapsed_time(self.end_event_gnn_aggregate[1])
    self.count_gnn_aggregate += 1
    average_gnn_aggregate = self.average_gnn_aggregate/self.count_gnn_aggregate
    average_fetch = self.average_fetch/self.count_fetch
    self.beta = self.beta * average_gnn_aggregate/average_fetch * (1 + self.wait_threshold)
    average_memory_sync_time = self.average_memory_sync/self.count_memory_sync
    average_memory_update_time = self.average_memory_update/self.count_memory_update
    self.alpha = self.alpha-math.log(average_memory_update_time/average_memory_sync_time * (1 + self.wait_threshold))
    self.beta = max(min(self.beta, self.max_beta),self.min_beta)
    self.alpha = max(min(self.alpha, self.max_alpha),self.min_alpha)
    ctx = DistributedContext.get_default_context()
    beta_comm=torch.tensor([self.beta])
    torch.distributed.all_reduce(beta_comm,group=ctx.gloo_group)
    self.beta = beta_comm[0].item()/ctx.world_size
    alpha_comm=torch.tensor([self.alpha])
    torch.distributed.all_reduce(alpha_comm,group=ctx.gloo_group)
    self.alpha = alpha_comm[0].item()/ctx.world_size
    #print('gnn aggregate {} fetch {} memory sync {} memory update {}'.format(average_gnn_aggregate,average_fetch,average_memory_sync_time,average_memory_update_time))
    #print('beta is {} alpha is {}\n'.format(self.beta,self.alpha))
    #self.reset_time()
    #log（2-a1 ） = log(2-a2) * t1/t2 * (1 + wait_threshold)
    #2-a1 = 2-a2 ^(t1/t2 * (1 + wait_threshold))
    #a1 = 2 - 2-a2 ^(t1/t2 * (1 + wait_threshold))


