import sys
from os.path import abspath, join, dirname

sys.path.insert(0, join(abspath(dirname(__file__))))
from torch import Tensor
import torch
from base import NegativeSampling
from base import NegativeSamplingMode
from typing import Any, List, Optional, Tuple, Union

class TgbNegativeSampling(NegativeSampling):
    def __init__(
            self,
            mode,
            amount,
            tgb_sampler ,
            ns_data
    ):
        
        super(TgbNegativeSampling,self).__init__(mode,amount)
        self.ns_sampler = tgb_sampler
        self.ns_data = ns_data

    def load_ns(self,split_mode):
        self.ns_sampler.load_eval_set(
            fname=self.ns_data["{}_ns".format(split_mode)], split_mode = split_mode
        )
        self.split_mode = split_mode

    def sample(self,pos_src,pos_dst,pos_t):
        neg_batch_list = self.ns_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=self.split_mode)
        self.amount = len(neg_batch_list[0])
        for i in range(len(neg_batch_list)):
            if(len(neg_batch_list[i])!=self.amount):
                print(len(neg_batch_list[i]))
                for _ in range(self.amount-len(neg_batch_list[i])):
                    neg_batch_list[i].append(0)
        neg =  torch.tensor(neg_batch_list).T.reshape(-1)
        return neg
class EvaluateNegativeSampling(NegativeSampling):
    def __init__(
        self,
        mode: Union[NegativeSamplingMode, str],
        src_node_ids: torch.Tensor,
        dst_node_ids: torch.Tensor,
        interact_times: torch.Tensor = None,
        last_observed_time: float = None,
        amount:int = 1,
        negative_sample_strategy: str = 'random',
        seed: int = None
    ):
        super(EvaluateNegativeSampling,self).__init__(mode,amount)
        self.seed = seed
        self.negative_sample_strategy = negative_sample_strategy
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.interact_times = interact_times
        self.unique_src_nodes_id = src_node_ids.unique()
        self.unique_dst_nodes_id = dst_node_ids.unique()
        self.src_id_mapper = torch.zeros(self.unique_src_nodes_id[-1])
        self.dst_id_mapper = torch.zeros(self.unique_dst_nodes_id[-1])
        self.src_id_mapper[self.unique_src_nodes_id] = torch.arange(self.unique_src_nodes_id.shape[0])
        self.dst_id_mapper[self.unique_dst_nodes_id] = torch.arange(self.unique_dst_nodes_id.shape[0]) 
        self.unique_interact_times = self.interact_times.unique()
        self.earliest_time = self.unique_interact_times.min().item()
        self.last_observed_time = last_observed_time


        if self.negative_sample_strategy == 'inductive':
            # set of observed edges
            self.observed_edges = self.get_unique_edges_between_start_end_time(self.earliest_time, self.last_observed_time)

        if self.seed is not None:
            self.random_state = torch.Generator()
            self.random_state.manual_seed(seed)
        else:
            self.random_state = torch.Generator()
            
    def get_unique_edges_between_start_end_time(self, start_time: float, end_time: float):

        selected_mask = ((self.interact_times >= start_time) and (self.interact_times <= end_time))
        # return the unique select source and destination nodes in the selected time interval
        return torch.cat((self.src_node_ids[selected_mask],self.dst_node_ids[selected_mask]),dim = 1)

    def sample(self, num_samples: int, batch_src_node_ids: Optional[torch.Tensor] = None,
               batch_dst_node_ids: Optional[torch.Tensor] = None, current_batch_start_time: Optional[torch.Tensor] = None,
               current_batch_end_time: Optional[torch.Tensor] = None) -> Tensor:
        if self.negative_sample_strategy == 'random':
            negative_src_node_ids, negative_dst_node_ids = self.random_sample(size=num_samples)
        elif self.negative_sample_strategy == 'historical':
            negative_src_node_ids, negative_dst_node_ids = self.historical_sample(size=num_samples, batch_src_node_ids=batch_src_node_ids,
                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                  current_batch_start_time=current_batch_start_time,
                                                                                  current_batch_end_time=current_batch_end_time)
        elif self.negative_sample_strategy == 'inductive':
            negative_src_node_ids, negative_dst_node_ids = self.inductive_sample(size=num_samples, batch_src_node_ids=batch_src_node_ids,
                                                                                 batch_dst_node_ids=batch_dst_node_ids,
                                                                                 current_batch_start_time=current_batch_start_time,
                                                                                 current_batch_end_time=current_batch_end_time)
        else:
            raise ValueError(f'Not implemented error for negative_sample_strategy {self.negative_sample_strategy}!')
        return negative_src_node_ids, negative_dst_node_ids
    
    def random_sample(self, size: int):

        if self.seed is None:
            random_sample_edge_src_node_indices = torch.randint(0, len(self.unique_src_nodes_id), size)
            random_sample_edge_dst_node_indices = torch.randint(0, len(self.unique_dst_nodes_id), size)
        else:
            random_sample_edge_src_node_indices = torch.randint(0, len(self.unique_src_nodes_id), size, generate = self.random_state)
            random_sample_edge_dst_node_indices = torch.randint(0, len(self.unique_dst_nodes_id), size, generate = self.random_state)
        return self.unique_src_nodes_id[random_sample_edge_src_node_indices], self.unique_dst_nodes_id[random_sample_edge_dst_node_indices]

    def random_sample_with_collision_check(self, size: int, batch_src_nodes_id:torch.Tensor, batch_dst_nodes_id:torch.Tensor):
        batch_edge = torch.stack((batch_src_nodes_id,batch_dst_nodes_id))
        batch_src_index = self.src_id_mapper[batch_src_nodes_id]
        batch_dst_index = self.dst_id_mapper[batch_dst_nodes_id]
        return_edge = torch.tensor([[],[]])
        while(True):
            src_ = torch.randint(0, len(self.unique_src_nodes_id), size*2)
            dst_ = torch.randint(0, len(self.unique_dst_nodes_id), size*2)
            edge = torch.stack((src_,dst_))
            sample_id = src_*self.unique_dst_nodes_id.shape[0] + dst_
            batch_id = batch_src_index * self.unique_dst_nodes_id.shape[0] + batch_dst_index
            mask = torch.isin(sample_id,batch_id,invert = True)
            edge = edge[:,mask]
            if(edge.shape[1] >= size):
                return_edge = torch.cat((return_edge,edge[:,:size]),1)
                break
            else:
                return_edge = torch.cat((return_edge,edge),1)
                size = size - edge.shape[1]
        return return_edge
    
    def historical_sample(self, size: int, batch_src_nodes_id: torch.Tensor, batch_dst_nodes_id: torch.Tensor,
                          current_batch_start_time: float, current_batch_end_time: float):
        assert self.seed is not None

        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time, end_time=current_batch_start_time)
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time, end_time=current_batch_end_time)
        uni,ids = torch.cat((current_batch_edges, historical_edges), dim = 1).unique(dim = 1, return_inverse = False)
        mask = torch.zeros(uni.shape[1],dtype = bool)
        mask[ids[:current_batch_edges.shape[1]]] = True
        mask = (~mask)
        unique_historical_edges = uni[:,mask]
        if size > unique_historical_edges.shape[1]:
            num_random_sample_edges = size - len(unique_historical_edges)
            random_sample_edge = self.random_sample_with_collision_check(size=num_random_sample_edges,batch_src_node_ids=batch_src_nodes_id,
                                                                                                    batch_dst_node_ids=batch_dst_nodes_id)

            sample_edges = torch.cat((unique_historical_edges,random_sample_edge),dim = 1)
        else:
            historical_sample_edge_node_indices = torch.randperm(unique_historical_edges.shape[1],generator=self.random_state)
            sample_edges = unique_historical_edges[:,historical_sample_edge_node_indices[:size]]

        return sample_edges

    def inductive_sample(self, size: int, batch_src_node_ids: torch.Tensor, batch_dst_node_ids: torch.Tensor,
                         current_batch_start_time: float, current_batch_end_time: float):
        assert self.seed is not None

        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time, end_time=current_batch_start_time)

        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time, end_time=current_batch_end_time)

        uni,ids = torch.cat((self.observed_edges,current_batch_edges, historical_edges), dim = 1).unique(dim = 1, return_inverse = False)
        mask = torch.zeros(uni.shape[1],dtype = bool)
        mask[ids[:current_batch_edges.shape[1]+historical_edges.shape[1]]] = True
        mask = (~mask)
        unique_inductive_edges = uni[:,mask]

        if size > len(unique_inductive_edges):
            num_random_sample_edges = size - len(unique_inductive_edges)
            random_sample_edge = self.random_sample_with_collision_check(size=num_random_sample_edges,
                                                                                                             batch_src_node_ids=batch_src_node_ids,
                                                                                                             batch_dst_node_ids=batch_dst_node_ids)

            sample_edges = torch.cat((unique_inductive_edges,random_sample_edge),dim = 1)
        else:
            inductive_sample_edge_node_indices = torch.randperm(unique_inductive_edges.shape[1],generator=self.random_state)
            sample_edges = unique_inductive_edges[:, inductive_sample_edge_node_indices[:size]]

        return sample_edges
