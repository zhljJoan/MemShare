import os
import time
import torch
class time_count:
    total_sample_time = 0
    total_next_batch_time = 0
    total_sample_core_time = 0
    total_fetch_prepare_time = 0
    total_comm_time = 0
    total_build_time = 0
    total_prepare_input_time = 0
    total_build_block_time = 0

    forward_embedding = 0
    forward_all_to_all = 0
    backward_all_to_all = 0

    memory_historical = 0
    memory_update = 0
    memory_get = 0
    memory_enc = 0
    memory_historical_count = 0


    time_forward = 0
    time_backward = 0
    time_sample = 0
    pre_batch = 0
    pre_input = 0
    pos_update = 0
    mem_update = 0
    time_zero = 0
    time_nbrs = 0
    time_attention = 0
    @staticmethod
    def add_memory_count(t1,t2,t3,t4):
        time_count.memory_update += t1
        time_count.memory_get += t2
        time_count.memory_enc += t3
        time_count.memory_historical_count += t4
    @staticmethod
    def add_train_forward_embedding(t1):
        time_count.forward_embedding += t1
    @staticmethod
    def add_train_foward_all_to_all(t1):
        time_count.forward_all_to_all += t1
    @staticmethod
    def add_backward_all_to_all(t1):
        time_count.backward_all_to_all += t1
    @staticmethod
    def add_next(t1,t2):
        time_count.total_sample_time += t2
        time_count.total_next_batch_time +=t1 
    @staticmethod  
    def add_batch(t1,t2,t3,t4) :
        time_count.total_fetch_prepare_time +=t1
        time_count.total_comm_time+=t2
        time_count.total_build_time+=t3
        time_count.total_prepare_input_time+=t4
        
    @staticmethod  
    def add_build_block(t1,t2) :
        time_count.total_sample_core_time += t1
        time_count.total_build_block_time+=t2
    @staticmethod
    def set_zero():
        time_count.total_sample_time =0
        time_count.total_next_batch_time=0
        time_count.total_sample_core_time =0
        time_count.total_fetch_prepare_time=0
        time_count.total_comm_time =0
        time_count.total_build_time =0
        time_count.total_prepare_input_time =0
        time_count.total_build_block_time=0
        
        time_count.forward_embedding = 0
        time_count.forward_all_to_all = 0
        time_count.backward_all_to_all = 0
        time_count.memory_update = 0
        time_count.memory_get = 0
        time_count.memory_enc = 0
        time_count.memory_historical_count = 0
    @staticmethod
    def query():
        return {
            "total_sample_time":time_count.total_sample_time,
            "total_next_batch_time":time_count.total_next_batch_time,
            "total_sample_core_time":time_count.total_sample_core_time,
            "total_fetch_prepare_time":time_count.total_fetch_prepare_time,
            "total_comm_time":time_count.total_comm_time,
            "total_build_time":time_count.total_build_time,
            "total_prepare_input_time":time_count.total_prepare_input_time,
            "total_build_block_time":time_count.total_build_block_time,
            "forward_embedding":time_count.forward_embedding,
            "forward_all_to_all":time_count.forward_all_to_all,
            "backward_all_to_all":time_count.backward_all_to_all,
            "memory_update":time_count.memory_update ,
            "memory_get":time_count.memory_get ,
            "memory_enc":time_count.memory_enc ,
            "memory_historical_count":time_count.memory_historical_count ,
        }
    
    @staticmethod
    def _zero():
        time_count.time_forward = 0
        time_count.time_backward = 0
        time_count.time_sample = 0
        time_count.pre_batch = 0
        time_count.pre_input = 0
        time_count.pos_update = 0
        time_count.mem_update = 0
        time_count.time_zero = 0
        time_count.time_nbrs = 0
        time_count.time_attention = 0
    @staticmethod
    def start():
        # Uncomment for better breakdown timings
        #torch.cuda.synchronize()
        return time.perf_counter()
    @staticmethod
    def elapsed(start):
        # Uncomment for better breakdown timings
        #torch.cuda.synchronize()
        return time.perf_counter() - start
    @staticmethod
    def print():
        print(
            'time_forward = {} time_backward =  {} time_sample =  {} pre_batch = {} pre_input = {} pos_update = {} mem_update = {} time_zero = {} time_nbrs = {} time_attention = {}'.format(
                time_count.time_forward,
                time_count.time_backward,
                time_count.time_sample,
                time_count.pre_batch,
                time_count.pre_input,
                time_count.pos_update,
                time_count.mem_update,
                time_count.time_zero,
                time_count.time_nbrs,
                time_count.time_attention,
            )
        )