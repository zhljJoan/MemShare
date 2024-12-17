from concurrent.futures import ThreadPoolExecutor, thread
from multiprocessing import Process
from multiprocessing.pool import ThreadPool
from typing import Deque
import torch
import torch.distributed as dist
stage = ['train_stream','write_memory','write_mail','lookup']


class PipelineManager:
    def __init__(self,num_tasks = 10):
        self.stream_set = {}
        self.dist_set = {}
        self.args_queue = {}
        self.thread_pool = ThreadPoolExecutor(num_tasks)
        for k in stage:
            self.stream_set[k] = torch.cuda.Stream()
            self.dist_set[k] = dist.new_group()
            self.args_queue[k] = Deque()
    
    def submit(self,state,func,kwargs):
        future = self.thread_pool.submit(self.run, state,func,kwargs)
        return future

    def run(self,state,func,kwargs):
        with torch.cuda.stream(self.stream_set[state]):
            return func(**kwargs,group = self.dist_set[state])

manger = None
def getPipelineManger():
    global manger
    if manger == None:
        manger = PipelineManager()
    return manger
    