import os
import time
import torch
class time_count:
    
    time_forward = 0
    time_backward = 0
    time_memory_updater = 0
    time_embedding = 0
    time_local_update = 0
    time_memory_sync = 0
    time_sample_and_build = 0
    time_memory_fetch = 0
    
    @staticmethod
    def _zero():
        time_count.time_forward = 0
        time_count.time_backward = 0
        time_count.time_memory_updater = 0
        time_count.time_embedding = 0
        time_count.time_local_update = 0
        time_count.time_memory_sync = 0
        time_count.time_sample_and_build = 0
        time_count.time_memory_fetch = 0
    @staticmethod
    def start_gpu():
        # Uncomment for better breakdown timings
        #torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return start_event,end_event
    @staticmethod
    def start():
        return time.perf_counter(),0 
    @staticmethod
    def elapsed_event(start_event,end_event):
        if start_event.isinstance(torch.cuda.Event):
            end_event.record()
            end_event.synchronize()
            return start_event.elapsed_time(end_event)
        else:
            torch.cuda.synchronize()
            return time.perf_counter() - start_event
    @staticmethod
    def print():
        print('time_count.time_forward={} time_count.time_backward={} time_count.time_memory_updater={} time_count.time_embedding={} time_count.time_local_update={} time_count.time_memory_sync={} time_count.time_sample_and_build={} time_count.time_memory_fetch={}\n'.format(
            time_count.time_backward,
            time_count.time_memory_updater,
            time_count.time_embedding,
            time_count.time_local_update,
            time_count.time_memory_sync,
            time_count.time_sample_and_build,
            time_count.time_memory_fetch ))