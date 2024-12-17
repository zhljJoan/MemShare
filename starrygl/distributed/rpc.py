import torch
import torch.distributed.rpc as rpc

from torch import Tensor
from typing import *

__all__ = [
    "rpc_remote_call",
    "rpc_remote_void_call",
    "rpc_remote_exec"
]

def rpc_remote_call(method, rref: rpc.RRef, *args, **kwargs):
    args = (method, rref) + args
    return rpc.rpc_async(rref.owner(), rpc_method_call, args=args, kwargs=kwargs)

def rpc_method_call(method, rref: rpc.RRef, *args, **kwargs):
    self = rref.local_value()
    return method(self, *args, **kwargs)

def rpc_remote_void_call(method, rref: rpc.RRef, *args, **kwargs):
    args = (method, rref) + args
    return rpc.rpc_async(rref.owner(), rpc_method_void_call, args=args, kwargs=kwargs)

def rpc_method_void_call(method, rref: rpc.RRef, *args, **kwargs):
    self = rref.local_value()
    method(self, *args, **kwargs) # return None

def rpc_remote_exec(method, rref: rpc.RRef, *args, **kwargs):
    args = (method, rref) + args
    return rpc.rpc_async(rref.owner(), rpc_method_exec, args=args, kwargs=kwargs)

@rpc.functions.async_execution
def rpc_method_exec(method, rref: rpc.RRef, *args, **kwargs):
    self = rref.local_value()
    return method(self, *args, **kwargs)
