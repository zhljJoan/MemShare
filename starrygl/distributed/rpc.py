import torch
import torch.distributed.rpc as rpc

from torch import Tensor
from typing import *


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
