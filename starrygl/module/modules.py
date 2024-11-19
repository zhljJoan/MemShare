import time
import time
import torch
from os.path import abspath, join, dirname
import sys

from starrygl.distributed.utils import DistIndex, DistributedTensor
from starrygl.sample.count_static import time_count
sys.path.insert(0, join(abspath(dirname(__file__))))
from layers import *
from memorys import *

class all_to_all_embedding(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input,metadata,neg_samples):
        ctx.save_for_backward(input,metadata['src_pos_index'],
                              metadata['dst_pos_index'],
                              metadata['dst_neg_index'], 
                              metadata['dist_src_index'],
                              metadata['dist_neg_src_index']
                            )
        out = input
        h_pos_src = out[metadata['src_pos_index']]
        h_pos_dst_data = out[metadata['dst_pos_index']]
        h_neg_dst_data = out[metadata['dst_neg_index']]
        h_pos_dst = DistributedTensor(torch.empty_like(h_pos_src))#DistributedTensor(out[metadata['dst_pos_index']].detach().clone())
        h_neg_dst = DistributedTensor(torch.empty(h_pos_src.shape[0]*neg_samples,
                                     h_pos_src.shape[1],dtype = h_pos_src.dtype,device = h_pos_src.device))#DistributedTensor(out[metadata['dst_neg_index']].detach().clone()
        dist_index,ind = metadata['dist_src_index'].sort()
        
        h_pos_dst.all_to_all_set(h_pos_dst_data[ind],dist_index)
        dist_index,ind = metadata['dist_neg_src_index'].sort()
        h_neg_dst.all_to_all_set(h_neg_dst_data[ind],dist_index)
        h_pos_dst = h_pos_dst.accessor.data
        h_neg_dst = h_neg_dst.accessor.data
        return h_pos_src,h_pos_dst,h_neg_dst
    
    @staticmethod
    def backward(ctx, grad_pos_src,remote_pos_dst,remote_neg_dst):
        out,src_pos_index,dst_pos_index,dst_neg_index,dist_src_index,dist_neg_src_index = ctx.saved_tensors
        remote_pos_dst = DistributedTensor(remote_pos_dst)
        remote_neg_dst = DistributedTensor(remote_neg_dst)
        dist_index,ind = dist_src_index.sort()
        grad_pos_dst = remote_pos_dst.all_to_all_get(dist_index)
        grad_pos_dst = grad_pos_dst[ind]
        dist_index,ind = dist_neg_src_index.sort()
        grad_neg_dst = remote_neg_dst.all_to_all_get(dist_index)
        grad_neg_dst = grad_neg_dst[ind]
        grad = torch.empty_like(out)
        grad[src_pos_index] = grad_pos_src
        grad[dst_pos_index] = grad_pos_dst
        grad[dst_neg_index] = grad_neg_dst
        return grad,None,None
class NegFixLayer(torch.autograd.Function):
    def __init__(self):
        super(NegFixLayer, self).__init__()

    def forward(ctx, input, weight):
        ctx.save_for_backward(weight)
        return input

    def backward(ctx, grad_output):
        # Define your backward pass
        # ...
        weight, = ctx.saved_tensors
        #print(weight)
        return grad_output/weight,None

class GeneralModel(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, sample_param, memory_param, gnn_param, train_param, num_nodes = None,mailbox = None,combined=False,train_ratio = None):
        super(GeneralModel, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.sample_param = sample_param
        self.memory_param = memory_param
        #self.train_pos_ratio,self.train_neg_ratio = train_ratio
        if not 'dim_out' in gnn_param:
            gnn_param['dim_out'] = memory_param['dim_out']
        self.gnn_param = gnn_param
        self.train_param = train_param
        #self.neg_fix_layer = NegFixLayer()
        if memory_param['type'] == 'node':
            if memory_param['memory_update'] == 'gru':
                #if memory_param['async'] == False:
                #    self.memory_updater = GRUMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
                #else:
                updater = torch.nn.GRUCell
                #    if memory_param['historical_fix'] == False:
                self.memory_updater = AsyncMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node, updater=updater, mailbox=mailbox, mode = memory_param['mode'])
                #    else:
                #        self.memory_updater = HistoricalMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node,updater=updater,learnable=True,num_nodes=num_nodes)
            elif memory_param['memory_update'] == 'rnn':
                #if memory_param['async'] == False:
                #    self.memory_updater = RNNMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
                #else:
                updater = torch.nn.RNNCell
                    #if memory_param['historical_fix'] == False:
                self.memory_updater = AsyncMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node, updater=updater, mailbox=mailbox, mode = memory_param['mode'])
                #    else:
                #        self.memory_updater = HistoricalMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node,updater=updater,learnable=True,num_nodes=num_nodes)
            elif memory_param['memory_update'] == 'transformer':
                updater = TransformerMemoryUpdater
                self.memory_updater = AsyncMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node, updater=updater, mailbox=mailbox, mode = memory_param['mode'],train_param=train_param)
            else:
                raise NotImplementedError
            self.dim_node_input = memory_param['dim_out']
        self.layers = torch.nn.ModuleDict()
        if gnn_param['arch'] == 'transformer_attention':
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = TransfomerAttentionLayer(self.dim_node_input, dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=combined)
            for l in range(1, gnn_param['layer']):
                for h in range(sample_param['history']):
                    self.layers['l' + str(l) + 'h' + str(h)] = TransfomerAttentionLayer(gnn_param['dim_out'], dim_edge, gnn_param['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=False)
        elif gnn_param['arch'] == 'identity':
            self.gnn_param['layer'] = 1
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = IdentityNormLayer(self.dim_node_input)
                if 'time_transform' in gnn_param and gnn_param['time_transform'] == 'JODIE':
                    self.layers['l0h' + str(h) + 't'] = JODIETimeEmbedding(gnn_param['dim_out'])
        else:
            raise NotImplementedError
        
        self.all_to_all_embedding = all_to_all_embedding.apply
        self.edge_predictor = EdgePredictor(gnn_param['dim_out'])
        if 'combine' in gnn_param and gnn_param['combine'] == 'rnn':
            self.combiner = torch.nn.RNN(gnn_param['dim_out'], gnn_param['dim_out'])
            
                
    def forward(self, mfgs, metadata = None,neg_samples=1, mode = 'triplet',async_param = None):
        t0 = tt.start_gpu()
        if self.memory_param['type'] == 'node':
            self.memory_updater(mfgs[0],async_param)
        t_mem = tt.elapsed_event(t0)
        tt.time_memory_updater += t_mem
        out = list()
        t1 = tt.start_gpu()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                if 'time_transform' in self.gnn_param and self.gnn_param['time_transform'] == 'JODIE':
                    rst = self.layers['l0h' + str(h) + 't'](rst, mfgs[l][h].srcdata['mem_ts'], mfgs[l][h].srcdata['ts'])
                if l != self.gnn_param['layer'] - 1:
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    out.append(rst)
        out = out[0]
        self.embedding = out.detach().clone()
        if self.gnn_param['dyrep']:
            out = self.memory_updater.last_updated_memory
        self.out = out
        h_pos_src = out[metadata['src_pos_index']]
        h_pos_dst = out[metadata['dst_pos_index']]
        h_neg_dst = out[metadata['dst_neg_index']]
    
        #end.record()
        #end.synchronize()
        #elapsed_time_ms = start.elapsed_time(end)
        #print('time {}\n'.format(elapsed_time_ms))
        #print('pos src {} \n pos dst {} \n neg dst{} \n'.format(h_pos_src, h_pos_dst,h_neg_dst))
        #print('pre predict {}'.format(mfgs[0][0].srcdata['ID']))
        #if self.training is True:
        #    with torch.no_grad():
        #        ones = torch.ones(h_neg_dst.shape[0],device = h_neg_dst.device,dtype=torch.float)
        #        weight = torch.where(DistIndex(mfgs[0][0].srcdata['ID'][metadata['dst_neg_index']]).part == torch.distributed.get_rank(),ones/self.train_pos_ratio,ones/self.train_neg_ratio).reshape(-1,1)
                #weight = torch.clip(weigh)
                #weight = weight/weight.max().item()
                #print(weight)
                #weight = 
            #h_neg_dst*weight
        #    pred = self.edge_predictor(h_pos_src, h_pos_dst, None , self.neg_fix_layer.apply(h_neg_dst,weight), neg_samples=neg_samples, mode = mode)
        #else:
        pred = self.edge_predictor(h_pos_src, h_pos_dst, None , h_neg_dst, neg_samples=neg_samples, mode = mode)
        t_embedding = tt.elapsed_event(t1)
        tt.time_embedding+=t_embedding
        return pred
    

class NodeClassificationModel(torch.nn.Module):

    def __init__(self, dim_in, dim_hid, num_class):
        super(NodeClassificationModel, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, dim_hid)
        self.fc2 = torch.nn.Linear(dim_hid, num_class)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x