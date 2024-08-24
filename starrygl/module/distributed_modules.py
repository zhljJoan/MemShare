import torch
from os.path import abspath, join, dirname
import sys

from starrygl.distributed.utils import DistIndex, DistributedTensor
from starrygl.sample.count_static import time_count
sys.path.insert(0, join(abspath(dirname(__file__))))
from layers import *
from memorys import *
import time
import concurrent.futures

forward_time = 0
backward_time = 0
t = [0,0,0,0]
def get_forward_time():
    global forward_time
    return forward_time
def get_backward_time():
    global backward_time
    return backward_time
def get_t():
    global t
    return t


class all_to_all_embedding(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input,metadata,neg_samples, memory = None, use_emb = False ):
        ctx.save_for_backward(input,
                              metadata['src_pos_index'],
                              metadata['dst_pos_index'],
                              metadata['dst_neg_index'], 
                              metadata['dist_src_index'],
                              metadata['dist_neg_src_index']
                            )
        with torch.no_grad():
            out = input
            h_pos_src = out[metadata['src_pos_index']]
            
            h_dst_index = torch.cat((metadata['dst_pos_index'],metadata['dst_neg_index']))
            #print(h_dst_index)
            h_dst_src_index = torch.cat((metadata['dist_src_index'],metadata['dist_neg_src_index']))
            #print(h_dst_src_index)
            h_dst = DistributedTensor(torch.empty(h_pos_src.shape[0]*(neg_samples+1),
                                         h_pos_src.shape[1],dtype = h_pos_src.dtype,device = h_pos_src.device))
            h_data = out[h_dst_index]
            dist_index,ind = h_dst_src_index.sort()
            h_dst.all_to_all_set(h_data[ind],dist_index)
            h_pos_dst = h_dst.accessor.data[:h_pos_src.shape[0],:]
            h_neg_dst = h_dst.accessor.data[h_pos_src.shape[0]:,:]
            #print(h_pos_dst,h_neg_dst)
            
            """
            h_pos_dst_data = out[metadata['dst_pos_index']]
            h_neg_dst_data = out[metadata['dst_neg_index']]
            h_pos_dst = DistributedTensor(torch.empty_like(h_pos_src,device = h_pos_src.device))
            h_neg_dst = DistributedTensor(torch.empty(h_pos_src.shape[0]*neg_samples,
                                        h_pos_src.shape[1],dtype = h_pos_src.dtype,device = h_pos_src.device))
            dist_index,ind = metadata['dist_src_index'].sort()
            h_pos_dst.all_to_all_set(h_pos_dst_data[ind],dist_index)
            h_pos_dst = h_pos_dst.accessor.data
            dist_index0,ind0 = metadata['dist_neg_src_index'].sort()
            h_neg_dst.all_to_all_set(h_neg_dst_data[ind0],dist_index0)
            h_neg_dst = h_neg_dst.accessor.data
            """
            src_mem = None
            mem = None
            
            if memory is not None:
                
                local_memory = DistributedTensor(memory[metadata['src_pos_index']])
                dst_memory = memory[metadata['dst_pos_index']]
                dist_index0,ind0 = metadata['dist_src_index'].sort()
                send_ptr = local_memory.all_to_all_ind2ptr(dist_index0)
                mem = DistributedTensor(torch.empty_like(local_memory.accessor.data))
                mem.all_to_all_set(dst_memory[ind0],**send_ptr)
                #print(send_ptr,ind.max(),src_mem,local_memory.shape)
                src_mem = local_memory.all_to_all_get(**send_ptr)[ind0]
                #print(src_mem)
                mem = mem.accessor.data
            
            """
                local_memory = DistributedTensor(memory[metadata['src_pos_index']])
                dst_memory = memory[metadata['dst_pos_index']]
                mem = DistributedTensor(torch.empty_like(local_memory.accessor.data))
                mem.all_to_all_set(dst_memory[ind],dist_index)

                src_mem = DistributedTensor(torch.empty_like(dst_memory))
                src_mem = local_memory.all_to_all_send(**metadata['dst_send_dict'])
                mem = mem.accessor.data
            """
            if use_emb is True:
                mem = h_pos_dst
                local_embedding = DistributedTensor(h_pos_src)
                src_mem = local_embedding.all_to_all_send(**metadata['dst_send_dict'])

        #t[2] += t3-t2
        #t[3] += t4-t3
        return h_pos_src,h_pos_dst,h_neg_dst,mem,src_mem
    
    @staticmethod
    def backward(ctx, grad_pos_src,remote_pos_dst,remote_neg_dst,grad0,grad1):
        out,src_pos_index,dst_pos_index,dst_neg_index,dist_src_index,dist_neg_src_index = ctx.saved_tensors
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.time()
            
            remote_dst = DistributedTensor(torch.cat((remote_pos_dst,remote_neg_dst),dim = 0))
            dist_index,ind = torch.cat((dist_src_index,dist_neg_src_index)).sort()
            grad_dst = remote_dst.all_to_all_get(dist_index)
            grad_dst[ind] = grad_dst.clone()
            grad = torch.empty_like(out)
            grad[src_pos_index] = grad_pos_src
            grad[dst_pos_index] = grad_dst[:dst_pos_index.shape[0],:]
            grad[dst_neg_index] = grad_dst[dst_pos_index.shape[0]:,:]
            
            """
            remote_pos_dst = DistributedTensor(remote_pos_dst)
            remote_neg_dst = DistributedTensor(remote_neg_dst)
            dist_index,ind = dist_src_index.sort()
            grad_pos_dst = remote_pos_dst.all_to_all_get(dist_index)
            grad_pos_dst[ind] = grad_pos_dst.clone()
            dist_index_neg,ind_neg = dist_neg_src_index.sort()
            grad_neg_dst = remote_neg_dst.all_to_all_get(dist_index_neg)
            grad_neg_dst[ind_neg] = grad_neg_dst.clone()
        
            grad = torch.empty_like(out)
            grad[src_pos_index] = grad_pos_src
            grad[dst_pos_index] = grad_pos_dst
            grad[dst_neg_index] = grad_neg_dst
            """
            torch.cuda.synchronize()
            t1 = time.time()
            time_count.add_backward_all_to_all(t1 -t0)
        return grad,None,None,None,None
executor = concurrent.futures.ThreadPoolExecutor(1)
class GeneralModel(torch.nn.Module):

    def __init__(self, dim_node, dim_edge, sample_param, memory_param, gnn_param, train_param, combined=False, cache_index=None):
        super(GeneralModel, self).__init__()
        self.dim_node = dim_node
        self.dim_node_input = dim_node
        self.dim_edge = dim_edge
        self.sample_param = sample_param
        self.memory_param = memory_param
        if not 'dim_out' in gnn_param:
            gnn_param['dim_out'] = memory_param['dim_out']
        self.gnn_param = gnn_param
        self.train_param = train_param
        if memory_param['type'] == 'node':
            if memory_param['memory_update'] == 'gru':
                self.memory_updater = GRUMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node,cache_index)
            elif memory_param['memory_update'] == 'rnn':
                self.memory_updater = RNNMemeoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], dim_node)
            elif memory_param['memory_update'] == 'transformer':
                self.memory_updater = TransformerMemoryUpdater(memory_param, 2 * memory_param['dim_out'] + dim_edge, memory_param['dim_out'], memory_param['dim_time'], train_param)
            else:
                raise NotImplementedError
            self.dim_node_input = memory_param['dim_out']
        self.layers = torch.nn.ModuleDict()
        if gnn_param['arch'] == 'transformer_attention':
            for h in range(sample_param['history']):
                self.layers['l0h' + str(h)] = TransfomerAttentionLayer(self.dim_node_input, dim_edge, gnn_param
                                                                       
                                                                       ['dim_time'], gnn_param['att_head'], train_param['dropout'], train_param['att_dropout'], gnn_param['dim_out'], combined=combined)
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
        if 'historical' in self.gnn_param and self.gnn_param['historical'] == True:
            self.historical_cache = {}
            for l in range(0,gnn_param['layer']):
                self.historical_cache['l'+str(l)] = HistoricalCache(cache_index,shape=(cache_index.size,gnn_param['dim_out']),dtype = torch.float,device = torch.device('cuda'))
        self.all_to_all_embedding = all_to_all_embedding.apply
        self.edge_predictor = EdgePredictor(gnn_param['dim_out'])
        self.history_embedding_time = TimeEncode(gnn_param['dim_out'])
        self.historical_update = torch.nn.Linear(gnn_param['dim_out'] + gnn_param['dim_out'],gnn_param['dim_out'])
        if 'combine' in gnn_param and gnn_param['combine'] == 'rnn':
            self.combiner = torch.nn.RNN(gnn_param['dim_out'], gnn_param['dim_out'])
            
    def empty_cache(self):
        if 'historical' in self.gnn_param and self.gnn_param['historical'] == True:
            for l in range(0,self.gnn_param['layer']):
                self.historical_cache['l'+str(l)].empty()
        if self.memory_param['type'] == 'node':
            self.memory_updater.empty_cache()
    
    def get_sub_block(self,block,src_index):
        all_dst_nodes = src_index
        src,dst,eids = block.in_edges(src_index,form = 'all')
        unq,ind = torch.cat((src,dst,all_dst_nodes)).unique(return_inverse=True)
        subgraph =  dgl.create_block((ind[:src.shape[0]],ind[src.shape[0]:src.shape[0]+dst.shape[0]]),
                                 num_src_nodes = int(unq.shape[0]),
                                 num_dst_nodes = int(ind[src.shape[0]:].max().item())+1,
                                 device = block.device)
        for k in block.srcdata:
            subgraph.srcdata[k] = block.srcdata[k][unq]
        for k in block.edata:
            subgraph.edata[k] = block.edata[k][eids]
        return subgraph
    @staticmethod
    def send_memory_async(metadata,memory = None):
        if memory is not None:
            local_memory = DistributedTensor(memory[metadata['src_pos_index']])
            dst_memory = memory[metadata['dst_pos_index']]
            dist_index0,ind0 = metadata['dist_src_index'].sort()
            send_ptr = local_memory.all_to_all_ind2ptr(dist_index0)
            mem = DistributedTensor(torch.empty_like(local_memory.accessor.data))
            mem.all_to_all_set(dst_memory[ind0],**send_ptr,)
            src_mem = local_memory.all_to_all_get(**send_ptr)[ind0]
            mem = mem.accessor.data
            return mem,src_mem
        else:
            return None,None
    def forward(self, mfgs, metadata = None,neg_samples=1, mode = 'triplet'):
        #torch.cuda.synchronize()
        t0 = time.time()
        #if(metadata['src_pos_index'].shape[0] == 0 or metadata['dst_pos_index'].shape[0] == 0 or metadata['dst_neg_index'].shape[0] == 0):
            #print(metadata['src_pos_index'].shape,metadata['dst_pos_index'].shape,metadata['dst_neg_index'].shape)
        if self.memory_param['type'] == 'node':
            self.memory_updater(mfgs[0])
            if (metadata is not None) and ('dist_src_index' in metadata):
                update_mem = self.memory_updater.last_updated_memory
                fut = executor.submit(GeneralModel.send_memory_async,metadata,update_mem)
        out = list()
        for l in range(self.gnn_param['layer']):
            for h in range(self.sample_param['history']):
                
                rst = self.layers['l' + str(l) + 'h' + str(h)](mfgs[l][h])
                
                if 'time_transform' in self.gnn_param and self.gnn_param['time_transform'] == 'JODIE':
                    rst = self.layers['l0h' + str(h) + 't'](rst, mfgs[l][h].srcdata['mem_ts'], mfgs[l][h].srcdata['ts'])
                if l != self.gnn_param['layer'] - 1:
                    if 'historical' in self.gnn_param and self.gnn_param['historical'] is True:
                        local_mask = DistIndex(mfgs[l+1][h].srcdata['ID']).part == torch.distributed.get_rank()
                        with torch.no_grad():
                            historical_embedding,historical_ts = self.historical_cache['l'+str(l)].get_data(mfgs[l+1][h].srcdata['_ID'][~local_mask])
                            self.historical_cache['l'+str(l)].update(mfgs[l+1][h].srcdata['_ID'][local_mask],rst[local_mask],mfgs[l+1][h].srcdata['ts'][:mfgs[l][h].num_dst_nodes()][local_mask])
                        rst[~local_mask] = historical_embedding #+ self.history_embedding_time(historical_ts)
                    mfgs[l + 1][h].srcdata['h'] = rst
                else:
                    ##test using historical embedding for remote nodes
                    
                    if 'historical' in self.gnn_param and self.gnn_param['historical'] is True:
                        local_mask = DistIndex(mfgs[l][h].srcdata['ID'][:mfgs[l][h].num_dst_nodes()]).part == torch.distributed.get_rank()
                        with torch.no_grad():
                            history_embedding,historical_ts = self.historical_cache['l'+str(l)].get_data(mfgs[l][h].srcdata['_ID'][:mfgs[l][h].num_dst_nodes()][~local_mask])
                            self.historical_cache['l'+str(l)].update(mfgs[l][h].srcdata['_ID'][:mfgs[l][h].num_dst_nodes()][local_mask],rst[local_mask],mfgs[l][h].srcdata['ts'][:mfgs[l][h].num_dst_nodes()][local_mask])
                        rst[~local_mask] = self.historical_update(torch.cat((history_embedding ,self.history_embedding_time(historical_ts)),dim = 1))
                        #history_embedding + self.history_embedding_time(historical_ts)
                    
                    #######
                    out.append(rst)
        if self.sample_param['history'] == 1:
            out = out[0]
        else:
            out = torch.stack(out, dim=0)
            out = self.combiner(out)[0][-1, :, :]
        #metadata需要在前面去重的时候记一下id
        if self.gnn_param['use_src_emb'] or self.gnn_param['use_dst_emb']:
            self.embedding = out.detach().clone()
        else:
            self.embedding = None
        if self.gnn_param['dyrep']:
            out = self.memory_updater.last_updated_memory
        self.out = out
        if self.memory_param['type'] == 'node':
            if (metadata is not None) and ('dist_src_index' in metadata):
                with torch.no_grad():
                    self.dst_mem,self.src_mem = fut.result()
        #torch.cuda.synchronize()
        t1 = time.time()

        if metadata is not None:
            #out = torch.cat((out[metadata['dst_pos_pos']],out[metadata['src_id_pos']],out[metadata['dst_neg_pos']]),0)
            if 'dist_src_index' not in metadata:
                h_pos_src = out[metadata['src_pos_index']]
                h_pos_dst = out[metadata['dst_pos_index']]
                h_neg_dst = out[metadata['dst_neg_index']]
                if 'src_neg_index' in metadata:
                    h_neg_src = out[metadata['src_neg_index']]
                    return self.edge_predictor(h_pos_src, h_pos_dst, h_neg_src, h_neg_dst, neg_samples=neg_samples, mode = mode)
                else:
                    return self.edge_predictor(h_pos_src, h_pos_dst, None , h_neg_dst, neg_samples=neg_samples, mode = mode)
            else:
                if self.memory_param['type'] == 'node':
                    h_pos_src,h_pos_dst,h_neg_dst,mem,src_mem =  self.all_to_all_embedding(out,metadata,neg_samples,None,self.gnn_param['use_src_emb'])
                    #self.dst_mem = mem.detach().clone()
                    #self.src_mem = src_mem.detach().clone()
                else:
                    h_pos_src,h_pos_dst,h_neg_dst,mem,src_mem =  self.all_to_all_embedding(out,metadata,neg_samples,None,self.gnn_param['use_src_emb'])
                #torch.cuda.synchronize()
                t2 = time.time()
                time_count.add_train_forward_embedding(t1-t0)
                time_count.add_train_foward_all_to_all(t2-t1)
                return self.edge_predictor(h_pos_src, h_pos_dst, None, h_neg_dst, neg_samples=neg_samples, mode = mode)
        else:
            return out
    

    


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