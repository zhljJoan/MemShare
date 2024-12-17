
import time

import time
from os.path import abspath, join, dirname
import os
from starrygl.distributed.context import DistributedContext
from starrygl.module.Filter import Filter
from starrygl.sample.count_static import time_count
import sys
from os.path import abspath, join, dirname
from starrygl.distributed.utils import DistIndex

from starrygl.module.historical_cache import HistoricalCache
sys.path.insert(0, join(abspath(dirname(__file__))))
import torch
import dgl
from layers import TimeEncode
from torch_scatter import scatter
from starrygl.sample.count_static import time_count as tt


cnt_local = 0
cnt_all_node = 0
def get_cnt_local():
    global cnt_local
    print('local {}'.format(cnt_local))
    return cnt_local
def get_cnt_all_node():
    global cnt_all_node
    print('all {}'.format(cnt_all_node))
    return cnt_all_node
def add_cnt_local(x):
    global cnt_local
    cnt_local += x
def add_all_node(x):
    global cnt_all_node
    cnt_all_node += x
    
class GRUMemeoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat,cache_index = None):
        super(GRUMemeoryUpdater, self).__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.updater = torch.nn.GRUCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        self.delta_memory = 0
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
            
        if memory_param['combine_node_feature']:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)
        ###for test
        self.update_node_list = []
        self.grad_memory_list = []
        self.update_delta_t = []
    @staticmethod
    def get_cnt_local():
        global cnt_local
        return cnt_local
    @staticmethod
    def get_cnt_all_node():
        global cnt_all_node
        return cnt_all_node
    def empty_cache(self):
        pass
    def forward(self, mfg, param = None):
        t_s = tt.start()
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])

                
                b.srcdata['mem_input']= torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
                #print(b.srcdata['mem_input'])
            updated_memory = self.updater(b.srcdata['mem_input'], b.srcdata['mem'])
            #print('updated_memory {}'.format(updated_memory))
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            self.lasted_memory = updated_memory.detach().clone()
            if self.memory_param['combine_node_feature']:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] += updated_memory
                    else:
                        b.srcdata['h'] = updated_memory + self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = updated_memory

        tt.mem_update += tt.elapsed(t_s)
class RNNMemeoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat):
        super(RNNMemeoryUpdater, self).__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.updater = torch.nn.RNNCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        self.delta_memory = 0
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if memory_param['combine_node_feature']:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)

    def forward(self, mfg, param = None):
        for b in mfg:
            if self.dim_time > 0:
                #print(b.srcdata['ts'].shape,b.srcdata['mem_ts'].shape)
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
                b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
            updated_memory = self.updater(b.srcdata['mem_input'], b.srcdata['mem'])
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()

            if self.memory_param['combine_node_feature']:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] += updated_memory
                    else:
                        b.srcdata['h'] = updated_memory + self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = updated_memory
    def empty_cache(self):
        pass

# class TransformerMemoryUpdater(torch.nn.Module):

    # def __init__(self, memory_param, dim_in, dim_out, dim_time, train_param):
        # super(TransformerMemoryUpdater, self).__init__()
        # self.memory_param = memory_param
        # self.dim_time = dim_time
        # self.att_h = memory_param['attention_head']
        # if dim_time > 0:
            # self.time_enc = TimeEncode(dim_time)
        # self.w_q = torch.nn.Linear(dim_out, dim_out)
        # self.w_k = torch.nn.Linear(dim_in + dim_time, dim_out)
        # self.w_v = torch.nn.Linear(dim_in + dim_time, dim_out)
        # self.att_act = torch.nn.LeakyReLU(0.2)
        # self.layer_norm = torch.nn.LayerNorm(dim_out)
        # self.mlp = torch.nn.Linear(dim_out, dim_out)
        # self.dropout = torch.nn.Dropout(train_param['dropout'])
        # self.att_dropout = torch.nn.Dropout(train_param['att_dropout'])
        # self.last_updated_memory = None
        # self.last_updated_ts = None
        # self.last_updated_nid = None

    # def forward(self, mfg, param = None):
        # for b in mfg:
            # Q = self.w_q(b.srcdata['mem']).reshape((b.num_src_nodes(), self.att_h, -1))
            # mails = b.srcdata['mem_input'].reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], -1))
            # if self.dim_time > 0:
                # time_feat = self.time_enc(b.srcdata['ts'][:, None] - b.srcdata['mail_ts']).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], -1))
                # mails = torch.cat([mails, time_feat], dim=2)
            # K = self.w_k(mails).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], self.att_h, -1))
            # V = self.w_v(mails).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], self.att_h, -1))
            # att = self.att_act((Q[:,None,:,:]*K).sum(dim=3))
            # att = torch.nn.functional.softmax(att, dim=1)
            # att = self.att_dropout(att)
            # rst = (att[:,:,:,None]*V).sum(dim=1)
            # rst = rst.reshape((rst.shape[0], -1))
            # rst += b.srcdata['mem']
            # rst = self.layer_norm(rst)
            # rst = self.mlp(rst)
            # rst = self.dropout(rst)
            # rst = torch.nn.functional.relu(rst)
            # b.srcdata['h'] = rst
            # self.last_updated_memory = rst.detach().clone()
            # self.last_updated_nid = b.srcdata['ID'].detach().clone()
            # self.last_updated_ts = b.srcdata['ts'].detach().clone()
    # def empty_cache(self):
        # pass







class HistoricalMemeoryUpdater(torch.nn.Module):
    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat,updater,learnable = None, num_nodes=None):
        super(HistoricalMemeoryUpdater, self).__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        print(dim_hid,dim_in,dim_time,dim_node_feat)
        self.updater = updater(dim_in + dim_time, dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        self.delta_memory = 0
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
            
        if memory_param['combine_node_feature']:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)
        #if 'historical' in memory_param and memory_param['historical']:
        #    self.history_cache = HistoricalCache(cache_index,(cache_index.size,dim_hid),torch.float,torch.device('cuda'))
        #self.time_for_historical = torch.nn.Linear(dim_hid+dim_time,dim_hid)
        ###for test
        #self.update_node_list = []
        #self.grad_memory_list = []
        #self.update_delta_t = []
        self.learnable = learnable
        if learnable:
            self.gamma = torch.nn.Parameter(torch.tensor([0.5]),
                               requires_grad=True)
            #self.gamma = 0.5
            self.filter = Filter(n_nodes=num_nodes,
                           memory_dimension=self.dim_hid,
                           )
        else:
            self.gamma = 0.9
    @staticmethod
    def get_cnt_local():
        global cnt_local
        return cnt_local
    @staticmethod
    def get_cnt_all_node():
        global cnt_all_node
        return cnt_all_node
    def empty_cache(self):
        self.filter.clear()
        #if 'historical' in self.memory_param and self.memory_param['historical']:
        #    self.history_cache.empty()
    def forward(self, mfg, param = None):
        t_s = tt.start()
        for b in mfg:
            #print(' 0 {}'.format(b.srcdata['h'].shape))
            #if 'historical' in self.memory_param and self.memory_param['historical']:
            #    local_mask = DistIndex(b.srcdata['ID']).part == torch.distributed.get_rank()
            #else:
            #    local_mask = torch.ones(b.srcdata['ID'].shape[0],dtype = torch.bool,device = 'cuda')
            #mask = 
            if self.dim_time > 0:
                #print(b.srcdata['his_mem'])
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
                b.srcdata['mem_input']= torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
            updated_memory = self.updater(b.srcdata['mem_input'], b.srcdata['mem'])
            #memory = updated_memory
            if self.learnable:
                self.prev_memory = b.srcdata['his_mem']
                #memory = self.gamma * self.prev_memory + (1 - self.gamma)*updated_memory
#
                with torch.no_grad():
                    ctx = DistributedContext.get_default_context()
                    mask = ~(DistIndex(b.srcdata['ID']).part == ctx.memory_group_rank)
                    if mask.sum() > 0:
                        transition_dense = self.filter.get_incretment_remote(b.srcdata['ID'])
                    else:
                        transition_dense = self.filter.get_incretment(DistIndex(b.srcdata['ID']).loc)
                    transition_dense[DistIndex(b.srcdata['ID']).is_shared]*=2
                    if not (transition_dense.max().item() == 0):      
                        transition_dense -= transition_dense.min()
                        transition_dense /=transition_dense.max() 
                        transition_dense = 2*transition_dense - 1
                        #print(transition_dense)
                    pred_memory = self.prev_memory + transition_dense
                memory = self.gamma * pred_memory + (1- self.gamma) * updated_memory
                #if not (memory.max().item() == 0):     
                #        memory =memory- memory.min()
                #        memory =memory/ memory.max() 
                #        memory = 2*memory - 1
                #print(self.gamma)
                with torch.no_grad():
                    mask = ((~mask) & ~DistIndex(b.srcdata['ID']).is_shared)
                    change = memory.data.clone() - self.prev_memory.data.clone()
                    change.detach()
                    if not (change.max().item() == 0):     
                        change -= change.min()
                        change /=change.max() 
                        change = 2*change - 1
                        self.filter.update(DistIndex(b.srcdata['ID'][mask]).loc,change[mask])
            self.update_memory = memory
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            ##print('last memory update {}\n'.format(b.srcdata['ID']))
            ##new_memory = torch.zeros(b.srcdata['ID'].shape[0],updated_memory.shape[1],dtype = updated_memory.dtype, device = #updated_memory.device)
            ##new_memory[local_mask] = updated_memory
            ##if 'historical' in self.memory_param and self.memory_param['historical']:
            ##    self.history_cache.update(b.srcdata['_ID'][local_mask].detach().clone(),updated_memory.detach().clone(), b.srcdata#['ts'][local_mask].detach().clone())
            ##    if(local_mask.sum().item()!=b.srcdata['ID'].shape[0]):
            ##        with torch.no_grad():mail_
            ##            history_mem,history_ts = self.history_cache.get_data(b.srcdata['_ID'][~local_mask])
            ##        history_mem_ts = self.time_enc(b.srcdata['ts'][~local_mask] - history_ts)
#
            ##        new_memory[~local_mask] = self.time_for_historical(torch.cat((history_mem,history_mem_ts),dim = 1))
            self.lasted_memory = memory.detach().clone()
            if self.memory_param['combine_node_feature'] and self.dim_node_feat > 0:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] += memory
                    else:
                        b.srcdata['h'] = memory + self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = memory
            else:
                b.srcdata['h'] = memory
        tt.mem_update += tt.elapsed(t_s)


class TransformerMemoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_out, dim_time, train_param):
        super(TransformerMemoryUpdater, self).__init__()
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.att_h = memory_param['attention_head']
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        self.w_q = torch.nn.Linear(dim_out, dim_out)
        self.w_k = torch.nn.Linear(dim_in + dim_time, dim_out)
        self.w_v = torch.nn.Linear(dim_in + dim_time, dim_out)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
        self.mlp = torch.nn.Linear(dim_out, dim_out)
        self.dropout = torch.nn.Dropout(train_param['dropout'])
        self.att_dropout = torch.nn.Dropout(train_param['att_dropout'])
        

    def forward(self, b, param = None):
        Q = self.w_q(b.srcdata['mem']).reshape((b.num_src_nodes(), self.att_h, -1))
        mails = b.srcdata['mem_input'].reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], -1))
        #print(mails.shape,b.srcdata['mem_input'].shape,b.srcdata['mail_ts'].shape)
        if self.dim_time > 0:
            time_feat = self.time_enc(b.srcdata['ts'][:, None] - b.srcdata['mail_ts']).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], -1))
            #print(time_feat.shape)
            mails = torch.cat([mails, time_feat], dim=2)
            #print(mails.shape)
        K = self.w_k(mails).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], self.att_h, -1))
        V = self.w_v(mails).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], self.att_h, -1))
        att = self.att_act((Q[:,None,:,:]*K).sum(dim=3))
        att = torch.nn.functional.softmax(att, dim=1)
        att = self.att_dropout(att)
        rst = (att[:,:,:,None]*V).sum(dim=1)
        rst = rst.reshape((rst.shape[0], -1))
        rst += b.srcdata['mem']
        rst = self.layer_norm(rst)
        rst = self.mlp(rst)
        rst = self.dropout(rst)
        rst = torch.nn.functional.relu(rst)
        return rst
"""
(self,index,memory,memory_ts,
                              mail_index,mail,mail_ts,
                              reduce_Op=None,
                              async_op=True,
                              set_p2p=False,
                              mode=None,
                              filter=None,
                              wait_submit=True,
                              spread_mail=True,
                              update_cross_mm = False)
"""
class AsyncMemeoryUpdater(torch.nn.Module):
    
    def all_update_func(self,index,memory,memory_ts,mail_index,mail,mail_ts,nxt_fetch_func,spread_mail=False):
        self.mailbox.set_memory_all_reduce(
                                           index,memory,memory_ts,
                                           mail_index,mail,mail_ts,
                                           reduce_Op='max',async_op=False,
                                           mode='all_reduce',
                                           wait_submit=False,spread_mail=spread_mail,
                                           update_cross_mm=True,
                                           )
        #self.mailbox.set_memory_all_reduce(index,memory,memory_ts,mail,mail_ts,reduce_Op = 'max', async_op = False,filter=None,mode='all_reduce',set_remote=True)
        if nxt_fetch_func is not None:
            nxt_fetch_func()
    # def p2p_func(self,index,memory,memory_ts,mail,mail_ts,nxt_fetch_func,mail_index = None):
    #     self.mailbox.handle_last_async()
    #     submit_to_queue = False
    #     if nxt_fetch_func is not None:
    #         nxt_fetch_func()
    #         submit_to_queue = True
    #     self.mailbox.set_memory_all_reduce(index,memory,memory_ts,mail,mail_ts,reduce_Op = 'max', async_op = True,filter=None,mode=None,set_remote=True,submit = submit_to_queue)
    # def all_reduce_func(self,index,memory,memory_ts,mail,mail_ts,nxt_fetch_func,mail_index = None):
    #    self.mailbox.set_memory_all_reduce(index,memory,memory_ts,mail,mail_ts,reduce_Op = 'max', async_op = False,filter=None,mode='all_reduce',set_remote=False)
    #    if nxt_fetch_func is not None:
    #        nxt_fetch_func()
    def historical_func(self,index,memory,memory_ts,mail_index,mail,mail_ts,nxt_fetch_func,spread_mail=False):
        self.mailbox.sychronize_shared()
        self.mailbox.handle_last_async()
        submit_to_queue = False
        if nxt_fetch_func is not None:
            submit_to_queue = True
        self.mailbox.set_memory_all_reduce(
                                           index,memory,memory_ts,
                                           mail_index,mail,mail_ts,
                                           reduce_Op='max',async_op=True,
                                           mode='historical',
                                           wait_submit=submit_to_queue,spread_mail=spread_mail,
                                           update_cross_mm=False,
                                           )
        if nxt_fetch_func is not None:
            nxt_fetch_func()

    def local_func(self,index,memory,memory_ts,mail_index,mail,mail_ts,nxt_fetch_func,spread_mail=False):
        if nxt_fetch_func is not None:
            nxt_fetch_func()
    def transformer_updater(self,b):
        return self.ceil_updater(b)
    def rnn_updater(self,b):
        if self.dim_time > 0:
                #print(b.srcdata['ts'].shape,b.srcdata['mem_ts'].shape)
            time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
            b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
        return self.ceil_updater(b.srcdata['mem_input'], b.srcdata['mem'])
    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat,updater,mode = None,mailbox = None,train_param=None):
        super(AsyncMemeoryUpdater, self).__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        if memory_param['memory_update'] == 'transformer':
            self.ceil_updater = updater(memory_param, dim_in, dim_hid, dim_time, train_param)
            self.updater = self.transformer_updater
        else: 
            self.ceil_updater = updater(dim_in + dim_time, dim_hid)
            self.updater = self.rnn_updater
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        self.delta_memory = 0
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if memory_param['combine_node_feature']:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)
        self.mailbox = mailbox
        self.mode = mode
        if self.mode == 'all_update':
            self.update_hunk = self.all_update_func
        elif self.mode == 'p2p':
            self.update_hunk = self.p2p_func
        elif self.mode == 'all_reduce':
            self.update_hunk = self.all_reduce_func
        elif self.mode == 'historical':
            self.update_hunk = self.historical_func
        elif self.mode == 'local' or self.mode=='all_local':
            self.update_hunk = self.local_func
        if self.mode == 'historical':
            self.gamma = torch.nn.Parameter(torch.tensor([0.5]),
                               requires_grad=True)
            self.filter = Filter(n_nodes=mailbox.shared_nodes_index.shape[0],
                           memory_dimension=self.dim_hid,
                           )
        else:
            self.gamma = 1   
    def forward(self, mfg, param = None):
        for b in mfg:
            #print(b.srcdata['ID'].shape[0])
            updated_memory0 = self.updater(b)
            mask = DistIndex(b.srcdata['ID']).is_shared
            #incr = updated_memory[mask] - b.srcdata['mem'][mask]
            #time_feat = self.time_enc(b.srcdata['ts'][mask].reshape(-1,1) - b.srcdata['his_ts'][mask].reshape(-1,1))
            #his_mem = torch.cat((mail_input[mask],time_feat),dim = 1)
            with torch.no_grad():
                upd0 = torch.zeros_like(updated_memory0)
                if self.mode == 'historical':
                    shared_ind = self.mailbox.is_shared_mask[DistIndex(b.srcdata['ID'][mask]).loc]

                    transition_dense = b.srcdata['his_mem'][mask] + self.filter.get_incretment(shared_ind)
                    #print(transition_dense.shape)
                    if not (transition_dense.max().item() == 0):      
                        transition_dense -= transition_dense.min()
                        transition_dense /= transition_dense.max() 
                        transition_dense = 2*transition_dense - 1
                    upd0[mask] = transition_dense#b.srcdata['his_mem'][mask] + transition_dense
                    #print(self.gamma)
                    #print('tran {} {} {}\n'.format(transition_dense.max().item(),upd0[mask].max().item(),b.srcdata['his_mem'][mask].max().item()))
                else:
                    upd0[mask] = updated_memory0[mask]
            #upd0[mask] = self.ceil_updater(his_mem, b.srcdata['his_mem'][mask])
            #updated_memory = torch.where(mask.unsqueeze(1),self.gamma*updated_memory0 + (1-self.gamma)*(b.srcdata['his_mem'])
            # ,updated_memory0)
            updated_memory = torch.where(mask.unsqueeze(1),self.gamma*updated_memory0 + (1-self.gamma)*(upd0),updated_memory0)
            with torch.no_grad():
                if self.mode == 'historical':
                    change = updated_memory[mask] - b.srcdata['his_mem'][mask]
                    self.filter.update(shared_ind,change)
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            
            with torch.no_grad():
                if param is not None:
                    _,src,dst,ts,edge_feats,nxt_fetch_func = param
                    indx = torch.cat((src,dst))
                    index, memory, memory_ts = self.mailbox.get_update_memory(self.last_updated_nid[indx],
                                                                    self.last_updated_memory[indx],
                                                                    self.last_updated_ts[indx],
                                                                    None)
                    #print(index.shape[0])
                    if param[0]:
                        index0, mail, mail_ts = self.mailbox.get_update_mail(
                                                    b.srcdata['ID'],src,dst,ts,edge_feats,
                                                    self.last_updated_memory, 
                                                    None,False,False,block=b
                                                )
                    #print(index.shape[0])
                    if torch.distributed.get_world_size() == 0:
                        self.mailbox.mon.add(index,self.mailbox.node_memory.accessor.data[index],memory)
                    ##print(index.shape,memory.shape,memory_ts.shape,mail.shape,mail_ts.shape)
                    local_mask = (DistIndex(index).part==torch.distributed.get_rank())
                    local_mask_mail = (DistIndex(index0).part==torch.distributed.get_rank())
                    
                    self.mailbox.set_mailbox_local(DistIndex(index0[local_mask_mail]).loc,mail[local_mask_mail],mail_ts[local_mask_mail],Reduce_Op = 'max')
                    self.mailbox.set_memory_local(DistIndex(index[local_mask]).loc,memory[local_mask],memory_ts[local_mask], Reduce_Op = 'max')
                    is_deliver=(self.mailbox.deliver_to == 'neighbors')
                    self.update_hunk(index,memory,memory_ts,index0,mail,mail_ts,nxt_fetch_func,spread_mail= is_deliver)
            
            if self.memory_param['combine_node_feature'] and self.dim_node_feat > 0:
                if self.dim_node_feat == self.dim_hid:
                    b.srcdata['h'] += updated_memory
                else:
                    b.srcdata['h'] = updated_memory + self.node_feat_map(b.srcdata['h'])
            else:
                b.srcdata['h'] = updated_memory

    def empty_cache(self):
        if self.mode == 'historical':
            print('clear\n')
            self.filter.clear()
