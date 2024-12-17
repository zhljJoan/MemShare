import starrygl
from typing import Union
from typing import List
from typing import Optional
import torch
from torch.distributed import rpc
import torch_scatter
from starrygl import distributed
from starrygl.distributed.context import DistributedContext
from starrygl.distributed.utils import DistIndex, DistributedTensor
import torch.distributed as dist

from starrygl.module import historical_cache
from starrygl.sample.graph_core.utils import _get_pin
from starrygl.sample.memory.change import MemoryMoniter

#from starrygl.utils.uvm import cudaMemoryAdvise

class SharedMailBox():
    '''
    We will first define our mailbox, including our definitions of mialbox and memory:
    .. code-block:: python
        from starrygl.sample.memory.shared_mailbox import SharedMailBox
        mailbox = SharedMailBox(num_nodes=num_nodes, mefinishmory_param=memory_param, dim_edge_feat=dim_edge_feat)

    Args:
        num_nodes (int): number of nodes

        memory_param (dict): the memory parameters in the yaml file,refer to TGL

        dim_edge_feat (int): the dim of edge feature

        device (torch.device): the device used to store MailBox

        uvm (bool): 1-use uvm, 0-don't use uvm
    
    Examples:

        .. code-block:: python
        
            from starrygl.sample.part_utils.partition_tgnn import partition_load
            from starrygl.sample.memory.shared_mailbox import SharedMailBox

            pdata = partition_load("PATH/{}".format(dataname), algo="metis_for_tgnn")
            mailbox = SharedMailBox(pdata.ids.shape[0], memory_param, dim_edge_feat=pdata.edge_attr.shape[1] if pdata.edge_attr is not None else 0)

    We then need to hand over the mailbox to the data loader as in the above example, so that the relevant memory/mailbox can be directly loaded during training.

    During the training, we will call `get_update_memory`/`get_update_mail` function constantly updates 
    the relevant storage,which is the idea related to TGN.
    '''
    def __init__(self,
                 num_nodes,
                 memory_param,
                 dim_edge_feat,
                 shared_nodes_index = None,
                 device = torch.device('cuda'),
                 ts_dtye = torch.float32,
                 uvm = False,
                 use_pin = False,
                 start_historical = False,
                 shared_ssim = 2):
        ctx = distributed.context._get_default_dist_context()
        self.device = device
        self.num_nodes = num_nodes
        self.num_parts = dist.get_world_size()
        if memory_param['type'] != 'node':
            raise NotImplementedError
        self.memory_param = memory_param
        self.memory_size =  memory_param['dim_out']
        assert not (device.type =='cpu' and uvm is True),\
            'set uvm must set device on cuda'
            
        memory_device = device
        if device.type == 'cuda' and uvm is True:
            memory_device = torch.device('cpu')
        node_memory = torch.zeros((
            self.num_nodes, memory_param['dim_out']), 
            dtype=torch.float32,device =memory_device)
        node_memory_ts = torch.zeros(self.num_nodes,
                                      dtype=ts_dtye,
                                      device = self.device)
        mailbox = torch.zeros(self.num_nodes, 
                              memory_param['mailbox_size'], 
                              2 * memory_param['dim_out'] + dim_edge_feat,
                              device = memory_device, dtype=torch.float32)
        mailbox_ts = torch.zeros((self.num_nodes, 
                                  memory_param['mailbox_size']), 
                                dtype=ts_dtye,device  = self.device)
        self.mailbox_shape = len(mailbox[0,:].reshape(-1))
        self.node_memory = DistributedTensor(node_memory)
        self.node_memory_ts = DistributedTensor(node_memory_ts)
        self.mailbox = DistributedTensor(mailbox)
        self.mailbox_ts = DistributedTensor(mailbox_ts)
        self.next_mail_pos = torch.zeros((self.num_nodes), 
                                         dtype=torch.long,
                                         device = self.device) 
        self.tot_comm_count  = 0
        self.tot_shared_count = 0
        self.shared_nodes_index = None
        self.deliver_to = memory_param['deliver_to'] if 'deliver_to' in memory_param else 'self'
        if shared_nodes_index is not None:
            self.shared_nodes_index = shared_nodes_index.to('cuda:{}'.format(ctx.local_rank))
            self.is_shared_mask = torch.zeros(self.num_nodes,dtype=torch.int,device=torch.device('cuda:{}'.format(ctx.local_rank)))-1
            self.is_shared_mask[shared_nodes_index] = torch.arange(self.shared_nodes_index.shape[0],dtype=torch.int,
                                                                   device=torch.device('cuda:{}'.format(ctx.local_rank)))
        
        if start_historical:
            self.historical_cache = historical_cache.HistoricalCache(self.shared_nodes_index,0,self.node_memory.shape[1],self.node_memory.dtype,self.node_memory.device,threshold=shared_ssim)
        else:
            self.historical_cache = None
        self._mem_pin = {}
        self._mail_pin = {}
        self.use_pin = use_pin
        self.last_memory_sync = None
        self.last_mail_sync = None
        self.next_wait_memory_job=None
        self.next_wait_mail_job=None
        self.next_wait_gather_memory_job=None

        self.mon = MemoryMoniter()

    def reset(self):
        self.node_memory.accessor.data.zero_()
        self.node_memory_ts.accessor.data.zero_()
        self.mailbox.accessor.data.zero_()
        self.mailbox_ts.accessor.data.zero_()
        self.next_mail_pos.zero_()
        if self.historical_cache is not None:
            self.historical_cache.empty()
        self.last_memory_sync = None
        self.last_mail_sync = None
    

    def set_memory_local(self,index,source,source_ts,Reduce_Op = None):
        if Reduce_Op == 'max' and self.num_parts > 1:
            unq_id,inv = index.unique(return_inverse = True)
            max_ts,id =  torch_scatter.scatter_max(source_ts,inv,dim=0)
            source_ts = max_ts
            source = source[id]
            index = unq_id
        self.node_memory.accessor.data[index] = source
        self.node_memory_ts.accessor.data[index] = source_ts.float()

    def set_mailbox_local(self,index,source,source_ts,Reduce_Op = None):
        if Reduce_Op == 'max' and self.num_parts > 1:
            unq_id,inv = index.unique(return_inverse = True)
            max_ts,id =  torch_scatter.scatter_max(source_ts,inv,dim=0)
            source_ts = max_ts
            source = source[id]
            index = unq_id
        #print(self.next_mail_pos[index])
        self.mailbox_ts.accessor.data[index, self.next_mail_pos[index]] = source_ts
        self.mailbox.accessor.data[index, self.next_mail_pos[index]] = source
        if self.memory_param['mailbox_size'] > 1:
            self.next_mail_pos[index] = torch.remainder(
                self.next_mail_pos[index] + 1, 
                self.memory_param['mailbox_size'])
    def get_update_mail(self,dist_indx_mapper,
                 src,dst,ts,edge_feats,
                 memory,embedding=None,use_src_emb=False,use_dst_emb=False,
                 block = None,Reduce_score=None,):
        if edge_feats is not None:
            edge_feats = edge_feats.to(self.device).to(self.mailbox.dtype)
        src = src.to(self.device)
        dst = dst.to(self.device)
        index = torch.cat([src, dst]).reshape(-1)
        index = dist_indx_mapper[index]
        mem_src = memory[src]
        mem_dst = memory[dst]
        if embedding is not None:
            emb_src = embedding[src]
            emb_dst = embedding[dst]
        src_mail = torch.cat([emb_src if use_src_emb else mem_src, emb_dst if use_dst_emb else mem_dst], dim=1)
        dst_mail = torch.cat([emb_dst if use_src_emb else mem_dst, emb_src if use_dst_emb else mem_src], dim=1)
        if edge_feats is not None:
            src_mail = torch.cat([src_mail, edge_feats], dim=1)
            dst_mail = torch.cat([dst_mail, edge_feats], dim=1)
        mail = torch.cat([src_mail, dst_mail], dim=0)
        mail_ts = torch.cat((ts,ts),-1).to(self.device).to(self.mailbox_ts.dtype)
            #print(mail_ts)
        #print(self.deliver_to)
        if self.deliver_to == 'neighbors':
            
            assert block is not None and Reduce_score is None
            root = torch.cat([src,dst]).reshape(-1)
            _,idx = torch_scatter.scatter_max(mail_ts,root,0)
            mail = torch.cat([mail, mail[idx[block.edges()[0].long()]]],dim=0)
            mail_ts = torch.cat([mail_ts, mail_ts[idx[block.edges()[0].long()]]], dim=0)
            index = torch.cat([index,block.dstdata['ID'][block.edges()[1].long()]],dim=0)

        if Reduce_score is not None:
            Reduce_score = torch.cat((Reduce_score,Reduce_score),-1).to(self.device)
        if Reduce_score is None:
            unq_index,inv = torch.unique(index,return_inverse = True)
            max_ts,idx = torch_scatter.scatter_max(mail_ts,inv,0)
            mail_ts = max_ts
            mail = mail[idx]
            index = unq_index
        else:
            unq_index,inv = torch.unique(index,return_inverse = True)
            print(inv.shape,Reduce_score.shape)
            max_score,idx = torch_scatter.scatter_max(Reduce_score,inv,0)
            mail_ts = mail_ts[idx]
            mail = mail[idx]
            index = unq_index
        #print('mail {} {}\n'.format(index.shape,mail.shape,mail_ts.shape))
        return index,mail,mail_ts
    
    def get_update_memory(self,index,memory,memory_ts,embedding):
        unq_index,inv = torch.unique(index,return_inverse = True)
        max_ts,idx = torch_scatter.scatter_max(memory_ts,inv,0)
        ts = max_ts
        index = unq_index
        memory = memory[idx]
        #print('memory {} {}\n'.format(index.shape,memory.shape,ts.shape)) 
        return index,memory,ts
    
    def pack(self,memory=None,memory_ts=None,mail=None,mail_ts=None,index = None,mode=None):
        if memory is not None and mail is not None:
            mem = torch.cat((memory,memory_ts.view(-1,1),mail,mail_ts.view(-1,1)),dim = 1)
        elif mail is not None:
            mem = torch.cat((mail,mail_ts.view(-1,1)),dim = 1)
        else:
            mem = torch.cat((memory,memory_ts.view(-1,1)),dim = 1)
        return mem
    
    def unpack(self,mem,mailbox = False):
        if mem.shape[1]  == self.node_memory.shape[1] + 1 or mem.shape[1]  == self.mailbox.shape[2] + 1 :
            mail = mem[:,: -1]
            mail_ts = mem[:,-1].view(-1)
            return mail,mail_ts
        elif mailbox is False:
            memory = mem[:,:self.node_memory.shape[1]]
            memory_ts = mem[:,self.node_memory.shape[1]].view(-1)
            mail = mem[:,self.node_memory.shape[1]+1:-1]
            mail_ts = mem[:,-1].view(-1)
            return memory,memory_ts,mail,mail_ts
        else:
            memory = mem[:,:self.node_memory.shape[1]]
            memory_ts = mem[:,self.node_memory.shape[1]].view(-1)
            mail = mem[:,self.node_memory.shape[1]+1:mem.shape[1]-self.mailbox_ts.shape[1]].reshape(mem.shape[0],self.mailbox.shape[1],self.mailbox.shape[2])
            mail_ts = mem[:,mem.shape[1]-self.mailbox_ts.shape[1]:]
            return memory,memory_ts,mail,mail_ts
    """
        sychronization last async all-to-all M&M communication task
    """
    def handle_last_memory(self,reduce_Op=None,):
        if self.last_memory_sync is not None:
            gather_id_list,handle0,gather_memory,handle1 = self.last_memory_sync
            self.last_memory_sync = None
            handle0.wait()
            handle1.wait()
            if isinstance(gather_memory,list):
                gather_memory = torch.cat(gather_memory,dim = 0)
            if gather_memory.shape[1] > self.node_memory.shape[1] + 1:
                gather_memory,gather_memory_ts,gather_mail,gather_mail_ts = self.unpack(gather_memory)
                self.set_mailbox_local(DistIndex(gather_id_list).loc,gather_mail,gather_mail_ts,Reduce_Op = reduce_Op)
            else:
                gather_memory,gather_memory_ts = self.unpack(gather_memory)
            #print(gather_id_list.shape,gather_memory.shape,gather_memory_ts.shape)
            self.set_memory_local(DistIndex(gather_id_list).loc,gather_memory,gather_memory_ts, Reduce_Op = reduce_Op)
    def handle_last_mail(self,reduce_Op=None,):
        if self.last_mail_sync is not None:
            gather_id_list,handle0,gather_memory,handle1 = self.last_mail_sync
            self.last_mail_sync = None
            handle0.wait()
            handle1.wait()
            if isinstance(gather_memory,list):
                gather_memory = torch.cat(gather_memory,dim = 0)
            gather_memory,gather_memory_ts = self.unpack(gather_memory)
            #print(gather_id_list.shape,gather_memory.shape,gather_memory_ts.shape)
            self.set_mailbox_local(DistIndex(gather_id_list).loc,gather_memory,gather_memory_ts, Reduce_Op = reduce_Op)
    def handle_last_async(self,reduce_Op = None):
       self.handle_last_memory(reduce_Op)
       self.handle_last_mail(reduce_Op)
        
    """
        sychronization last async all-gather memory communication task
    """
    def sychronize_shared(self):
        if self.historical_cache is None:
            return
        out=self.historical_cache.synchronize_shared_update()
        if out is not None:
            shared_index,shared_data,shared_ts = out
            index = self.shared_nodes_index[shared_index]
            mask= (shared_ts > self.node_memory_ts.accessor.data[index])
            self.node_memory.accessor.data[index][mask] = shared_data[mask]
            self.node_memory_ts.accessor.data[index][mask] = shared_ts[mask]

    def update_shared(self):
        ctx = DistributedContext.get_default_context()
        if self.next_wait_gather_memory_job is not None:
            shared_list,mem,shared_id_list,shared_memory_ind = self.next_wait_gather_memory_job
            self.next_wait_gather_memory_job = None
            handle0 = dist.all_gather(shared_list,mem,group=ctx.memory_nccl_group,async_op=True)
            handle1 = dist.all_gather(shared_id_list,shared_memory_ind,group=ctx.memory_nccl_group,async_op=True)
            self.historical_cache.add_shared_to_queue(handle0,handle1,shared_id_list,shared_list)
    def update_p2p_mem(self):
        if self.next_wait_memory_job is None:
            return
        index,gather_id_list,mem,gather_memory,input_split,output_split,group,async_op = self.next_wait_memory_job
        self.next_wait_memory_job = None
        handle0 = torch.distributed.all_to_all_single(
            gather_id_list,index,output_split_sizes=output_split, 
            input_split_sizes=input_split,group = group,async_op=async_op)
        handle1 = torch.distributed.all_to_all_single(
            gather_memory,mem,
            output_split_sizes=output_split, 
            input_split_sizes=input_split,group = group,async_op=async_op)
        self.last_memory_sync = (gather_id_list,handle0,gather_memory,handle1)
    def update_p2p_mail(self):
        if self.next_wait_mail_job is None:
            return
        index,gather_id_list,mem,gather_memory,input_split,output_split,group,async_op = self.next_wait_mail_job
        self.next_wait_mail_job = None
        #print(index,gather_id_list)
        handle0 = torch.distributed.all_to_all_single(
            gather_id_list,index,output_split_sizes=output_split, 
            input_split_sizes=input_split,group = group,async_op=async_op)
        handle1 = torch.distributed.all_to_all_single(
            gather_memory,mem,
            output_split_sizes=output_split, 
            input_split_sizes=input_split,group = group,async_op=async_op)
        self.last_mail_sync = (gather_id_list,handle0,gather_memory,handle1)
    """
    submit: take the task request into queue wait for start
    """
    def build_all_to_all_route(self,index,mm,mm_ts,is_no_redundant):
        ctx = DistributedContext.get_default_context()
        gather_len_list = torch.empty([self.num_parts],dtype=int,device=self.device)
        ind = torch.ops.torch_sparse.ind2ptr(DistIndex(index).part,self.num_parts)
        scatter_len_list = ind[1:] - ind[0:-1]
        torch.distributed.all_to_all_single(gather_len_list,scatter_len_list,group = ctx.memory_nccl_group)
        input_split = scatter_len_list.tolist()
        output_split = gather_len_list.tolist()
        if is_no_redundant:
            gather_ts = torch.empty(
                [gather_len_list.sum()],
                dtype = mm_ts.dtype,device = self.device
            )
            gather_id = torch.empty(
                [gather_len_list.sum()],
                dtype = index.dtype,device = self.device
            )
            torch.distributed.all_to_all_single(gather_id,index,
                                                output_split_sizes=output_split,
                                                input_split_sizes = input_split,
                                                group = ctx.memory_nccl_group,
                                                )
            torch.distributed.all_to_all_single(gather_ts,mm_ts,
                                                output_split_sizes=output_split,
                                                input_split_sizes = input_split,
                                                group = ctx.memory_nccl_group,
                                                )
            unq_id,inv = gather_id.unique(return_inverse=True)
            max_ts,pos = torch_scatter.scatter_max(gather_ts,inv,dim = 0)
            is_used = torch.zeros(gather_ts.shape,device=gather_ts.device,dtype=torch.int8)
            is_used[pos] = 1
            send_mm_to_dst = torch.zeros([scatter_len_list.sum().item()],device = is_used.device,dtype=torch.int8)
            torch.distributed.all_to_all_single(send_mm_to_dst,is_used,
                                                output_split_sizes = input_split,
                                                input_split_sizes = output_split,
                                                group = ctx.memory_nccl_group,
                                                )
            index = index[send_mm_to_dst>0]
            mm = mm[send_mm_to_dst>0]
            mm_ts = mm_ts[send_mm_to_dst>0]
            gather_len_list = torch.empty([self.num_parts],dtype=int,device=self.device)
            ind = torch.ops.torch_sparse.ind2ptr(DistIndex(index).part,self.num_parts)
            scatter_len_list = ind[1:] - ind[0:-1]
            torch.distributed.all_to_all_single(gather_len_list,scatter_len_list,group = ctx.memory_nccl_group)
            input_split = scatter_len_list.tolist()
            output_split = gather_len_list.tolist()
            gather_id = gather_id[is_used>0]
        else:
            gather_id = torch.empty([gather_len_list.sum()],dtype = index.dtype,device = self.device)
        gather_memory = torch.empty(
            [gather_len_list.sum(),mm.shape[1]],
            dtype = mm.dtype,device=self.device
        )
        return index,gather_id,mm,gather_memory,input_split,output_split
    
    def build_all_to_all_async_task(self,index,mm,mm_ts,is_no_redundant=False,is_async=True):
        ctx = DistributedContext.get_default_context()
        p2p_async_info = self.build_all_to_all_route(index,mm,mm_ts,is_no_redundant)
        if mm.shape[1] != self.mailbox.shape[-1] + 1:
            self.next_wait_memory_job = (*p2p_async_info,ctx.memory_nccl_group,is_async)
        else:
            self.next_wait_mail_job = (*p2p_async_info,ctx.memory_nccl_group,is_async)
    
    def set_memory_all_reduce(self,index,memory,memory_ts,
                              mail_index,mail,mail_ts,
                              reduce_Op=None,
                              async_op=True,
                              mode=None,    
                              wait_submit=True,
                              spread_mail=True,
                              update_cross_mm = False,
                              is_no_redundant = True):
        if self.num_parts == 1:
            return
        if not spread_mail and not update_cross_mm:
            pass
        else:
            if spread_mail:
                mm = torch.cat((mail,mail_ts.reshape(-1,1)),dim=1)
                mm_ts = mail_ts
                self.build_all_to_all_async_task(mail_index,mm,mm_ts,is_async=True,is_no_redundant=False)
                
                if update_cross_mm:
                    mm = torch.cat((memory,memory_ts.reshape(-1,1)),dim=1)
                    mm_ts = memory_ts
                    self.build_all_to_all_async_task(index,mm,mm_ts,is_async=True,is_no_redundant=False)
            else:
                if update_cross_mm:
                    mm = self.pack(memory,memory_ts,mail,mail_ts,index)
                    mm_ts = memory_ts
                    self.build_all_to_all_async_task(index,mm,mm_ts,is_async=True,is_no_redundant=False)
            if async_op is False:
                self.update_p2p_mail()
                self.update_p2p_mem()
                self.handle_last_async()

        ctx = DistributedContext.get_default_context()
        

        if self.shared_nodes_index is not None and (mode == 'all_reduce' or mode == 'historical'):
            shared_memory_ind = self.is_shared_mask[torch.min(DistIndex(index).loc,torch.tensor([self.num_nodes-1],device=index.device))]
            mask = ((shared_memory_ind>-1)&(DistIndex(index).part==ctx.memory_group_rank))
            shared_memory_ind = shared_memory_ind[mask]
            shared_memory = memory[mask]
            shared_memory_ts = memory_ts[mask]
            if spread_mail:
                shared_mail_indx = self.is_shared_mask[torch.min(DistIndex(mail_index).loc,torch.tensor([self.num_nodes-1],device=index.device))]
                mask = ((shared_mail_indx>-1)&(DistIndex(shared_mail_indx).part==ctx.memory_group_rank))
                shared_mail_indx = shared_mail_indx[mask]
            else:
                shared_mail_indx = shared_memory_ind
            shared_mail = mail[mask]
            shared_mail_ts = mail_ts[mask]
            if mode == 'historical':
                update_index = self.historical_cache.historical_check(shared_memory_ind,shared_memory,shared_memory_ts)
                shared_memory_ind = shared_memory_ind[update_index]
                shared_memory = shared_memory[update_index]
                shared_memory_ts = shared_memory_ts[update_index]
                mem = self.pack(memory=shared_memory,memory_ts=shared_memory_ts,index=shared_memory_ind,mode=mode)    
            else:
                if not spread_mail:
                    mem = self.pack(memory=shared_memory,memory_ts=shared_memory_ts,mail=shared_mail,mail_ts = shared_mail_ts,index=shared_memory_ind,mode=mode)
                else:
                    mem = self.pack(memory=shared_memory,memory_ts=shared_memory_ts,index=shared_memory_ind,mode=mode)
            self.tot_shared_count += shared_memory_ind.shape[0]
            broadcast_len = torch.empty([1],device = mem.device,dtype = torch.int)
            broadcast_len[0] = shared_memory_ind.shape[0]
            shared_len = [torch.empty([1],device = mem.device,dtype = torch.int) for _ in range(ctx.memory_group_size)] 
            dist.all_gather(shared_len,broadcast_len,group = ctx.memory_nccl_group)
            shared_list = [torch.empty([l.item(),mem.shape[1]],device = mem.device,dtype=mem.dtype) for l in shared_len]
            shared_id_list = [torch.empty([l.item()],device = shared_memory_ind.device,dtype=shared_memory_ind.dtype) for l in shared_len]
            if mode == 'all_reduce':
                dist.all_gather(shared_list,mem,group=ctx.memory_nccl_group)
                dist.all_gather(shared_id_list,shared_memory_ind,group=ctx.memory_nccl_group)
                mem = torch.cat(shared_list,dim = 0)
                shared_index = torch.cat(shared_id_list)
                if spread_mail:
                    shared_memory,shared_memory_ts  = self.unpack(mem)
                    unq_index,inv = torch.unique(shared_index,return_inverse = True)
                    max_ts,idx = torch_scatter.scatter_max(shared_memory_ts,inv,0)
                    shared_memory = shared_memory[idx]
                    shared_memory_ts = shared_memory_ts[idx]
                    shared_index = unq_index
                    
                    #self.historical_cache.local_historical_data[shared_index] = shared_memory
                    #self.historical_cache.local_ts[shared_index] = shared_memory_ts
                    broadcast_len = torch.empty([1],device = mem.device,dtype = torch.int)
                    broadcast_len[0] = shared_mail_indx.shape[0]
                    shared_len = [torch.empty([1],device = mail.device,dtype = torch.int) for _ in range(ctx.memory_group_size)] 
                    dist.all_gather(shared_len,broadcast_len,group = ctx.memory_nccl_group)
                    mail = self.pack(memory=shared_mail,memory_ts=shared_mail_ts,index=shared_mail_indx,mode=mode)
                    shared_mail_list = [torch.empty([l.item(),mail.shape[1]],device = mail.device,dtype=mail.dtype) for l in shared_len]
                    shared_mail_id_list = [torch.empty([l.item()],device = shared_mail_indx.device,dtype=shared_mail_indx.dtype) for l in shared_len]
                    #print(mail.shape)
                    dist.all_gather(shared_mail_list,mail,group=ctx.memory_nccl_group)
                    dist.all_gather(shared_mail_id_list,shared_mail_indx,group=ctx.memory_nccl_group)
                    shared_mail_indx = torch.cat(shared_mail_id_list,dim=0)
                    mail = torch.cat(shared_mail_list,dim=0)
                    shared_mail,shared_mail_ts = self.unpack(mail)
                    unq_index,inv = torch.unique(shared_mail_indx,return_inverse = True)
                    max_ts,idx = torch_scatter.scatter_max(shared_mail_ts,inv,0)
                    shared_mail= shared_mail[idx]
                    shared_mail_ts = shared_mail_ts[idx]
                    shared_mail_indx = unq_index
                    
                else:
                    shared_memory,shared_memory_ts,shared_mail,shared_mail_ts = self.unpack(mem)
                    unq_index,inv = torch.unique(shared_index,return_inverse = True)
                    max_ts,idx = torch_scatter.scatter_max(shared_memory_ts,inv,0)
                    shared_memory = shared_memory[idx]
                    #shared_memory = shared_memory[idx]
                    shared_memory_ts = shared_memory_ts[idx]
                    shared_mail_ts = shared_mail_ts[idx]
                    shared_mail = shared_mail[idx]
                    shared_index = unq_index
                    shared_mail_indx = unq_index
                self.set_memory_local(self.shared_nodes_index[shared_index],shared_memory,shared_memory_ts)
                #self.historical_cache.local_historical_data[shared_index] = shared_memory
                #self.historical_cache.local_ts[shared_index] = shared_memory_ts
                self.set_mailbox_local(self.shared_nodes_index[shared_mail_indx],shared_mail,shared_mail_ts)
            else:
                self.next_wait_gather_memory_job = (shared_list,mem,shared_id_list,shared_memory_ind)

            if not wait_submit:
                self.update_shared()
                self.update_p2p_mail()
                self.update_p2p_mem()
                #self.handle_last_async()
                #self.sychronize_shared()
                #self.historical_cache.add_shared_to_queue(handle0,handle1,shared_id_list,shared_list)
            """
            shared_memory = self.node_memory.accessor.data[self.shared_nodes_index]
            shared_memory_ts = self.node_memory_ts.accessor.data[self.shared_nodes_index]
            shared_mail = self.mailbox.accessor.data[self.shared_nodes_index]
            shared_mail_ts = self.mailbox_ts.accessor.data[self.shared_nodes_index]
            torch.distributed.all_reduce(shared_memory,group = ctx.memory_nccl_group,async_op = async_op,op=dist.ReduceOp.SUM)
            torch.distributed.all_reduce(shared_memory_ts,group = ctx.memory_nccl_group,async_op = async_op,op=dist.ReduceOp.SUM)
            torch.distributed.all_reduce(shared_mail,group = ctx.memory_nccl_group,async_op = async_op,op=dist.ReduceOp.SUM)
            torch.distributed.all_reduce(shared_mail_ts,group = ctx.memory_nccl_group,async_op = async_op,op=dist.ReduceOp.SUM)
            self.node_memory.accessor.data[self.shared_nodes_index]=shared_memory/ctx.memory_group_size
            self.node_memory_ts.accessor.data[self.shared_nodes_index]=shared_memory_ts/ctx.memory_group_size
            self.mailbox.accessor.data[self.shared_nodes_index]=shared_mail/ctx.memory_group_size
            self.mailbox_ts.accessor.data[self.shared_nodes_index]=shared_mail_ts/ctx.memory_group_size     
            """
        
    
    def set_mailbox_all_to_all_empty(self,index,memory,
                               memory_ts,mail,mail_ts,
                               reduce_Op = None,group = None):
        pass

        
    def set_mailbox_all_to_all(self,index,memory,
                               memory_ts,mail,mail_ts,
                               reduce_Op = None,group = None,async_op=False,submit = True):
        pass
        # #futs: List[torch.futures.Future] = []
        # if self.num_parts == 1:
        #     dist_index = DistIndex(index)
        #     index = dist_index.loc
        #     self.set_mailbox_local(index,mail,mail_ts)
        #     self.set_memory_local(index,memory,memory_ts)
        # else:
        #     self.tot_comm_count += (DistIndex(index).part != dist.get_rank()).sum()
        #     gather_len_list = torch.empty([self.num_parts],
        #                                   dtype = int,
        #                                   device = self.device)
        #     indic = torch.ops.torch_sparse.ind2ptr(DistIndex(index).part, self.num_parts)
        #     scatter_len_list = indic[1:] - indic[0:-1]
        #     torch.distributed.all_to_all_single(gather_len_list,scatter_len_list,group = group)
        #     input_split = scatter_len_list.tolist()
        #     output_split = gather_len_list.tolist()
        #     mem = self.pack(memory,memory_ts,mail,mail_ts)
        #     gather_memory = torch.empty(
        #         [gather_len_list.sum(),mem.shape[1]],
        #         dtype = memory.dtype,device = self.device)
        #     gather_id_list = torch.empty([gather_len_list.sum()],dtype = torch.long,device = self.device)
        #     input_split = scatter_len_list.tolist()
        #     output_split = gather_len_list.tolist()
        #     if async_op == True:
        #         self.last_job =  index,gather_id_list,mem,gather_memory,input_split,output_split,group,async_op 
        #         if not submit:
        #             self.update_p2p()
        #     else:
        #         torch.distributed.all_to_all_single(
        #             gather_id_list,index,output_split_sizes=output_split, 
        #             input_split_sizes=input_split,group = group,async_op=async_op)
        #         torch.distributed.all_to_all_single(
        #             gather_memory,mem,
        #             output_split_sizes=output_split, 
        #             input_split_sizes=input_split,group = group)
        #         if gather_memory.shape[1] > self.node_memory.shape[1] + 1:
        #             gather_memory,gather_memory_ts,gather_mail,gather_mail_ts = self.unpack(gather_memory)
        #             self.set_mailbox_local(DistIndex(gather_id_list).loc,gather_mail,gather_mail_ts,Reduce_Op = reduce_Op)
        #         else:
        #             gather_memory,gather_memory_ts = self.unpack(gather_memory)
        #         self.set_memory_local(DistIndex(gather_id_list).loc,gather_memory,gather_memory_ts, Reduce_Op = reduce_Op)

    def gather_memory(
            self,
            dist_index: Union[torch.Tensor, DistIndex, None] = None,
            send_ptr: Optional[List[int]] = None,
            recv_ptr: Optional[List[int]] = None,
            recv_ind: Optional[List[int]] = None,
            group = None,is_async=False,
        ):
        if dist_index is None:
            return self.node_memory.all_to_all_get(dist_index,send_ptr,recv_ptr,recv_ind,group=group,is_async=is_async),\
                self.node_memory_ts.all_to_all_get(dist_index,send_ptr,recv_ptr,recv_ind,group=group,is_async=is_async),\
                self.mailbox.all_to_all_get(dist_index,send_ptr,recv_ptr,recv_ind,group=group,is_async=is_async),\
                self.mailbox_ts.all_to_all_get(dist_index,send_ptr,recv_ptr,recv_ind,group=group,is_async=is_async)
        else:
            ids = self.node_memory.all_to_all_ind2ptr(dist_index)
            return self.node_memory.all_to_all_get(**ids,group = group,is_async=is_async),\
                self.node_memory_ts.all_to_all_get(**ids,group = group,is_async=is_async),\
                self.mailbox.all_to_all_get(**ids,group = group,is_async=is_async),\
                self.mailbox_ts.all_to_all_get(**ids,group = group,is_async=is_async)
            
    def gather_local_memory(
            self,
            dist_index: Union[torch.Tensor, DistIndex, None] = None,
            compute_device = torch.device('cuda')
        ):
        local_index = DistIndex(dist_index).loc
        rows = local_index.shape[0]
        #print(local_index.max(),local_index.min(),self.num_nodes)
        if self.node_memory.device.type == 'cpu' and self.use_pin:
            mem_pin = self._get_mem_pin(0,rows)
            mail_pin = self._get_mail_pin(0,rows)
            torch.index_select(self.node_memory.accessor.data,0,local_index,mem_pin)
            torch.index_select(self.mailbox.accessor.data,0,local_index,mail_pin)
            return mem_pin.to(compute_device,non_blocking=True),self.node_memory_ts[local_index.to('cpu')].to(compute_device),\
                    mail_pin.to(compute_device,non_blocking=True),self.node_memory_ts[local_index.to('cpu')].to(compute_device),
        return self.node_memory.accessor.data[local_index.to(self.device)].to(compute_device),\
                self.node_memory_ts.accessor.data[local_index.to(self.device)].to(compute_device),\
                self.mailbox.accessor.data[local_index.to(self.device)].to(compute_device),\
                self.mailbox_ts.accessor.data[local_index.to(self.device)].to(compute_device)

    def  _get_mem_pin(self, layer: int, rows: int) -> torch.Tensor:
        return _get_pin(self._mem_pins, layer, rows, self.node_memory.shape[1:])
    def  _get_mail_pin(self, layer: int, rows: int) -> torch.Tensor:
        return _get_pin(self._mail_pins, layer, rows, self.mailbox.shape[1:])
    """
    def set_memory_async(self,index,source,source_ts):
        dist_index = DistIndex(index)
        part_idx = dist_index.part
        index = dist_index.loc
        futs: List[torch.futures.Future] = []
        if self.num_parts == 1:
            self.set_memory_local(index,source,source_ts)
        for i in range(self.num_parts):
            fut = self.ctx.remote_call(
                SharedMailBox.set_memory_local,
                self.rrefs[i],
                index[part_idx == i], 
                source[part_idx == i],
                source_ts[part_idx == i])
            futs.append(fut)
        return torch.futures.collect_all(futs)


    def add_to_mailbox_async(self,index,source,source_ts):
        dist_index = DistIndex(index)
        part_idx = dist_index.part
        index = dist_index.loc
        futs: List[torch.futures.Future] = []
        if self.num_parts == 1:
            self.set_mailbox_local(index,source,source_ts)
        else:
            for i in range(self.num_parts):
                fut = self.ctx.remote_call(
                    SharedMailBox.set_mailbox_local,
                    self.rrefs[i],
                    index[part_idx == i], 
                    source[part_idx == i],
                    source_ts[part_idx == i])
                futs.append(fut)
            return torch.futures.collect_all(futs)


    def set_mailbox_all_to_all(self,index,memory,
                               memory_ts,mail,mail_ts,
                               reduce_Op = None,group = None):
        #futs: List[torch.futures.Future] = []
        if self.num_parts == 1:
            dist_index = DistIndex(index)
            part_idx = dist_index.part
            index = dist_index.loc
            self.set_mailbox_local(index,mail,mail_ts)
            self.set_memory_local(index,memory,memory_ts)
        else:
            gather_len_list = torch.empty([self.num_parts],
                                          dtype = int,
                                          device = self.device)
            indic = torch.searchsorted(index,self.partptr,right=False)
            scatter_len_list = indic[1:] - indic[0:-1]
            torch.distributed.all_to_all_single(gather_len_list,scatter_len_list,group = group)
            input_split = scatter_len_list.tolist()
            output_split = gather_len_list.tolist()
            gather_id_list = torch.empty(
                [gather_len_list.sum()],
                dtype = torch.long,
                device = self.device)
            input_split = scatter_len_list.tolist()
            output_split = gather_len_list.tolist()
            torch.distributed.all_to_all_single(
                gather_id_list,index,output_split_sizes=output_split, 
                input_split_sizes=input_split,group = group)
            index = gather_id_list
            gather_memory = torch.empty(
                [gather_len_list.sum(),memory.shape[1]],
                dtype = memory.dtype,device = self.device)
            gather_memory_ts = torch.empty(
                [gather_len_list.sum()],
                dtype = memory_ts.dtype,device = self.device)
            gather_mail = torch.empty(
                [gather_len_list.sum(),mail.shape[1]],
                dtype = mail.dtype,device = self.device)
            gather_mail_ts = torch.empty(
                [gather_len_list.sum()],
                dtype = mail_ts.dtype,device = self.device)
            torch.distributed.all_to_all_single(
                gather_memory,memory,
                output_split_sizes=output_split, 
                input_split_sizes=input_split,group = group)
            torch.distributed.all_to_all_single(
                gather_memory_ts,memory_ts,
                output_split_sizes=output_split, 
                input_split_sizes=input_split,group = group)
            torch.distributed.all_to_all_single(
                gather_mail,mail,
                output_split_sizes=output_split, 
                input_split_sizes=input_split,group = group)
            torch.distributed.all_to_all_single(
                gather_mail_ts,mail_ts,
                output_split_sizes=output_split, 
                input_split_sizes=input_split,group = group)
            self.set_mailbox_local(DistIndex(index).loc,gather_mail,gather_mail_ts,Reduce_Op = reduce_Op)
            self.set_memory_local(DistIndex(index).loc,gather_memory,gather_memory_ts, Reduce_Op = reduce_Op)

    def set_mailbox_all_to_all(self,index,memory,
                               memory_ts,mail,mail_ts,
                               reduce_Op = None,group = None):
        #futs: List[torch.futures.Future] = []
        if self.num_parts == 1:
            dist_index = DistIndex(index)
            index = dist_index.loc
            self.set_mailbox_local(index,mail,mail_ts)
            self.set_memory_local(index,memory,memory_ts)
        else:
            gather_len_list = torch.empty([self.num_parts],
                                          dtype = int,
                                          device = self.device)
            indic = torch.searchsorted(index,self.partptr,right=False)
            scatter_len_list = indic[1:] - indic[0:-1]
            torch.distributed.all_to_all_single(gather_len_list,scatter_len_list,group = group)
            input_split = scatter_len_list.tolist()
            output_split = gather_len_list.tolist()
            gather_id_list = torch.empty(
                [gather_len_list.sum()],
                dtype = torch.long,
                device = self.device)
            input_split = scatter_len_list.tolist()
            output_split = gather_len_list.tolist()
            torch.distributed.all_to_all_single(
                gather_id_list,index,output_split_sizes=output_split, 
                input_split_sizes=input_split,group = group)
            index = gather_id_list
            gather_memory = torch.empty(
                [gather_len_list.sum(),memory.shape[1]],
                dtype = memory.dtype,device = self.device)
            gather_memory_ts = torch.empty(
                [gather_len_list.sum()],
                dtype = memory_ts.dtype,device = self.device)
            gather_mail = torch.empty(
                [gather_len_list.sum(),mail.shape[1]],
                dtype = mail.dtype,device = self.device)
            gather_mail_ts = torch.empty(
                [gather_len_list.sum()],
                dtype = mail_ts.dtype,device = self.device)
            torch.distributed.all_to_all_single(
                gather_memory,memory,
                output_split_sizes=output_split, 
                input_split_sizes=input_split,group = group)
            torch.distributed.all_to_all_single(
                gather_memory_ts,memory_ts,
                output_split_sizes=output_split, 
                input_split_sizes=input_split,group = group)
            torch.distributed.all_to_all_single(
                gather_mail,mail,
                output_split_sizes=output_split, 
                input_split_sizes=input_split,group = group)
            torch.distributed.all_to_all_single(
                gather_mail_ts,mail_ts,
                output_split_sizes=output_split, 
                input_split_sizes=input_split,group = group)
            self.set_mailbox_local(DistIndex(index).loc,gather_mail,gather_mail_ts,Reduce_Op = reduce_Op)
            self.set_memory_local(DistIndex(index).loc,gather_memory,gather_memory_ts, Reduce_Op = reduce_Op)

    def get_update_mail(self,dist_indx_mapper,
                 src,dst,ts,edge_feats,
                 memory,embedding=None,use_src_emb=False,use_dst_emb=False,
                 remote_src = None, remote_dst = None,Reduce_score=None):
        if edge_feats is not None:
            edge_feats = edge_feats.to(self.device).to(self.mailbox.dtype)
        src = src.to(self.device)
        dst = dst.to(self.device)
        index = torch.cat([src, dst]).reshape(-1)
        index = dist_indx_mapper[index]
        mem_src = memory[src]
        mem_dst = memory[dst]
        if embedding is not None:
            emb_src = embedding[src]
            emb_dst = embedding[dst]
        if remote_src is None:
            src_mail = torch.cat([emb_src if use_src_emb else mem_src, emb_dst if use_dst_emb else mem_dst], dim=1)
            dst_mail = torch.cat([emb_dst if use_src_emb else mem_dst, emb_src if use_dst_emb else mem_src], dim=1)
            if edge_feats is not None:
                src_mail = torch.cat([src_mail, edge_feats], dim=1)
                dst_mail = torch.cat([dst_mail, edge_feats], dim=1)
            mail = torch.cat([src_mail, dst_mail], dim=0)
            mail_ts = torch.cat((ts,ts),-1).to(self.device).to(self.mailbox_ts.dtype)
            if Reduce_score is not None:
                Reduce_score = torch.cat((Reduce_score,Reduce_score),-1).to(self.device)
        else:
            src_mail = torch.cat([emb_src if use_src_emb else mem_src, remote_dst], dim=1)
            dst_mail = torch.cat([emb_dst if use_src_emb else mem_dst, remote_src], dim=1)
            if edge_feats is not None:
                src_mail = torch.cat([src_mail, edge_feats[:src_mail.shape[0]]], dim=1)
                dst_mail = torch.cat([dst_mail, edge_feats[src_mail.shape[0]:]], dim=1)
            mail = torch.cat([src_mail, dst_mail], dim=0)
            mail_ts = ts.to(self.device).to(self.mailbox_ts.dtype)
        #.reshape(-1, src_mail.shape[1])
        if Reduce_score is None:
            unq_index,inv = torch.unique(index,return_inverse = True)
            max_ts,idx = torch_scatter.scatter_max(mail_ts,inv,0)
            mail_ts = max_ts
            mail = mail[idx]
            index = unq_index
        else:
            unq_index,inv = torch.unique(index,return_inverse = True)
            #print(inv.shape,Reduce_score.shape)
            max_score,idx = torch_scatter.scatter_max(Reduce_score,inv,0)
            mail_ts = mail_ts[idx]
            mail = mail[idx]
            index = unq_index
        return index,mail,mail_ts
    
    def get_update_memory(self,index,memory,memory_ts):
        unq_index,inv = torch.unique(index,return_inverse = True)
        max_ts,idx = torch_scatter.scatter_max(memory_ts,inv,0)
        ts = max_ts
        memory = memory[idx]
        index = unq_index
        return index,memory,ts
    

    def get_memory(self,index,local = False):
        if self.num_parts == 1 or local is True:
            return self.node_memory.accessor.data[index],\
                    self.node_memory_ts.accessor.data[index],\
                        self.mailbox.accessor.data[index],\
                            self.mailbox_ts.accessor.data[index]
        elif self.node_memory.rrefs is None:
            return self.gather_memory(dist_index = index)
        else:
            memory = self.node_memory.index_select(index)
            memory_ts = self.node_memory_ts.index_select(index)
            mail = self.mailbox.index_select(index)
            mail_ts = self.mailbox_ts.index_select(index)
            def callback(fs):
                memory,memory_ts,mail,mail_ts = fs.value()
                memory = memory.value()
                memory_ts = memory_ts.value()
                mail = mail.value()
                mail_ts = mail_ts.value()
                #print(memory.shape[0])
                return memory,memory_ts,mail,mail_ts
            return torch.futures.collect_all([memory,memory_ts,mail,mail_ts]).then(callback)
        
    def gather_memory(
            self,
            dist_index: Union[torch.Tensor, DistIndex, None] = None,
            send_ptr: Optional[List[int]] = None,
            recv_ptr: Optional[List[int]] = None,
            recv_ind: Optional[List[int]] = None,
            group = None
        ):
        if dist_index is None:
            return self.node_memory.all_to_all_get(dist_index,send_ptr,recv_ptr,recv_ind,group),\
                self.node_memory_ts.all_to_all_get(dist_index,send_ptr,recv_ptr,recv_ind,group),\
                self.mailbox.all_to_all_get(dist_index,send_ptr,recv_ptr,recv_ind,group),\
                self.mailbox_ts.all_to_all_get(dist_index,send_ptr,recv_ptr,recv_ind,group)
        else:
            ids = self.node_memory.all_to_all_ind2ptr(dist_index)
            return self.node_memory.all_to_all_get(**ids,group = group),\
                self.node_memory_ts.all_to_all_get(**ids,group = group),\
                self.mailbox.all_to_all_get(**ids,group = group),\
                self.mailbox_ts.all_to_all_get(**ids,group = group)

    """
