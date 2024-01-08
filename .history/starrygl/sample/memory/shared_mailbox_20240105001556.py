from typing import Union
from typing import List
from typing import Optional
import torch
from torch.distributed import rpc
import torch_scatter
from starrygl.distributed.context import DistributedContext
from starrygl.distributed.utils import DistIndex, DistributedTensor
import torch.distributed as dist

from starrygl.utils.uvm import cudaMemoryAdvise, uvm_advise, uvm_empty, uvm_prefetch, uvm_share

class SharedMailBox():
    '''
    We will first define our mailbox, including our definitions of mialbox and memory:
    .. code-block:: python
        from starrygl.sample.memory.shared_mailbox import SharedMailBox
        mailbox = SharedMailBox(num_nodes=num_nodes, memory_param=memory_param, dim_edge_feat=dim_edge_feat)

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
    the relevant storage,which is the idea related to TGN and APAN.
    '''
    def __init__(self,
                 num_nodes,
                 memory_param,
                 dim_edge_feat,
                 device = torch.device('cuda'),
                 uvm = False):
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
                                      dtype=torch.float32,
                                      device = self.device)
        mailbox = torch.zeros(self.num_nodes, 
                              memory_param['mailbox_size'], 
                              2 * memory_param['dim_out'] + dim_edge_feat,
                              device = memory_device, dtype=torch.float32)
        mailbox_ts = torch.zeros((self.num_nodes, 
                                  memory_param['mailbox_size']), 
                             dtype=torch.float32,device  = self.device)
        self.uvm = uvm
        if uvm is True:
            
            ctx = DistributedContext.get_default_context()
            node_memory = uvm_empty(*node_memory.shape,
                                    dtype=node_memory.dtype,
                                    device=ctx.device)
            uvm_share(node_memory,device = ctx.device)
            uvm_advise(node_memory,cudaMemoryAdvise.cudaMemAdviseSetAccessedBy)
            uvm_prefetch(node_memory)
            mailbox = uvm_empty(*mailbox.shape,
                                    dtype=mailbox.dtype,
                                    device=ctx.device)
            uvm_share(mailbox,device = ctx.device)
            uvm_advise(mailbox,cudaMemoryAdvise.cudaMemAdviseSetAccessedBy)
            uvm_prefetch(mailbox)
        self.node_memory = DistributedTensor(node_memory)
        self.node_memory_ts = DistributedTensor(node_memory_ts)
        self.mailbox = DistributedTensor(mailbox)
        self.mailbox_ts = DistributedTensor(mailbox_ts)
        self.next_mail_pos = torch.zeros((self.num_nodes), 
                                         dtype=torch.long,
                                         device = self.device) 
        self._ctx = DistributedContext.get_default_context()
        if self._ctx._use_rpc == True:
            self.rref = rpc.RRef(self)  
            self.rrefs = self._ctx.all_gather_remote_objects(self.rref)
        self.partptr = torch.tensor([ ((i & 0xFFFF)<<48) for i in range(self.num_parts+1) ],device = device)


    def reset(self):
        self.node_memory.accessor.data.zero_()
        self.node_memory_ts.accessor.data.zero_()
        self.mailbox.accessor.data.zero_()
        self.mailbox_ts.accessor.data.zero_()
        self.next_mail_pos.zero_()
    

    def set_memory_local(self,index,source,source_ts,Reduce_Op = None):
        if Reduce_Op == 'max':
            unq_id,inv = index.unique(return_inverse = True)
            max_ts,id =  torch_scatter.scatter_max(source_ts,inv,dim=0)
            source_ts = max_ts
            source = source[id]
            index = unq_id
        self.node_memory.accessor.data[index] = source
        self.node_memory_ts.accessor.data[index] = source_ts


    def set_mailbox_local(self,index,source,source_ts,Reduce_Op = None):
        if Reduce_Op == 'max' and self.num_parts > 1:
            unq_id,inv = index.unique(return_inverse = True)
            max_ts,id =  torch_scatter.scatter_max(source_ts,inv,dim=0)
            source_ts = max_ts
            source = source[id]
            index = unq_id
        self.mailbox_ts.accessor.data[index, self.next_mail_pos[index]] = source_ts
        self.mailbox.accessor.data[index, self.next_mail_pos[index]] = source
        if self.memory_param['mailbox_size'] > 1:
            self.next_mail_pos[index] = torch.remainder(
                self.next_mail_pos[index] + 1, 
                self.memory_param['mailbox_size'])
            

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

    def get_update_mail(self,dist_indx_mapper,
                 src,dst,ts,edge_feats,
                 memory):
        if edge_feats is not None:
            edge_feats = edge_feats.to(self.device).to(self.mailbox.dtype)
        src = src.to(self.device)
        dst = dst.to(self.device)
        index = torch.cat([src, dst]).reshape(-1)
        index = dist_indx_mapper[index]

        mem_src = memory[src]
        mem_dst = memory[dst]
        if edge_feats is not None:
            src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
            dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
        else:
            src_mail = torch.cat([mem_src, mem_dst], dim=1)
            dst_mail = torch.cat([mem_dst, mem_src], dim=1)
        mail = torch.cat([src_mail, dst_mail], dim=1).reshape(-1, src_mail.shape[1])
        mail_ts = torch.cat((ts,ts),-1).to(self.device).to(self.mailbox_ts.dtype)
        unq_index,inv = torch.unique(index,return_inverse = True)
        max_ts,idx = torch_scatter.scatter_max(mail_ts,inv,0)
        mail_ts = max_ts
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
    

    def get_memory(self,index):
        if self.num_parts == 1:
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
            

