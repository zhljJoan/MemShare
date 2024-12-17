import torch

import matplotlib.pyplot as plt
import numpy as np
import os
class MemoryMoniter:
    memorychange = []
    nid_list = []
    memory_ssim = []
    def ssim(self,x,y,method = 'cos'):
        if method  == 'cos':
            return torch.cosine_similarity(x,y)
        else:
            return torch.sum((x-y)**2,dim=1) 
    def add(self,nid,pre_memory,now_memory):
        pass
        #self.memorychange.append(self.ssim(pre_memory,now_memory,method = 'F'))
        #self.memory_ssim.append(self.ssim(pre_memory,now_memory,method = 'cos'))
        #self.nid_list.append(nid)
    def draw(self,degree,data,model,e):
        pass
        #torch.save(self.nid_list,'all_args.seed/{}/{}/memorynid_{}.pt'.format(data,model,e))
        #torch.save(self.memorychange,'all_args.seed/{}/{}/memoryF_{}.pt'.format(data,model,e))
        #torch.save(self.memory_ssim,'all_args.seed/{}/{}/memcos_{}.pt'.format(data,model,e))
        # path = './memory/{}/'.format(data)

        # if not os.path.exists(path):

        #     os.makedirs(path)
        #     print(f"Folder '{path}' created.")
        # else:
 
        #     print(f"Folder '{path}' already exists.")
        # self.nid = torch.cat(self.nid_list).to('cpu')
        # self.mem = torch.cat(self.memorychange).to('cpu')
        # self.mem_cos = torch.cat(self.memory_ssim).to('cpu')
        # d = torch.log(degree[self.nid])
        # sd,_ = degree.sort()
        # value,idx =torch.topk(sd,int(degree.shape[0]*0.01))
        # print(value[-1],torch.log(value[-1]))
        # x=d
        # y = torch.min(self.mem,torch.tensor([20]))
        # plt.hist2d(x, y, bins=10,  range=[[0,x.max().item()],[0,20]])
        # plt.title('Memory increment vs degree')
        # plt.xlabel('log(degree)')
        # plt.ylabel('Memory increment')
        # plt.savefig('./memory/{}/test1-{}-global.png'.format(data,e))
        # plt.clf()

        # counts, xedges, yedges = np.histogram2d(x, y, bins=10, range=[[0,x.max().item()], [0, 20]])


        # row_sums = counts.sum(axis=1, keepdims=True)
        # row_sums[row_sums == 0] = 1
        # counts_normalized = counts / row_sums


        # plt.pcolormesh(xedges, yedges, counts_normalized.T)
        # plt.colorbar(label='Counts')
        # plt.title('Memory increment vs degree')
        # plt.xlabel('log(degree)')
        # plt.ylabel('Memory increment')
        # plt.savefig('./memory/{}/test1-{}.png'.format(data,e))
        # plt.clf()
        # y = self.mem_cos
        # plt.hist2d(x, y, bins=10,range=[[0,x.max().item()],[-1,1]])
        # plt.title('Memory increment vs degree')
        # plt.xlabel('log(degree)')
        # plt.ylabel('Memory increment')
        # plt.savefig('./memory/{}/test1-{}-global-cos.png'.format(data,e))
        # plt.clf()

        # counts, xedges, yedges = np.histogram2d(x, y, bins=10, range=[[0,x.max().item()], [-1,1]])


        # row_sums = counts.sum(axis=1, keepdims=True)
        # row_sums[row_sums == 0] = 1
        # counts_normalized = counts / row_sums


        # plt.pcolormesh(xedges, yedges, counts_normalized.T)
        # plt.colorbar(label='Counts')
        # plt.title('Memory increment vs degree')
        # plt.xlabel('log(degree)')
        # plt.ylabel('Memory increment')
        # plt.savefig('./memory/{}/test1-{}-cos.png'.format(data,e))
        # plt.clf()


    def set_zero(self):
        pass
        #self.memorychange = []
        #self.nid_list =[]
        #self.memory_ssim = []
