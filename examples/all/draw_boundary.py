import matplotlib.pyplot as plt
import numpy as np
import torch
# 读取文件内容
ssim_values = [0, 0.1, 0.2, 0.3, 0.4, 2]  # 假设这是你的 ssim 参数值
probability_values = [1,0.5,0.1,0.05,0.01,0]
data_values = ['WIKI_3','LASTFM_3','WikiTalk','StackOverflow']  # 存储从文件中读取的数据
partition = 'ours'
# 从文件中读取数据，假设数据存储在文件 data.txt 中
#all/"$data"/"$partitions"-ours_shared-0.01-"$mem"-"$ssim"-"$sample".out
partitions=4
topk=0
mem='all_update'#'historical'
model='TGN'
for data in data_values:
    ap_list = []
    comm_list = []
    for p in probability_values:

        file = '{}/{}/{}-{}-{}-{}-boundery_recent_uniform-{}.out'.format(data,model,partitions,partition,topk,mem,p)
        prefix = 'best test AP:'
        cnt = 0
        sum = 0
        with open(file, 'r') as file:
            for line in file:
                if line.startswith(prefix):
                    ap = float(line.lstrip(prefix).split(' ')[0])
                pos = line.find('remote node number tensor')
                if(pos!=-1):
                    posr = line.find(']',pos+2+len('remote node number tensor'),)
                    #print(line,line[pos+2+len('remote node number tensor'):posr])
                    comm = int(line[pos+2+len('remote node number tensor'):posr])
                    #print()
                    sum = sum+comm
                    cnt = cnt+1
                    #print(comm)
        ap_list.append(ap)
        comm_list.append(sum/cnt*4)


    # 绘制柱状图
    print('{} TestAP={}\n'.format(data,ap_list))
    bar_width = 0.4
    #shared comm tensor

# 设置柱状图的位置
    bars = range(len(probability_values))

# 绘制柱状图
    plt.bar([b for b in bars], ap_list, width=bar_width)
    # 绘制柱状图
    plt.ylim([0.9,1])
    plt.xticks([b for b in bars], probability_values)
    plt.xlabel('probability')
    plt.ylabel('Test AP')
    plt.title('{}({} partitions)'.format(data,partitions))
    plt.savefig('boundary_AP_{}_{}_{}.png'.format(data,partitions,model))
    plt.clf()
    print(comm_list)
    plt.bar([b for b in bars], comm_list, width=bar_width)
    # 绘制柱状图
    plt.xticks([b for b in bars], probability_values)
    plt.xlabel('probability')
    plt.ylabel('Communication volume')
    plt.title('{}({} partitions)'.format(data,partitions))
    plt.savefig('boundary_comm_{}_{}_{}.png'.format(data,partitions,model))
    plt.clf()

    if partition == 'ours_shared':
        partition0 = 'ours'
    else:
        partition0=partition
    for p in probability_values:
        file = '{}/{}/test_{}_{}_{}_0_boundery_recent_uniform_{}_all_update_2.pt'.format(data,model,partition0,topk,partitions,float(p))
        val_ap = torch.tensor(torch.load(file))[:,0]
        epoch = torch.arange(val_ap.shape[0])
    #绘制曲线图
        plt.plot(epoch,val_ap, label='probability={}'.format(p))
    plt.xlabel('Epoch')
    plt.ylabel('Val AP')
    plt.title('{}({} partitions)'.format(data,partitions))
   # plt.grid(True)
    plt.legend()
    plt.savefig('{}_{}_{}_boundary_Convergence_rate.png'.format(data,partitions,model))
    plt.clf()
