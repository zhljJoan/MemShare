import matplotlib.pyplot as plt
import numpy as np
import torch
# 读取文件内容
ssim_values = [-1,0,0.3,0.7,2]  # 假设这是你的 ssim 参数值
data_values = ['WIKI','LASTFM','WikiTalk','REDDIT','LASTFM','DGraphFin']  # 存储从文件中读取的数据
partition = 'ours_shared'
# 从文件中读取数据，假设数据存储在文件 data.txt 中
#all/"$data"/"$partitions"-ours_shared-0.01-"$mem"-"$ssim"-"$sample".out
partitions=4
model = 'JODIE'
topk=0.01
mem='historical'
for data in data_values:
    ap_list = []
    comm_list = []
    for ssim in ssim_values:
        if ssim == 2:
            file = '{}/{}/{}-{}-{}-local-recent.out'.format(data,model,partitions,partition,topk)
        elif ssim == -1:
            file = '{}/{}/{}-{}-{}-all_update-recent.out'.format(data,model,partitions,partition,topk)
        else:
            file = '{}/{}/{}-{}-{}-{}-{}-recent.out'.format(data,model,partitions,partition,topk,mem,ssim)
        prefix = 'best test AP:'
        with open(file, 'r') as file:
            for line in file:
                if line.startswith(prefix):
                    ap = float(line.lstrip(prefix).split(' ')[0])
                pos = line.find('shared comm tensor')
                if(pos!=-1):
                    comm = int(line[pos+2+len('shared comm tensor'):len(line)-3])
        print(ap)
        ap_list.append(ap)
        comm_list.append(comm)
    print('{} TestAP={}\n'.format(data,ap_list))

    # 绘制柱状图
    bar_width = 0.4
    #shared comm tensor
    print('{} TestAP={}\n'.format(data,ap_list))
# 设置柱状图的位置
    bars = range(len(ssim_values))

# 绘制柱状图
    plt.bar([b for b in bars], ap_list, width=bar_width)
    # 绘制柱状图
    plt.xticks([b for b in bars], ssim_values)
    plt.xlabel('SSIM threshold Values')
    plt.ylabel('Test AP')
    #if(data=='WIKI'):
    #    plt.ylim([0.97,1])
    plt.title('{}({} partitions)'.format(data,partitions))
    plt.savefig('ssim_{}_{}_{}.png'.format(data,partitions,model))
    plt.clf()

    plt.bar([b for b in bars], comm_list, width=bar_width)
    # 绘制柱状图
    plt.xticks([b for b in bars], ssim_values)
    plt.xlabel('SSIM threshold Values')
    plt.ylabel('Communication volume')
    plt.title('{}({} partitions)'.format(data,partitions))
    plt.savefig('ssim_comm_{}_{}_{}.png'.format(data,partitions,model))
    plt.clf()

    if partition == 'ours_shared':
        partition0 = 'ours'
    else:
        partition0=partition
    for ssim in ssim_values:
        if ssim == 2:
            file = '{}/{}/test_{}_{}_{}_0_recent_0.1_local_2.pt'.format(data,model,partition0,topk,partitions,)
        elif ssim == -1:
            file = '{}/{}/test_{}_{}_{}_0_recent_0.1_all_update_2.pt'.format(data,model,partition0,topk,partitions,)
        else:
            file = '{}/{}/test_{}_{}_{}_0_recent_0.1_{}_{}.pt'.format(data,model,partition0,topk,partitions,mem,float(ssim))
        val_ap = torch.tensor(torch.load(file))[:,0]
        print(val_ap)
        epoch = torch.arange(val_ap.shape[0])
    #绘制曲线图
        #print(val_ap)
        if ssim == -1:
            plt.plot(epoch,val_ap, label='all-update')
        elif ssim == 2:
            plt.plot(epoch,val_ap, label='local')
        else:
            plt.plot(epoch,val_ap, label='ssim = {}'.format(ssim))
    if(data=='WIKI'):
        plt.ylim([0.85,0.90])
    plt.xlabel('Epoch')
    plt.ylabel('Val AP')
    plt.title('{}({} partitions)'.format(data,partitions))
   # plt.grid(True)
    plt.legend()
    plt.savefig('{}_{}_{}_ssim_Convergence_rate.png'.format(data,partitions,model))
    plt.clf()
