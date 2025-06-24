import matplotlib.pyplot as plt
import numpy as np
import torch
# 读取文件内容
import os
probability_values = [1,0.1,0.05,0.01,0]#[0.1,0.05,0.01,0]
data_values = ['WIKI','LASTFM','WikiTalk','StackOverflow','GDELT']  # 存储从文件中读取的数据
seed = ['13357','12347','53473','54763','12347','63377','53473','54763']
partition = 'ours_shared'

# 从文件中读取数据，假设数据存储在文件 data.txt 中
#all/"$data"/"$partitions"-ours_shared-0.01-"$mem"-"$ssim"-"$sample".out
partitions=4
topk=0.01
mem='all_update'#'historical'
model0='TGN'
def average(l):
    return sum(l)/len(l)
for p in probability_values:
    for data in data_values:
        ap_list = []
        comm_list = []
        test_time = []
        train_time = []
        total_communication = []
        shared_synchronize = []
        for sd in seed :
        
            if data == 'WIKI' or data =='LASTFM':
                model = model0
            else:
                model = model0+'_large'
            if p == 1:
                file = '../examples-probability-sample/all_{}/{}/{}/{}-{}-{}-{}-recent.out'.format(sd,data,model,partitions,partition,topk,mem)
            else:
                file = '../examples-probability-sample/all_{}/{}/{}/{}-{}-{}-{}-boundery_recent_decay-{}.out'.format(sd,data,model,partitions,partition,topk,mem,p)
            #print(file)
            prefix = "val ap:"
            max_val_ap = 0
            test_ap = 0
            #if 
            #print(file)
            if os.path.exists(file):
                with open(file, 'r') as file:
                    _total_communication = []
                    _shared_synchronize = []
                    for line in file:
                        #if line.find('Epoch 50:')!=-1:
                        #    break
                        if line.find(prefix)!=-1:
                            pos = line.find(prefix)+len(prefix)
                            posr = line.find(' ',pos)
                            #print(line[pos:posr])
                            val_ap = float(line[pos:posr])

                            pos = line.find("test ap ")+len("test ap ")
                            posr = line.find(' ',pos)
                            #print(line[pos:posr])
                            _test_ap = float(line[pos:posr])
                            if(val_ap>max_val_ap):
                                max_val_ap = val_ap
                                test_ap = _test_ap
                        elif line.find('avg_time ')!=-1:
                            pl = line.find('avg_time ') + len('avg_time ')
                            pr = line.find(' test time')
                            train_time.append(float(line[pl:pr]))
                            test_time.append(float(line[pr+len(' test time '):]))
                            #print(line)
                            ap_list.append(test_ap)
                            total_communication.append(average(_total_communication))
                            shared_synchronize.append(average(_shared_synchronize))
                            """
                            local node number tensor([114453298]) remote node number tensor([20457479]) local edge tensor([1592867807]) remote edgetensor([326448632])
local node number tensor([114453298]) remote node number tensor([20457479]) local edge tensor([1592867807]) remote edgetensor([326448632])


 comm local node number 0 remote node number 0 local edge 0 remote edge0

 comm local node number 0 remote node number 0 local edge 0 remote edge0

memory comm tensor([0]) shared comm tensor([24860])
memory comm tensor([0]) shared comm tensor([24860])
                            """
                        elif line.find('remote node number tensor([')!=-1:
                            pl = line.find('remote node number tensor([')+len('remote node number tensor([')
                            pr = line.find('])',pl)
                            _total_communication.append(int(line[pl:pr]))
                            #if(p==0):
                                #print(file)
                                #print(line)
                        elif line.find('shared comm tensor([')!=-1:
                            pl = line.find('shared comm tensor([')+len('shared comm tensor([')
                            pr = line.find('])',pl)
                            _shared_synchronize.append(int(line[pl:pr]))
            #else:
            #    print(file)
        if len(ap_list) > 0:
            #print('prob {} data {} model {} remote volume : {} synchronize volume : {}'.format(p,data,model,average(total_communication),average(shared_synchronize)))
            print('prob {} data {} model {} ap: {} train_time: {} eval time: {} remote volume : {} synchronize volume : {}'.format(p,data,model,average(ap_list),average(train_time),average(test_time),average(total_communication),average(shared_synchronize)))
