import matplotlib.pyplot as plt
import numpy as np
import torch
# 读取文件内容

probability_values = [0.1]#[0.1,0.05,0.01,0]
data_values = ['WikiTalk']  # 存储从文件中读取的数据
seed = ['12357']#,'12347','63377','53473','54763']
partition = 'ours_shared'

# 从文件中读取数据，假设数据存储在文件 data.txt 中
#all/"$data"/"$partitions"-ours_shared-0.01-"$mem"-"$ssim"-"$sample".out
partitions=4
topk=0.01
mem='historical-0.3'#'historical'
model0='APAN'
def average(l):
    return sum(l)/len(l)
for data in data_values:
    ap_list = []
    comm_list = []
    for sd in seed :
        for p in probability_values:
            if data == 'WIKI' or data =='LASTFM':
                model = model0
            else:
                model = model0+'_large'
            if p == 1:
                file = 'all_{}/{}/{}/{}-{}-{}-{}-recent.out'.format(sd,data,model,partitions,partition,topk,mem)
            else:
                file = 'all_{}/{}/{}/{}-{}-{}-{}-boundery_recent_decay-{}.out'.format(sd,data,model,partitions,partition,topk,mem,p)
            prefix = "val ap:"
            max_val_ap = 0
            test_ap = 0
            
            with open(file, 'r') as file:
                for line in file:
                    if line.find('Epoch 50:')!=-1:
                        break
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
                ap_list.append(test_ap)
        print('data {} model {} ap: {}'.format(data,model,ap_list))
