import matplotlib.pyplot as plt
import re
import os
# 数据
dict = {'LASTFM': {
    1:0.8963 ,
    0.1:0.9301,
    0.05:0.9293 ,
    0.01:0.9208,
    0:0.8134 
},
'WikiTalk' : {
    1:0.9756,0.1: 	0.9752 	,0.05:0.9798 	,0.01:0.9797 	,0:0.9674 

}
}
def read(file_name):

# 定义正则表达式模式
    pattern = re.compile(r"time_count\.time_forward=(\d+\.\d+).*?time_count\.time_backward=(\d+\.\d+).*?time_count\.time_memory_updater=(\d+\.\d+).*?time_count\.time_embedding=(\d+\.\d+).*?time_count\.time_local_update=(\d+\.\d+).*?time_count\.time_memory_sync=(\d+\.\d+).*?time_count\.time_sample_and_build=(\d+\.\d+).*?time_count\.time_memory_fetch=(\d+\.\d+)")
    pattern = re.compile(r"time_count\.time_forward=(\d+\.\d+).*?time_count\.time_backward=(\d+\.\d+).*?time_count\.time_memory_updater=(\d+\.\d+).*?time_count\.time_embedding=(\d+\.\d+).*?time_count\.time_local_update=(\d+).*?time_count\.time_memory_sync=(\d+\.\d+).*?time_count\.time_sample_and_build=(\d+\.\d+).*?time_count\.time_memory_fetch=(\d+\.\d+)")

    # 读取日志文件
    if not os.path.exists(file_name):
        return 0
    with open(file_name, 'r') as file:
        log_content = file.read()
    #print(log_content)

    # 查找所有匹配的行
    matches = pattern.findall(log_content)
    # 初始化累加器
    forward_times = []
    backward_times = []
    memory_updater_times = []
    embedding_times = []
    local_update_times = []
    memory_sync_times = []
    sample_and_build_times = []
    memory_fetch_times = []

    # 提取数据并累加
    for match in matches:

        forward_times.append(float(match[0]))
        backward_times.append(float(match[1]))
        memory_updater_times.append(float(match[2]))
        embedding_times.append(float(match[3]))
        local_update_times.append(float(match[4]))
        memory_sync_times.append(float(match[5]))
        sample_and_build_times.append(float(match[6]))
        memory_fetch_times.append(float(match[7]))

    # 计算平均值
    def calculate_average(times):
        return sum(times) / len(times) if times else 0

    average_forward = calculate_average(forward_times)
    average_backward = calculate_average(backward_times)
    average_memory_updater = calculate_average(memory_updater_times)
    average_embedding = calculate_average(embedding_times)
    average_local_update = calculate_average(local_update_times)
    average_memory_sync = calculate_average(memory_sync_times)
    average_sample_and_build = calculate_average    (sample_and_build_times)
    average_memory_fetch = calculate_average(memory_fetch_times)
    averages = {
        'average_forward': average_forward,
        'average_backward': average_backward,
        'average_memory_updater': average_memory_updater,
        'average_embedding': average_embedding,
        'average_local_update': average_local_update,
        'average_memory_sync': average_memory_sync,
        'average_sample_and_build': average_sample_and_build,
        'average_memory_fetch':average_memory_fetch,
    }
    return averages['average_memory_fetch']/1000

def readap(file_name):
    if not os.path.exists(file_name):
        return 0
    pattern = re.compile(r"best test AP:([\d.]+) test auc([\d.]+)")

    with open(file_name, 'r') as file:
        log_content = file.read()
    ap = []
    matches = pattern.findall(log_content)
    for match in matches:
        #print(match)
        ap.append(float(match[0]))
    return sum(ap)/4
# 创建新的图像和子图
fig, axs = plt.subplots(1,4, figsize=(18, 4))

# 设置全局标题
#fig.suptitle('Test AP vs topK', fontsize=24, fontweight='bold')

# 绘制LASTFM折线图
color = ['b','g','r','c']
for i,data in enumerate(['LASTFM','WikiTalk','StackOverflow','GDELT']):


    model = 'TGN' if data == 'LASTFM' else 'TGN_large'
    neighbor = 10 if model == 'TGN' else 20 
    his = [0.1,0.3,0.5,0.7,0.9,1.3,1.5]#1.7,2]
    ap = [ readap(f"all_12345_new_gamma_set_para/{data}/{model}/8-ours-0.1-historical-{p}-boundery_recent_decay-{0.1}-{neighbor}.out") for p in his]
    #ap = [dict[data][p] if ap_ == 0 and data in ['LASTFM','WikiTalk'] else ap_ for ap_,p in zip(ap_new,p_list)]
    print(ap)
    #type='recent' if p == 1 else "decay"
    axs[i].plot(his, ap, marker='o', linestyle='-',   linewidth=3,color = color[i],label=f'{data}')
    #axs[i].legend()
    axs[i].set_title(f'{data}', fontsize=20, fontweight='bold')
    axs[i].set_xlabel(r'$\theta$', fontsize=16)
    axs[i].set_ylabel('AP', fontsize=16)
    axs[i].grid(True)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
print('save')
plt.savefig('his.png')
