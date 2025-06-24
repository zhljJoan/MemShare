import matplotlib.pyplot as plt
import re
# 数据
k_values = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3]
dict = {'LASTFM': [0.903726, 0.921197, 0.931237, 0.926789, 0.930719, 0.929332, 0.915848],
'WikiTalk' : [0.981577, 0.980716, 0.979996, 0.979597, 0.979248, 0.975468, 0.972785],
'StackOverflow' : [0.974805, 0.978219, 0.97924, 0.979436, 0.979456, 0.976746, 0.972544]}
def read(file_name):

# 定义正则表达式模式
    pattern = re.compile(r"time_count\.time_forward=(\d+\.\d+).*?time_count\.time_backward=(\d+\.\d+).*?time_count\.time_memory_updater=(\d+\.\d+).*?time_count\.time_embedding=(\d+\.\d+).*?time_count\.time_local_update=(\d+\.\d+).*?time_count\.time_memory_sync=(\d+\.\d+).*?time_count\.time_sample_and_build=(\d+\.\d+).*?time_count\.time_memory_fetch=(\d+\.\d+)")
    pattern = re.compile(r"time_count\.time_forward=(\d+\.\d+).*?time_count\.time_backward=(\d+\.\d+).*?time_count\.time_memory_updater=(\d+\.\d+).*?time_count\.time_embedding=(\d+\.\d+).*?time_count\.time_local_update=(\d+).*?time_count\.time_memory_sync=(\d+\.\d+).*?time_count\.time_sample_and_build=(\d+\.\d+).*?time_count\.time_memory_fetch=(\d+\.\d+)")

    # 读取日志文件
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
    pattern = re.compile(r"best test AP:([\d.]+) test auc([\d.]+)")

    with open(file_name, 'r') as file:
        log_content = file.read()
    ap = []
    matches = pattern.findall(log_content)
    for match in matches:
        print(match)
        ap.append(float(match[1]))
    return sum(ap)/4
# 创建新的图像和子图
fig, axs = plt.subplots(1,3, figsize=(15, 4))

# 设置全局标题
#fig.suptitle('Test AP vs topK', fontsize=24, fontweight='bold')

# 绘制LASTFM折线图
for i,data in enumerate(['LASTFM','StackOverflow']):


    model = 'TGN' if data == 'LASTFM' else 'TGN_large'
    neighbor = 10 if model == 'TGN' else 20 
    for p in [0.1,0.6,1]:
        #if p > 0.1:
            #if p != 1:
        ap = [ 27.19218282 if data == 'StackOverflow' and p ==0.1 and topk==0.1 else read(f"all_12357/{data}/{model}/8-ours-{topk}-historical-{0.3}-boundery_recent_decay-{p}-{neighbor}.out") for topk in k_values]
            #else:
            #    ap = [readap(f"all_12357/{data}/{model}/8-ours-{topk}-historical-{0.3}-recent-{neighbor}.out") for topk in k_values]
        #else:
        #    ap = dict[data]
        print(p,ap)
        type='recent' if p == 1 else "decay"
        axs[i].plot(k_values, ap, marker='o', linestyle='-',   linewidth=3,label=f'p={p}({type})')
    axs[i].legend()
    axs[i].set_title(f'{data}', fontsize=20, fontweight='bold')
    axs[i].set_xlabel('Fetch Time(s)', fontsize=16)
    axs[i].set_ylabel('Test AP', fontsize=16)
    axs[i].grid(True)





plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('topk_time.png')
