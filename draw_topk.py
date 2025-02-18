import matplotlib.pyplot as plt

# 数据
k_values = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3]
lastfm = [0.903726, 0.921197, 0.931237, 0.926789, 0.930719, 0.929332, 0.915848]
wikitalk = [0.981577, 0.980716, 0.979996, 0.979597, 0.979248, 0.975468, 0.972785]
stackoverflow = [0.974805, 0.978219, 0.97924, 0.979436, 0.979456, 0.976746, 0.972544]

# 创建新的图像和子图
fig, axs = plt.subplots(1,3, figsize=(15, 4))

# 设置全局标题
#fig.suptitle('Test AP vs topK', fontsize=24, fontweight='bold')

# 绘制LASTFM折线图
axs[0].plot(k_values, lastfm, marker='o', linestyle='-', color='blue', linewidth=3)
axs[0].set_title('LASTFM', fontsize=20, fontweight='bold')
axs[0].set_xlabel('k', fontsize=16)
axs[0].set_ylabel('Test AP', fontsize=16)
axs[0].grid(True)

# 绘制WikiTalk折线图
axs[1].plot(k_values, wikitalk, marker='o', linestyle='-', color='green', linewidth=3)
axs[1].set_title('WikiTalk', fontsize=20, fontweight='bold')
axs[1].set_xlabel('k', fontsize=16)
axs[1].set_ylabel('Test AP', fontsize=16)
axs[1].grid(True)

# 绘制StackOverflow折线图
axs[2].plot(k_values, stackoverflow, marker='o', linestyle='-', color='red', linewidth=3)
axs[2].set_title('StackOverflow', fontsize=20, fontweight='bold')
axs[2].set_xlabel('k', fontsize=16)
axs[2].set_ylabel('Test AP', fontsize=16)
axs[2].grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('topk.png')
