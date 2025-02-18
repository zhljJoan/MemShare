import matplotlib.pyplot as plt

# 数据
alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.3, 1.5, 1.7, 2]
lastfm = [0.933538, 0.931761, 0.932499, 0.933383, 0.931225, 0.929983, 0.933971, 0.928771, 0.932748]
wikitalk = [0.979627, 0.97997, 0.979484, 0.980269, 0.980758, 0.97979, 0.980233, 0.980004, 0.980353]
stackoverflow = [0.979641, 0.979372, 0.97967, 0.978169, 0.979624, 0.978846, 0.978428, 0.978397, 0.978925]

# 创建新的图像和子图
fig, axs = plt.subplots(1,3, figsize=(15,4))

# 设置全局标题
#fig.suptitle('精度折线图', fontsize=24, fontweight='bold')

# 绘制LASTFM折线图
axs[0].plot(alpha_values, lastfm, marker='o', linestyle='-', color='blue', linewidth=3)
axs[0].set_title('LASTFM', fontsize=20, fontweight='bold')
axs[0].set_xlabel('α', fontsize=16)
axs[0].set_ylabel('Test AP', fontsize=16)
axs[0].grid(True)

# 绘制WikiTalk折线图
axs[1].plot(alpha_values, wikitalk, marker='o', linestyle='-', color='green', linewidth=3)
axs[1].set_title('WikiTalk', fontsize=20, fontweight='bold')
axs[1].set_xlabel('α', fontsize=16)
axs[1].set_ylabel('Test AP', fontsize=16)
axs[1].grid(True)

# 绘制StackOverflow折线图
axs[2].plot(alpha_values, stackoverflow, marker='o', linestyle='-', color='red', linewidth=3)
axs[2].set_title('StackOverflow', fontsize=20, fontweight='bold')
axs[2].set_xlabel('α', fontsize=16)
axs[2].set_ylabel('Test AP', fontsize=16)
axs[2].grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('alpha.png')
