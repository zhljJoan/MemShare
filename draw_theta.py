import matplotlib.pyplot as plt

# 数据
theta_values = [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
lastfm = [0.813412, 0.920839, 0.92926, 0.930115, 0.930719, 0.923493, 0.927908, 0.925335, 0.896277]
wikitalk = [0.967408, 0.979705, 0.979805, 0.97515, 0.979248, 0.979162, 0.978207, 0.977578, 0.975614]
stackoverflow = [0.919117667, 0.9778155, 0.978890667, 0.979456, 0.97914, 0.980143, 0.980606, 0.980387, 0.977536]

# 创建新的图像和子图
fig, axs = plt.subplots(1,3, figsize=(15, 4))

# 设置全局标题
#fig.suptitle('精度折线图', fontsize=24, fontweight='bold')

# 绘制LASTFM折线图
axs[0].plot(theta_values, lastfm, marker='o', linestyle='-', color='blue', linewidth=3)
axs[0].set_title('LASTFM', fontsize=20, fontweight='bold')
axs[0].set_xlabel('θ', fontsize=16)
axs[0].set_ylabel('Test AP', fontsize=16)
axs[0].grid(True)

# 绘制WikiTalk折线图
axs[1].plot(theta_values, wikitalk, marker='o', linestyle='-', color='green', linewidth=3)
axs[1].set_title('WikiTalk', fontsize=20, fontweight='bold')
axs[1].set_xlabel('θ', fontsize=16)
axs[1].set_ylabel('Test AP', fontsize=16)
axs[1].grid(True)

# 绘制StackOverflow折线图
axs[2].plot(theta_values, stackoverflow, marker='o', linestyle='-', color='red', linewidth=3)
axs[2].set_title('StackOverflow', fontsize=20, fontweight='bold')
axs[2].set_xlabel('θ', fontsize=16)
axs[2].set_ylabel('Test AP', fontsize=16)
axs[2].grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('theta.png')