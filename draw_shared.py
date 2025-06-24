import matplotlib.pyplot as plt

# 数据
x_values = [0]
y_values = [0.310754, 0.19102, 0.117148, 0.075549]

# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(y_values, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)

# 添加标题和标签
plt.title('Number of Cross Edge vs shared node ratio', fontsize=16)
plt.xlabel('The ratio of Cross Edge', fontsize=14)
plt.ylabel('shared nodes ratio', fontsize=14)

# 显示网格
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图表
plt.show()
plt.savefig('shared_ratio.png')
