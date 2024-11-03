import matplotlib.pyplot as plt
import numpy as np

# 数据
p_values = ['recent', 'p=0.1', 'p=0.05', 'p=0.01', 'p=0']
wiki_values = [0.979832, 0.980298, 0.975079, 0.97349, 0.96381]
lastfm_values = [0.820161, 0.852725, 0.848085, 0.817381, 0.796689]
wikitalk_values = [0.969647, 0.974473, 0.973996, 0.968961, 0.964867]
gdelt_values = [0.987338, 0.987454, 0.987038, 0.98812, 0.98726]

# 柱状图的宽度
barWidth = 0.15

# 柱状图的位置
r1 = np.arange(len(wiki_values))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# 创建图形
plt.figure(figsize=(12,8))
plt.bar(r1, wiki_values, color='b', width=barWidth, edgecolor='grey', label='WIKI')
plt.bar(r2, lastfm_values, color='r', width=barWidth, edgecolor='grey', label='LASTFM')
plt.bar(r3, wikitalk_values, color='g', width=barWidth, edgecolor='grey', label='WikiTalk')
plt.bar(r4, gdelt_values, color='y', width=barWidth, edgecolor='grey', label='GDELT')

# 添加标签
plt.xlabel('p values', fontweight='bold', fontsize=15)
plt.ylabel('SSIM', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(len(wiki_values))], p_values)
plt.savefig('bound.png')
plt.legend()
plt.show()
