from matplotlib import pyplot as plt
import random as r

font = {'family': 'Microsoft YaHei',
        'weight': 'bold',
        'size': '10'}
plt.rc('font', **font)

a = [i for i in range(1, 101)]

# 计算组数
d = 5  # 组距
num_bins = (max(a) - min(a)) // d  # 组数

plt.hist(a, num_bins)

# 设置x轴的刻度
plt.xticks(range(min(a), max(a) + d, d))

plt.show()
