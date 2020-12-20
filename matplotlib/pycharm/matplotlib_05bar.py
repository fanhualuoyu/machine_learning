# 绘制条形图
from matplotlib import pyplot as plt

font = {'family': 'Microsoft YaHei',
        'weight': 'bold',
        'size': '10'}
plt.rc('font', **font)

x = ["战狼1", "战狼2", "战狼3", "战狼4", "战狼5", "战狼6", "战狼7", "战狼8"]
y = [1, 2, 3, 4, 5, 6, 7, 8]

# 设置图形大小
plt.figure(figsize=(10, 8), dpi=80)

# 绘制条形图
plt.bar(range(len(x)), y)

# 横着画条形图
# plt.barh()

# 设置x轴
_x = [i for i in range(len(x))]
_xticks_labels = x
plt.xticks(_x, _xticks_labels)

# 绘制网格
plt.grid(alpha=0.4)

# 设置提示
plt.xlabel("电影名")
plt.ylabel("票房")
plt.title("电影票房")

plt.show()
