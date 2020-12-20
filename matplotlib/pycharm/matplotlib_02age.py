# 练习
from matplotlib import pyplot as plt

font = {'family': 'Microsoft YaHei',
        'weight': 'bold',
        'size': '10'}
plt.rc('font', **font)

y_1 = [1, 0, 1, 1, 2, 4, 3, 2, 3, 4, 4, 5, 6, 5, 4, 3, 3, 1, 1, 1]
y_2 = [1, 0, 3, 1, 2, 2, 3, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
x = range(11, 31)

# 设置图形大小
plt.figure(figsize=(20, 8), dpi=80)
plt.plot(x, y_1, label="自己", color="orange", linestyle=":")
plt.plot(x, y_2, label="同桌", color="cyan", linestyle="--")

# 设置x轴刻度
_xtick_labels = ["{}岁".format(i) for i in x]
plt.xticks(x, _xtick_labels, rotation=45)
plt.yticks(range(0, 9))

# 绘制网格,alpha调整透明度
plt.grid(alpha=0.4)

# 添加图例
plt.legend(loc="upper left")

# 添加水印
plt.text(x=18, y=6, s="交友曲线", fontsize=50, color="gray")

# 设置标题
plt.xlabel("年龄")
plt.ylabel("个数")
plt.title("交友数量走势")

plt.show()
