# coding=utf-8
from matplotlib import pyplot as plt
import random
import matplotlib

font = {'family': 'Microsoft YaHei',
        'weight': 'bold',
        'size': '10'}
matplotlib.rc('font', **font)

x = range(0, 120)
y = [random.randint(20, 35) for i in range(120)]

plt.figure(figsize=(20, 8), dpi=80)

plt.plot(x, y)

# 调整x轴的刻度
_x = list(x)[::10]
_xtick_labels = ["hello,{}".format(i) for i in _x]
plt.xticks(_x, _xtick_labels, rotation=45)  # rotation表示旋转的度数

# 添加描述信息
plt.xlabel("时间")
plt.ylabel("温度 单位()")
plt.title("10点到12点每分钟的气温变化情况")

plt.show()
