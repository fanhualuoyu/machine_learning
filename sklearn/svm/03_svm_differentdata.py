import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn.datasets import make_circles, make_moons, make_blobs, make_classification

n_samples = 100
# 四个数据集
datasets = [
    make_moons(n_samples=n_samples, noise=0.2, random_state=0),
    make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1),  # factor:表示里圆和外圆的距离
    make_blobs(n_samples=n_samples, centers=2, random_state=5),
    make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0, random_state=5)
    # n_features(特征数) = n_informative(多信息特征的个数) + n_redundant(冗余信息) + n_repeated(重复信息)
]
Kernel = ['linear', 'poly', 'rbf', 'sigmoid']  # 核函数

'''
title = ['moons', 'circles', 'blobs', 'classification']  # 数据名称
# 观察四个数据图
plt.figure(figsize=(10, 8))
for i, k in enumerate(datasets):
    X, y = k
    plt.subplot(2, 2, i+1)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow')
    plt.title(title[i])
plt.show()
'''

# 构建子图
rows = len(datasets)
cols = len(Kernel) + 1
fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
# 数据集循环
for ds_cnt, (X, Y) in enumerate(datasets):
    ax = axes[ds_cnt, 0]
    if ds_cnt == 0:
        ax.set_title('Input data')
    ax.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired, edgecolors='k')
    ax.set_xticks(())
    ax.set_yticks(())
    # 核函数的循环
    for est_idx, kernel in enumerate(Kernel):
        # 定义子图位置
        ax = axes[ds_cnt, est_idx + 1]
        # 建立模型
        clf = svm.SVC(kernel=kernel, gamma=2).fit(X, Y)  # gamma:表示核函数的系数
        score = clf.score(X, Y)
        # 绘制图像本身分布的散点图
        ax.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired, edgecolors='k')
        # 绘制支持向量
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=50, facecolors='none', zorder=10,
                   edgecolors='k')
        # 绘制决策边界
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        # np.mgrid:合并之前使用的np.linspace和np.meshgrid的用法.[起始值:结束值:步长],如果步长是复数,则整数部分就是起始值和结束值之间创建的点的数量，并且结束值被包含在内
        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        # np.c_,类似于np.vstack的功能
        z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
        # 绘制等高线
        ax.contour(XX, YY, z, colors='k', linestyles=['--', '-', '--'], levels=[-1, 0, 1])
        # 设置坐标轴不显示
        ax.set_xticks(())
        ax.set_yticks(())
        # 将标题放在第一行的顶上
        if ds_cnt == 0:
            ax.set_title(kernel)
        # 为每张图添加分类的分数
        ax.text(0.95, 0.06, ('%.2f' % score).lstrip('0'), fontsize=15, # .lstrip(s):去掉字符串左侧头部为0的值
                bbox=dict(boxstyle='round', alpha=0.8, facecolor='white'),  # 增加外边框
                transform=ax.transAxes, # 确定文字所对应的坐标轴，就是ax子图的坐标轴本身
                horizontalalignment='right')  # 水平对齐方式
plt.tight_layout()  # 自动调整子图的参数，使之填充整个图像区域
plt.show()