from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# 线性情况
X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow')
'''
ax = plt.gca()  # 获取当前的子图
# 制作网格
xlim = ax.get_xlim()
ylim = ax.get_ylim()
axisx = np.linspace(xlim[0], xlim[1], 30)  # (30,)
axisy = np.linspace(ylim[0], ylim[1], 30)  # (30,)
axisy, axisx = np.meshgrid(axisy, axisx)  # axisy:(30, 30), axisx:(30, 30), 生成网格点
xy = np.vstack([axisx.ravel(), axisy.ravel()]).T  # (900,2)，np.vstack:按行进行拼接
# 建立模型
clf = SVC(kernel='linear').fit(X, y)
Z = clf.decision_function(xy).reshape(axisx.shape)  # decision_function，返回每个输入的样本所对应的到决策边界的距离
# 画决策边界和平行于决策边界的超平面
ax.contour(axisx, axisy, Z
           , colors='k'
           , levels=[-1, 0, 1]
           , alpha=0.5
           , linestyles=['--', '-', '--'])
ax.set_xlim(xlim)
ax.set_ylim(ylim)
'''


# 将上述过程包装成函数
def plot_svc_decision_function(model, ax=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# 绘制图形
clf = SVC(kernel='linear').fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow')
plot_svc_decision_function(clf)
plt.show()
