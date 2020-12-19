from sklearn.datasets import make_circles
from sklearn.svm import SVC
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

# 非线性情况
X, y = make_circles(100, factor=0.1, noise=.1)
X.sum(1)
# 为非线性数据增加维度并绘制3D图像
r = np.exp(-(X**2).sum(1))    # .sum(1):对每一行求和
# rlim = np.linspace(min(r), max(r), 0.2)


# 定义一个绘制三维图像的函数
# elev:上下旋转的角度  azim:平行旋转的角度
def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='rainbow')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')
    plt.show()


clf = SVC(kernel='rbf').fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow')


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


plot_svc_decision_function(clf)
plt.show()
