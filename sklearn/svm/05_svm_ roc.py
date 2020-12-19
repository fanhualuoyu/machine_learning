# 解决样本不均衡的问题
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

class_1 = 500  # 类别1有500个样本点
class_2 = 50  # 类别2有50个样本点
centers = [[0.0, 0.0], [2.0, 2.0]]  # 两个样本类别的中心
clusters_std = [1.5, 0.5]  # 设定两个类别的方差，通常来说，样本量比较大的类别会更加松散
X, y = make_blobs(n_samples=[class_1, class_2],
                  centers=centers,
                  cluster_std=clusters_std,
                  random_state=0, shuffle=False)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', s=10)  # s表示点的大小

# 不设定class_weight
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)
# 设定class_weight
wclf = svm.SVC(kernel='linear', class_weight={1: 10})
wclf.fit(X, y)

# 首先要有数据分布
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="rainbow", s=10)
ax = plt.gca()  # 获取当前的子图，如果不存在，则创建新的子图
# 绘制决策边界的第一步：要有网格
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
# 第二步：找出我们的样本点到决策边界的距离
Z_clf = clf.decision_function(xy).reshape(XX.shape)
a = ax.contour(XX, YY, Z_clf, colors='black', levels=[0], alpha=0.5, linestyles=['-'])
Z_wclf = wclf.decision_function(xy).reshape(XX.shape)
b = ax.contour(XX, YY, Z_wclf, colors='red', levels=[0], alpha=0.5, linestyles=['-'])
# 第三步：画图例
plt.legend([a.collections[0], b.collections[0]], ["non weighted", "weighted"],
           loc="upper right")
plt.show()

# Confusion Matrix:混淆矩阵
# 精确度(查准率):在被识别为正样本的样本中，确实为正类别的比例
precision_clf = (y[y == clf.predict(X)] == 1).sum()/(clf.predict(X) == 1).sum()
precision_wclf = (y[y == wclf.predict(X)] == 1).sum()/(wclf.predict(X) == 1).sum()
# 召回率(查全率):在所有的正样本中，被正确识别为正类别的比例
recall_clf = (y[y == clf.predict(X)] == 1).sum()/(y == 1).sum()
recall_wclf = (y[y == wclf.predict(X)] == 1).sum()/(y == 1).sum()
# 假负率
fnr_clf = 1 - recall_clf
fnr_wclf = 1 - recall_wclf
# 特异度(真负率):所有负样本中，被正确预测为负样本所占的比例
tnr_clf = (y[y == clf.predict(X)] == 0).sum()/(y == 0).sum()
tnr_wclf = (y[y == wclf.predict(X)] == 0).sum()/(y == 0).sum()
# 假正率:所有负样本中，被预测为正样本所占的比例
fpr_clf = 1 - tnr_clf
fpr_wclf = 1 - tnr_wclf
