from sklearn import datasets
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import copy


# 查找每个点的邻域点
def find_neighbor(j, X, eps):
    N = list()
    for i in range(X.shape[0]):
        temp = np.sqrt(np.sum(np.square(X[j] - X[i])))  # 计算欧式距离
        if temp <= eps:
            N.append(i)
    return set(N)


# DBSCAN算法
def DBSCAN(X, eps, minPts):  # X:数据集. eps:邻域. minPts:邻域内的最低样本数
    neighbor_list = []  # 用来保存每个数据的邻域
    omega_list = []  # 核心对象集合
    gama = set([x for x in range(len(X))])  # 初始时将所有点标记为未访问
    cluster = [-1 for _ in range(len(X))]  # 聚类
    for i in range(len(X)):
        neighbor_list.append(find_neighbor(i, X, eps))
        if len(neighbor_list[-1]) >= minPts:  # 点的邻域样本数符合要求为核心对象
            omega_list.append(i)
    omega_list = set(omega_list)  # 转化为集合便于操作
    k = -1  # 数据标签
    while len(omega_list) > 0:
        gama_old = copy.deepcopy(gama)
        j = random.choice(list(omega_list))  # 随机选择一个核心对象
        k = k + 1
        Q = list()  # 来保存选出的核心对象的同类点
        Q.append(j)
        gama.remove(j)
        while len(Q) > 0:
            q = Q[0]
            Q.remove(q)
            if len(neighbor_list[q]) >= minPts:
                delta = neighbor_list[q] & gama  # 取出两个元组的交集
                deltaList = list(delta)
                for i in range(len(delta)):
                    Q.append(deltaList[i])
                    gama = gama - delta
        Ck = gama_old - gama  # 将已经划分好的点去除
        CkList = list(Ck)
        for i in range(len(Ck)):
            cluster[CkList[i]] = k  # 给同一类的点坐上标记
        omega_list = omega_list - Ck    # 将这些已经分好的点从核心对象中去除
    return cluster


# 获取两个数据集
x1, y1 = datasets.make_circles(n_samples=200, factor=0.6, noise=0.02)
# centers:样本中心, cluster_std:标准差
x2, y2 = datasets.make_blobs(n_samples=400, n_features=2, centers=[[1.2, 1.2]], cluster_std=[[0.1]], random_state=9)
# 将两个数据集合并
X = np.concatenate((x1, x2))    # (600,2)
# 初始化参数
eps = 0.08
min_Pts = 10
C = DBSCAN(X, eps, min_Pts)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=C)
plt.show()