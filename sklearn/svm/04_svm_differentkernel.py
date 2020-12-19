from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from time import time
import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

# 加载数据集
data = load_breast_cancer()
X = data.data  # (256, 30)
y = data.target  # (256,)
# np.unique(y):查看y有多少不同值
Kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# 将数据归一化
X = StandardScaler().fit_transform(X)
# 将数据分为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=420)
# 分别看不同核函数的训练
for kernel in Kernels:
    start = time()
    # degree:多核函数的阶数.cache_size:分配多少内存来进行运算
    clf = SVC(kernel=kernel, gamma='auto', degree=1, cache_size=5000).fit(X_train, Y_train)
    print('The accuracy under kernel %s is %f' % (kernel, clf.score(X_test, Y_test)))
    print(datetime.datetime.fromtimestamp(time() - start).strftime('%M:%S:%f'))

# 调整核函数的参数
'''
    degree:多项式核函数的次数（'poly'），如果核函数没有选择"poly"，这个参数会被忽略
    gamma:核函数的系数，仅在参数Kernel的选项为”rbf","poly"和"sigmoid”的时候有效
            输入“auto"，自动使用1/(n_features)作为gamma的取值
            输入"scale"，则使用1/(n_features * X.std())作为gamma的取值
            输入"auto_deprecated"，则表示没有传递明确的gamma值（不推荐使用）
    coef0:核函数中的常数项，它只在参数kernel为'poly'和'sigmoid'的时候有效        
'''
# 高斯径向基核函数rbf的参数gamma
score = []
gamma_range = np.logspace(-10, 1, 50)  # np.logspace(0,3,4):对数等比数列. eg: array([   1.,   10.,  100., 1000.])
for i in gamma_range:
    clf = SVC(kernel='rbf', gamma=i, cache_size=5000).fit(X_train, Y_train)
    score.append(clf.score(X_test, Y_test))
print(max(score), gamma_range[score.index(max(score))])
plt.plot(gamma_range, score)
plt.show()

# 三个参数对多项式核函数的影响
gamma_range = np.logspace(-10, 1, 20)
ceof0_range = np.linspace(0, 5, 10)
param_grid = dict(gamma=gamma_range, cef0=ceof0_range)
# StratifiedShuffleSplit: n_splits:train/test要生成多少组，test_size:每组中测试集的数目
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=420)
# 为想要调参的参数设定一组候选值，然后网格搜索会穷举各种参数组合，根据设定的评分机制找到最好的那一组设置
grid = GridSearchCV(SVC(kernel='poly', degree=1, cache_size=5000), param_grid=param_grid, cv=cv)
grid.fit(X, y)
print("The best parameters are %s with a score of %0.5f" % (grid.best_params_, grid.best_score_))

# 硬间隔与软间隔:调整C
# 调线性核函数
score = []
C_range = np.linspace(0.01, 30, 50)
for i in C_range:
    clf = SVC(kernel="linear", C=i, cache_size=5000).fit(X_train, Y_train)
score.append(clf.score(X_test, Y_test))
print(max(score), C_range[score.index(max(score))])
plt.plot(C_range, score)
plt.show()

# 换rbf
score = []
C_range = np.linspace(0.01, 30, 50)
for i in C_range:
    clf = SVC(kernel="rbf", C=i, gamma=0.012742749857031322, cache_size=5000).fit(X_train, Y_train)
score.append(clf.score(X_test, Y_test))
print(max(score), C_range[score.index(max(score))])
plt.plot(C_range, score)
plt.show()

# 进一步细化
score = []
C_range = np.linspace(5, 7, 50)
for i in C_range:
    clf = SVC(kernel="rbf", C=i, gamma=0.012742749857031322, cache_size=5000).fit(X_train, Y_train)
score.append(clf.score(X_test, Y_test))
print(max(score), C_range[score.index(max(score))])
plt.plot(C_range, score)
plt.show()
