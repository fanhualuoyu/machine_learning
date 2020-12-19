# 概率与阈值
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression as LogiR
# 混淆矩阵，精确率，召回率
from sklearn.metrics import confusion_matrix as CM, precision_score as P, recall_score as R
# 计算ROC曲线的横坐标假正率FPR，纵坐标Recall和对应的阈值的类
from sklearn.metrics import roc_curve
# 计算AUC面积的类
from sklearn.metrics import roc_auc_score as AUC

# 建立数据集
class_1_ = 7
class_2_ = 4
centers_ = [[0.0, 0.0], [1, 1]]
clusters_std = [0.5, 1]
X_, y_ = make_blobs(n_samples=[class_1_, class_2_],
                    centers=centers_,
                    cluster_std=clusters_std,
                    random_state=0, shuffle=False)
plt.scatter(X_[:, 0], X_[:, 1], c=y_, cmap="rainbow", s=30)

# 建模，调用概率
clf_lo = LogiR().fit(X_, y_)
prob = clf_lo.predict_proba(X_)  # 返回预测某标签的概率
prob = pd.DataFrame(prob)
prob.columns = ["0", "1"]

# 手动调节阈值，来改变我们的模型效果
for i in range(prob.shape[0]):
    if prob.loc[i, '1'] > 0.5:  # 判断(i, 1)这个位置的值是否大于0.5
        prob.loc[i, 'pred'] = 1
    else:
        prob.loc[i, "pred"] = 0
prob["y_true"] = y_
prob = prob.sort_values(by="1", ascending=False)  # ascending:False表示降序

# 使用混淆矩阵查看结果
CM(prob.loc[:, "y_true"], prob.loc[:, "pred"], labels=[1, 0])
# Precision和Recall
P(prob.loc[:, "y_true"], prob.loc[:, "pred"], labels=[1, 0])
R(prob.loc[:, "y_true"], prob.loc[:, "pred"], labels=[1, 0])

# 使用最初的X和y，样本不均衡的这个模型
class_1 = 500  # 类别1有500个样本
class_2 = 50  # 类别2只有50个
centers = [[0.0, 0.0], [2.0, 2.0]]  # 设定两个类别的中心
clusters_std = [1.5, 0.5]  # 设定两个类别的方差，通常来说，样本量比较大的类别会更加松散
X, y = make_blobs(n_samples=[class_1, class_2],
                  centers=centers,
                  cluster_std=clusters_std,
                  random_state=0, shuffle=False)
# 看看数据集长什么样
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="rainbow", s=10)
# 其中红色点是少数类，紫色点是多数类
clf_proba = svm.SVC(kernel="linear", C=1.0, probability=True).fit(X, y)
# clf_proba.predict_proba(X)
# clf_proba.predict_proba(X).shape
# clf_proba.decision_function(X)
# clf_proba.decision_function(X).shape

'''
# 绘制SVM的ROC曲线
cm = CM(prob.loc[:, "y_true"], prob.loc[:, "pred"], labels=[1, 0])
# FPR
cm[1, 0] / cm[1, :].sum()
# Recall
cm[0, 0] / cm[0, :].sum()
'''

# 开始绘图
recall = []
FPR = []
probrange = np.linspace(clf_proba.predict_proba(X)[:, 1].min(),
                        clf_proba.predict_proba(X)[:, 1].max(),
                        num=50, endpoint=False)  # 设置阈值
for i in probrange:
    y_predict = []
    for j in range(X.shape[0]):
        if clf_proba.predict_proba(X)[j, 1] > i:
            y_predict.append(1)
        else:
            y_predict.append(0)
    cm = CM(y, y_predict, labels=[1, 0])
    recall.append(cm[0, 0] / cm[0, :].sum())
    FPR.append(cm[1, 0] / cm[1, :].sum())
recall.sort()
FPR.sort()
plt.plot(FPR, recall, c="red")  # 绘制ROC曲线，希望FPR靠近左，recall靠近上边(左上方效果不错)
plt.plot(probrange + 0.05, probrange + 0.05, c="black", linestyle="--")
plt.show()

# sklearn中的ROC曲线和AUC面积
# 获取FPR,recall和阈值
FPR, recall, thresholds = roc_curve(y, clf_proba.decision_function(X), pos_label=1)  # pos_label:被认为是正样本的类别
# 计算AUC面积
area = AUC(y, clf_proba.decision_function(X))
plt.figure()
plt.plot(FPR, recall, color='red',
         label='ROC curve (area = %0.2f)' % area)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# 利用ROC曲线找出最佳阈值：寻找Recall和FPR差距最大的点，这个点也叫做约登指数
maxindex = (recall - FPR).tolist().index(max(recall - FPR))
thresholds[maxindex]  # 最大阈值

# 把上述代码放入这段代码中：
plt.figure()
plt.plot(FPR, recall, color='red',
         label='ROC curve (area = %0.2f)' % area)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.scatter(FPR[maxindex], recall[maxindex], c="black", s=30)   # 看这个点在哪里
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

'''
# 属性n_support_:调用每个类别下的支持向量的数目
clf_proba.n_support_
# 属性coef_：每个特征的重要性，这个系数仅仅适合于线性核
clf_proba.coef_
# 属性intercept_：查看生成的决策边界的截距
clf_proba.intercept_
# 属性dual_coef_：查看生成的拉格朗日乘数
clf_proba.dual_coef_
clf_proba.dual_coef_.shape
# 注意到这个属性的结构了吗？来看看查看支持向量的属性
clf_proba.support_vectors_
clf_proba.support_vectors_.shape
'''