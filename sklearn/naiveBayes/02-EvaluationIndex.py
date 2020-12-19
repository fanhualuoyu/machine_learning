from sklearn.datasets import load_digits
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import log_loss
import pandas as pd
import matplotlib as plt
from sklearn.datasets import make_classification as mc

# 布里尔分数Brier Score
# 布里尔分数的范围是从0到1，分数越高则预测结果越差劲，校准程度越差，因此布里尔分数越接近0越好
digits = load_digits()
X, y = digits.data, digits.target
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)
gnb = GaussianNB().fit(Xtrain, Ytrain)
Y_pred = gnb.predict(Xtest)
prob = gnb.predict_proba(Xtest)  # 返回预测某标签的概率
# 注意，第一个参数是真实标签，第二个参数是预测出的概率值
# 在二分类情况下，接口predict_proba会返回两列，但SVC的接口decision_function却只会返回一列
brier_score_loss(Ytest, prob[:, 1], pos_label=1)
# 我们的pos_label与prob中的索引一致，就可以查看这个类别下的布里尔分数是多少

logi = LR(C=1., solver='lbfgs', max_iter=3000, multi_class="auto").fit(Xtrain, Ytrain)
svc = SVC(kernel="linear", gamma=1).fit(Xtrain, Ytrain)
brier_score_loss(Ytest, logi.predict_proba(Xtest)[:, 1], pos_label=1)
# 由于SVC的置信度并不是概率，为了可比性，我们需要将SVC的置信度“距离”归一化，压缩到[0,1]之间
svc_prob = (svc.decision_function(Xtest) -
            svc.decision_function(Xtest).min()) / (svc.decision_function(Xtest).max() -
                                                   svc.decision_function(Xtest).min())
brier_score_loss(Ytest, svc_prob[:, 1], pos_label=1)

name = ["Bayes", "Logistic", "SVC"]
color = ["red", "black", "orange"]
df = pd.DataFrame(index=range(10), columns=name)
for i in range(10):
    df.loc[i, name[0]] = brier_score_loss(Ytest, prob[:, i], pos_label=i)
    df.loc[i, name[1]] = brier_score_loss(Ytest, logi.predict_proba(Xtest)
    [:, i], pos_label=i)
    df.loc[i, name[2]] = brier_score_loss(Ytest, svc_prob[:, i], pos_label=i)
for i in range(df.shape[1]):
    plt.plot(range(10), df.iloc[:, i], c=color[i])
plt.legend()
plt.show()

# 对数似然函数Log Loss:由于是损失，因此对数似然函数的取值越小，则证明概率估计越准确，模型越理想。
log_loss(Ytest, prob)
log_loss(Ytest, logi.predict_proba(Xtest))
log_loss(Ytest, svc_prob)

# 可靠性曲线Reliability Curve:一个模型/算法的概率校准曲线越靠近对角线越好
X, y = mc(n_samples=100000, n_features=20  # 总共20个特征
          , n_classes=2  # 标签为2分类
          , n_informative=2  # 其中两个代表较多信息
          , n_redundant=10  # 10个都是冗余特征
          , random_state=42)
# 样本量足够大，因此使用1%的样本作为训练集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y
                                                , test_size=0.99
                                                , random_state=42)
gnb = GaussianNB()
gnb.fit(Xtrain, Ytrain)
y_pred = gnb.predict(Xtest)
prob_pos = gnb.predict_proba(Xtest)[:, 1]  # 我们的预测概率 - 横坐标
# Ytest - 我们的真实标签 - 横坐标
# 在我们的横纵表坐标上，概率是由顺序的（由小到大），为了让图形规整一些，我们要先对预测概率和真实标签按照预测概率进行一个排序，这一点我们通过DataFrame来实现
df = pd.DataFrame({"ytrue": Ytest[:500], "probability": prob_pos[:500]})
df = df.sort_values(by="probability")
df.index = range(df.shape[0])
fig = plt.figure()
ax1 = plt.subplot()
ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")  # 得做一条对角线来对比呀
ax1.plot(df["probability"], df["ytrue"], "s-", label="%s (%1.3f)" % ("Bayes", clf_score))
ax1.set_ylabel("True label")
ax1.set_xlabel("predcited probability")
ax1.set_ylim([-0.05, 1.05])
ax1.legend()
plt.show()
