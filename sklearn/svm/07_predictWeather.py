# SVM处理多分类问题
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import re
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import roc_curve as ROC
from sklearn.metrics import accuracy_score as AC
from sklearn.svm import SVC
from math import radians, sin, cos, acos
from time import time

weather = pd.read_csv(r'D:\web\python\sklearn\svm\weather.csv', index_col=0)  # index_col=0:第一列为索引值
X = weather.iloc[:, :-1]  # 获取特征矩阵
Y = weather.iloc[:, -1]  # 获取标签
# 划分训练集和测试集X_train:(99535,21), X_test:(42658, 21), Y_train:(99535,), Y_test:(42658,)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=420)
for i in [X_train, X_test, Y_train, Y_test]:  # 恢复每一个索引
    i.index = range(i.shape[0])
encorder = LabelEncoder().fit(Y_train)  # 将yes和no替换成1和0
Y_train = pd.DataFrame(encorder.transform(Y_train))
Y_test = pd.DataFrame(encorder.transform(Y_test))

X_train.drop(index=71737)
Y_train.drop(index=71737)
X_test = X_test.drop(index=[19646, 29632])
Y_test = Y_test.drop(index=[19646, 29632])
for i in [X_train, X_test, Y_train, Y_test]:
    i.index = range(i.shape[0])

# 处理困难特征:日期
# 将日期转换为"今天是否会下雨"这个特征
X_train.loc[X_train['Rainfall'] >= 1, 'RainToday'] = 'Yes'
X_train.loc[X_train['Rainfall'] < 1, 'RainToday'] = 'No'
X_train.loc[X_train['Rainfall'] == np.nan, 'RainToday'] = np.nan
X_test.loc[X_test['Rainfall'] >= 1, 'RainToday'] = 'Yes'
X_test.loc[X_test['Rainfall'] < 1, 'RainToday'] = 'No'
X_test.loc[X_test['Rainfall'] == np.nan, 'RainToday'] = np.nan
# 将日期转换为"月份或季节"这个特征
X_train['Date'] = X_train['Date'].apply(lambda x: int(x.split('-')[1]))
X_train = X_train.rename(columns={'Date': 'Month'})
X_test['Date'] = X_test['Date'].apply(lambda x: int(x.split('-')[1]))
X_test = X_test.rename(columns={'Date': 'Month'})

# 处理困难特征:地点
# 将不同城市打包到同一个气候中
cityll = pd.read_csv(r'D:\web\python\sklearn\svm\cityll.csv', index_col=0)  # 城市的经纬度
city_climate = pd.read_csv(r'D:\web\python\sklearn\svm\Cityclimate.csv')  # 城市的气候
# 去掉度数符号
cityll["Latitudenum"] = cityll["Latitude"].apply(lambda x: float(x[:-1]))
cityll["Longitudenum"] = cityll["Longitude"].apply(lambda x: float(x[:-1]))
# 观察一下所有的经纬度方向都是一致的，全部是南纬，东经，因为澳大利亚在南半球，东半球.所以经纬度的方向我们可以舍弃了
citylld = cityll.iloc[:, [0, 5, 6]]
# 将city_climate中的气候添加到我们的citylld中
citylld["climate"] = city_climate.iloc[:, -1]
samplecity = pd.read_csv(r'D:\web\python\sklearn\svm\samplecity.csv', index_col=0)
# 我们对samplecity也执行同样的处理：去掉经纬度中度数的符号，并且舍弃我们的经纬度的方向
samplecity["Latitudenum"] = samplecity["Latitude"].apply(lambda x: float(x[:-1]))
samplecity["Longitudenum"] = samplecity["Longitude"].apply(lambda x: float(x[:-1]))
samplecityd = samplecity.iloc[:, [0, 5, 6]]
# 首先使用radians将角度转换成弧度
citylld.loc[:, "slat"] = citylld.iloc[:, 1].apply(lambda x: radians(x))
citylld.loc[:, "slon"] = citylld.iloc[:, 2].apply(lambda x: radians(x))
samplecityd.loc[:, "elat"] = samplecityd.iloc[:, 1].apply(lambda x: radians(x))
samplecityd.loc[:, "elon"] = samplecityd.iloc[:, 2].apply(lambda x: radians(x))
for i in range(samplecityd.shape[0]):
    slat = citylld.loc[:, "slat"]
    slon = citylld.loc[:, "slon"]
    elat = samplecityd.loc[i, "elat"]
    elon = samplecityd.loc[i, "elon"]
    # 计算我们样本上的地点到每个澳大利亚主要城市的距离，而离我们的样本地点最近的那个澳大利亚主要城市的气候，就是我们样本点的气候
    dist = 6371.01 * np.arccos(np.sin(slat) * np.sin(elat) + np.cos(slat) * np.cos(elat) * np.cos(slon.values - elon))
    city_index = np.argsort(dist)[0]  # np.argsort():将元素从小到大排序，然后返回每个元素对应的索引
    # 每次计算后，取距离最近的城市，然后将最近的城市和城市对应的气候都匹配到samplecityd中
    samplecityd.loc[i, "closest_city"] = citylld.loc[city_index, "City"]  # 找出该样本对应的最近的城市
    samplecityd.loc[i, "climate"] = citylld.loc[city_index, "climate"]  # 找出样本对应的气候
# 取出样本城市所对应的气候，并保存
locafinal = samplecityd.iloc[:, [0, -1]]
locafinal.columns = ['Location', 'Climate']
locafinal = locafinal.set_index(keys="Location")
# x.strip():去掉首尾的空字符. re.sub(换掉的符号，代替的符号，要处理的字符串)
X_train["Location"] = X_train["Location"].map(locafinal.iloc[:, 0]).apply(lambda x: re.sub(",", "", x.strip()))
X_test["Location"] = X_test["Location"].map(locafinal.iloc[:, 0]).apply(lambda x: re.sub(",", "", x.strip()))
# 使用新列名代替
X_train = X_train.rename(columns={"Location": "Climate"})
X_test = X_test.rename(columns={"Location": "Climate"})

# 处理分类型变量：缺失值
# 找出所有的分类型特征
cate = X_train.columns[X_train.dtypes == 'object'].tolist()
cloud = ['Cloud9am', 'Cloud3pm']  # 虽然用数字表示，但是本质上还是为分类型特征的云层遮蔽程度
cate = cate + cloud

# 利用训练集中的众数对训练集和测试集进行填充
si = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
si.fit(X_train.loc[:, cate])
X_train.loc[:, cate] = si.transform(X_train.loc[:, cate])
X_test.loc[:, cate] = si.transform(X_test.loc[:, cate])

# 将分类型变量进行编码
oe = OrdinalEncoder()
oe.fit(X_train.loc[:, cate])
X_train.loc[:, cate] = oe.transform(X_train.loc[:, cate])
X_test.loc[:, cate] = oe.transform(X_test.loc[:, cate])

# 处理连续型变量
col = X_train.columns.tolist()
for i in cate:
    col.remove(i)

# 利用均值来填充连续型变量的缺失值
impmean = SimpleImputer(missing_values=np.nan, strategy='mean')
impmean = impmean.fit(X_train.loc[:, col])
X_train.loc[:, col] = impmean.transform(X_train.loc[:, col])
X_test.loc[:, col] = impmean.transform(X_test.loc[:, col])

# 对连续型变量要进行归一化(SVM处理时要进行无量纲化)
ss = StandardScaler()
ss = ss.fit(X_train.loc[:, col])
X_train.loc[:, col] = ss.transform(X_train.loc[:, col])
X_test.loc[:, col] = ss.transform(X_test.loc[:, col])

# 建立模型和模型评估
Y_train = Y_train.iloc[:, 0].ravel()
Y_Test = Y_test.iloc[:, 0].ravel()
for kernel in ["linear", "poly", "rbf", "sigmoid"]:
    clf = SVC(kernel=kernel,
              gamma="auto",
              degree=1,
              cache_size=5000).fit(X_train, Y_train)
    result = clf.predict(X_test)
    score = clf.score(X_test, Y_test)
    recall = recall_score(Y_test, result)
    auc = roc_auc_score(Y_test, clf.decision_function(X_test))
    print("%s 's testing accuracy %f, recall is %f', auc is %f" % (kernel, score, recall, auc))
print('-' * 20)
'''
# 最求更高的recall
for kernel in ["linear", "poly", "rbf", "sigmoid"]:  # 求出最好的核函数
    clf = SVC(kernel=kernel
              , gamma="auto"
              , degree=1
              , cache_size=5000
              , class_weight="balanced"
              ).fit(X_train, Y_train)
    result = clf.predict(X_test)
    score = clf.score(X_test, Y_test)
    recall = recall_score(Y_test, result)
    auc = roc_auc_score(Y_test, clf.decision_function(X_test))
    print("%s 's testing accuracy %f, recall is %f', auc is %f" % (kernel, score, recall, auc))
clf = SVC(kernel="linear"  # 修改class_weight来最大化recall
          , gamma="auto"
          , cache_size=5000
          , class_weight={1: 10}
          ).fit(X_train, Y_train)
result = clf.predict(X_test)
score = clf.score(X_test, Y_test)
recall = recall_score(Y_test, result)
auc = roc_auc_score(Y_test, clf.decision_function(X_test))
print("testing accuracy %f, recall is %f', auc is %f" % (score, recall, auc))
print('-' * 20)

# 最求更高的准确率
clf = SVC(kernel="linear"
          , gamma="auto"
          , cache_size=5000
          ).fit(X_train, Y_train)
result = clf.predict(X_test)
cm = CM(Y_test, result, labels=(1, 0))
specificity = cm[1, 1] / cm[1, :].sum()
irange = np.linspace(0.01, 0.05, 10)
for i in irange:
    clf = SVC(kernel="linear"
              , gamma="auto"
              , cache_size=5000
              , class_weight={1: 1 + i}
              ).fit(X_train, Y_train)
    result = clf.predict(X_test)
    score = clf.score(X_test, Y_test)
    recall = recall_score(Y_test, result)
    auc = roc_auc_score(Y_test, clf.decision_function(X_test))
    print("under ratio 1:%f testing accuracy %f, recall is %f', auc is %f" % (1 + i, score, recall, auc))
print('-' * 20)

# 追求平衡，调整核函数的C值
C_range = np.linspace(0.01, 20, 20)
recallall = []
aucall = []
scoreall = []
for C in C_range:
    times = time()
    clf = SVC(kernel="linear", C=C, cache_size=5000
              , class_weight="balanced"
              ).fit(X_train, Y_train)
    result = clf.predict(X_test)
    score = clf.score(X_test, Y_test)
    recall = recall_score(Y_test, result)
    auc = roc_auc_score(Y_test, clf.decision_function(X_test))
    recallall.append(recall)
    aucall.append(auc)
    scoreall.append(score)
    print("under C %f, testing accuracy is %f,recall is %f', auc is %f" % (C, score, recall, auc))
    print(max(aucall), C_range[aucall.index(max(aucall))])
plt.figure()
plt.plot(C_range, recallall, c="red", label="recall")
plt.plot(C_range, aucall, c="black", label="auc")
plt.plot(C_range, scoreall, c="orange", label="accuracy")
plt.legend()
plt.show()
# 由上一步选出的最佳的C值代入模型
clf = SVC(kernel="linear", C=3.1663157894736838, cache_size=5000
          , class_weight="balanced"
          ).fit(X_train, Y_train)
result = clf.predict(X_test)
score = clf.score(X_test, Y_test)
recall = recall_score(Y_test, result)
auc = roc_auc_score(Y_test, clf.decision_function(X_test))
print("testing accuracy %f,recall is %f', auc is %f" % (score, recall, auc))
# 调整阈值对模型进行修改
FPR, Recall, thresholds = ROC(Y_test, clf.decision_function(X_test), pos_label=1)
area = roc_auc_score(Y_test, clf.decision_function(X_test))
plt.figure()
plt.plot(FPR, Recall, color='red', label='ROC curve (area = %0.2f)' % area)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
maxindex = (Recall - FPR).tolist().index(max(Recall - FPR))
clf = SVC(kernel="linear", C=3.1663157894736838, cache_size=5000
          , class_weight="balanced"
          ).fit(X_train, Y_train)
prob = pd.DataFrame(clf.decision_function(X_test))
prob.loc[prob.iloc[:, 0] >= thresholds[maxindex], "y_pred"] = 1
prob.loc[prob.iloc[:, 0] < thresholds[maxindex], "y_pred"] = 0
# 检查模型本身的准确度
score = AC(Y_test, prob.loc[:, "y_pred"].values)
recall = recall_score(Y_test, prob.loc[:, "y_pred"])
print("testing accuracy %f,recall is %f" % (score, recall))

'''
