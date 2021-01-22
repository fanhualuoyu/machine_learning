import pandas as pd
import numpy as np

t1 = pd.DataFrame(np.arange(12).reshape(3, 4))
print("t1:\n", t1)

t2 = pd.DataFrame(np.arange(12).reshape(3, 4), index=list("abc"), columns=list("wxyz"))
print("t2:\n", t2)

# 字典转换为DataFrame
d1 = {"name": ["zh", "wa"], "age": [20, 19], "tel": [10086, 10010]}
t3 = pd.DataFrame(d1)
print("t3:\n", t3)

d2 = [{"name": "zs", "age": 17, "tel": 10010}, {"name": "ls", "tel": 10011}, {"name": "we", "age": 10012}]
t4 = pd.DataFrame(d2)
print("t4:\n", t4)

# DataFrame中的基础属性
print("t3.index:", t3.index)
print("t3.columns:", t3.columns)
print("t3.values:", t3.values)
print("t3.shape:", t3.shape)
print("t3.dtypes:", t3.dtypes)
print("t3.ndim:", t3.ndim)
print("t3.head:\n", t3.head(1))
print("t3.tail:\n", t3.tail(1))
print("t3.info:\n", t3.info())
print("t3.describe:\n", t3.describe())

# 排序方法
t5 = t3.sort_values(by="age")
print("t5(排序):", t5)

# pandas取行或取列
# - 方括号写数组，表示取行，对行进行操作
# - 写字符串，表示取列索引，对列进行操作
print("t5(取行):", t5[:2])
print("t5(取列):", t5["age"])

# loc和iloc
print("t2.loc:\n", t2.loc[["b", "c"], "w"])

print("t2.iloc[1,:]:\n", t2.iloc[1, :])

# 缺失数据的处理
t2.iloc[1:, :2] = np.nan
t6 = pd.isnull(t2)
print("t6(isnull):\n", t6)
t7 = pd.notnull(t2)
print("t6(notnull):\n", t7)
