import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# join:行合并
df1 = pd.DataFrame(np.ones((2, 4)), index=["A", "B"], columns=list("abcd"))
print("df1:\n", df1)
df2 = pd.DataFrame(np.zeros((3, 3)), index=["A", "B", "C"], columns=list("xyz"))
print("df1.join(df2):\n", df1.join(df2))
print("df2.join(df1):\n", df2.join(df1))

# merge:列合并
df3 = pd.DataFrame(np.zeros((3, 3)), columns=list("fax"))
# 默认取交集
print("df1.merge(df3,on=\"a\"):\n", df1.merge(df3, on="a"))
df3.loc[1, "a"] = 1
print("df1.merge(df3,on=\"a\")(修改后):\n", df1.merge(df3, on="a"))
# 外连接
print("df1.merge(df3,on=\"a\",how=\"outer\")(外连接):\n", df1.merge(df3, on="a", how="outer"))
print("df1.merge(df3,on=\"a\",how=\"left\")(左连接):\n", df1.merge(df3, on="a", how="left"))
print("df1.merge(df3,on=\"a\",how=\"right\")(右连接):\n", df1.merge(df3, on="a", how="right"))
