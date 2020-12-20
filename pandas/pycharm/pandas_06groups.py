import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

file_path = "pandas/starbucks_store_worldwide.csv"
df = pd.read_csv(file_path)

grouped = df.groupby(by="Country")

# 遍历
# for i in grouped:
#     print(i)
#     print("*" * 100)

# 分组聚合
# 计算每个国家的数量*
print("grouped.count:\n", grouped["Brand"].count())

# 计算中国每个省份的数量
china_data = df[df["Country"] == "CN"]
province = china_data.groupby(by="State/Province").count()["Brand"]
print(province)

# 按照多个条件进行分组,返回Series
group = df["Brand"].groupby(by=[df["Country"], df["State/Province"]]).count()
print(group)

# 按照多个条件进行分组，返回DataFrame
group1 = df[["Brand"]].groupby(by=[df["Country"], df["State/Province"]]).count()
group2 = df.groupby(by=[df["Country"], df["State/Province"]])[["Brand"]].count()
group3 = df.groupby(by=[df["Country"], df["State/Province"]]).count()[["Brand"]]
print("*" * 100)
print(group1)
print("*" * 100)
print(group2)
print("*" * 100)
print(group3)
print("*" * 100)

# 索引的方法和属性
df1 = pd.DataFrame(np.ones((2, 4)), index=["A", "B"], columns=list("abcd"))
df1.loc["A", "a"] = 100.0
# 指定index
df1.index = ["a", "b"]
print("df1.index = [\"a\",\"b\"]:\n", df1)
# reindex
print("df1.reindex([\"a\",\"f\"]):\n", df1.reindex(["a", "f"]))
# set_index，指定某一列作为index
print("df1.setindex(\"a\"):", df1.set_index("a"))
print("*" * 10)
print("df1.setindex(\"a\"):", df1.set_index("a", drop=False))
print("*" * 10)

a = pd.DataFrame(
    {'a': range(7), 'b': range(7, 0, -1), 'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'], 'd': list("hjklmno")})
b = a.set_index(["c", "d"])
c = b["a"]
print("c:", c)
print("c[\"one\"][\"j\"]:", c["one"]["j"])
print("*" * 10)
d = a.set_index(["d", "c"])["a"]
print("d.swaplevel[\"one\"]:", d.swaplevel()["one"])
print("*" * 10)
print("b.loc[\"one\"].loc[\"j\"]", b.loc["one"].loc["j"])
