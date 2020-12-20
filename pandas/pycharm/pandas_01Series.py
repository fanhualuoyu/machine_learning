import pandas as pd

# 一维,带标签的数组
t1 = pd.Series([1, 2, 31, 12, 3, 4])
print("t1:", t1)
# t2 = pd.Series([1,2,31,12,3,4],index = list('abcdef'))
t2 = pd.Series([1, 2, 31, 12, 3, 4], index=['a', 'b', 'c', 'd', 'e', 'f'])
print("t2(index):", t2)

temp_dict = {"name": "zh", "age": 20, "tel": 10086}
t3 = pd.Series(temp_dict)
print("t3(dict):", t3)

t4 = pd.Series(range(5))
print("t4.where(t4 > 0):\n", t4.where(t4 > 0))
print("t4.where(t4 > 1,10):\n", t4.where(t4 > 1, 10))

# index & values
print("t2.index:\n", t2.index)
print("t2.values:\n", t2.values)

# 读取csv中的文件pd.read_csv()
