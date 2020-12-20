import numpy as np
import random as r

# 创建数组
t1 = np.array([1, 2, 3])
print("t1:", t1)

t2 = np.array(range(10))
print("t2:", t2)

t3 = np.arange(4, 10, 2)
print("t3:", t3)

print("t3的.dtype:", t3.dtype)

t4 = np.array(range(1, 4), dtype=float)
print("t4:", t4)
print("t4的.dtype:", t4.dtype)

# numpy中的bool类型
t5 = np.array([1, 1, 0, 1, 0, 0], dtype=bool)
print("t5:", t5)
print("t5的.dtype", t5.dtype)

# 调整数据类型
t6 = t5.astype("int8")
print("t6更改数据类型:", t6)

# numpy中的小数
t7 = np.array([r.random() for i in range(10)])
print("t7:", t7)
print("t7的.dtype:", t7.dtype)

t8 = np.round(t7, 2)
print("t8:", t8)
