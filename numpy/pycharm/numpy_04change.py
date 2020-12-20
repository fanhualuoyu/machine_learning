import numpy as np

t1 = np.arange(100).reshape((25, 4))

t2 = t1[2:6]

print("原本的t2:", t2)

t2[t2 < 10] = 3

print("改后的t2:", t2)

# np.where(),t2<10替换成0，否则替换成10
t3 = np.where(t2 <= 10, 0, 10)
print("where后的t2:", t3)

# clip(裁剪),小于1的替换成1，大于8的替换成8
t4 = t3.clip(1, 8)
print("改变后的t3:", t4)
