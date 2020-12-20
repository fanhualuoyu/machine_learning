import numpy as np

t1 = np.arange(0, 12).reshape(2, 6)
t2 = np.arange(12, 24).reshape(2, 6)

# 行列拼接
print("t1与t2竖直拼接:", np.vstack((t1, t2)))
print("t1与t2水平拼接:", np.hstack((t1, t2)))

# 行列交换
t1[[0, 1], :] = t1[[1, 0], :]
print("t1第一行和第二行交换", t1)

t2[:, [0, 1]] = t2[:, [1, 0]]
print("t2第一列和第二列交换", t2)

# numpy中的其他方法
# 获取最大值和最小值的位置,np.argmax(t,axis = 0),np.argmin(t,axis = 1)

# 创建一个全0的数组,np.zeros((3,4))

# 创建一个全1的数组,np.ones((3,4))

# 创建一个对角线为1的正方形数组(方阵):np.eye(3)
