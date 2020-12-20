import numpy as np

# nan表示不是一个数字

# 两个nan不相等
print("两个nan是否相等:", np.nan == np.nan)

t1 = np.array([[3., 3., 3., 3., 3., 3.],
               [3., 3., 3., 3., 10., 11.],
               [12., 13., 14., 15., 16., 17.],
               [18., 19., 20., np.nan, 20., 20.]])

t1[:, 0] = 0

# 计算非0的个数
print("t1中不为0的个数:", np.count_nonzero(t1))

# 计算nan的个数
print("nan的个数:", np.count_nonzero(t1 != t1))

print("isnan:\n", np.isnan(t1))

# 任何数和nan相加都为nan
print(np.sum(t1))

t2 = np.arange(12).reshape((3, 4))
print("t2:\n", t2)
print("t2的列的和:", np.sum(t2, axis=0))
print("t2的行的和:", np.sum(t2, axis=1))


# 将数组中的nan替换为均值
def fill_ndarray(t3):
    for i in range(t3.shape[1]):
        temp_col = t3[:, i]  # 当前的列
        nan_num = np.count_nonzero(temp_col != temp_col)
        if (nan_num != 0):  # 不为0，说明当前这一列中有nan
            # 当前一列不为nan的array
            temp_not_nan_col = temp_col[temp_col == temp_col]
            # 选中当前为nan的位置，把值赋值为不为nan的均值
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
    return t3


if __name__ == '__main__':
    t3 = np.arange(12).reshape((3, 4)).astype("float")
    t3[1, 2:] = np.nan
    t3 = fill_ndarray(t3)
    print("t3:", t3)
