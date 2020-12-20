import numpy as np

# 二维数组
t1 = np.arange(12)
print("t1:", t1)
t2 = t1.reshape((3, 4))
print("t1.reshape:", t2)

# 三维数组
t3 = np.arange(24).reshape((2, 3, 4))
print("t3:", t3)

t4 = t3.reshape((4, 6))
print("t3.reshape:", t4)

print("t4的行:", t4.shape[0])
print("t4的列:", t4.shape[1])

t5 = t4.flatten()
print("t4.flatten():", t5)

# 计算操作
print("t4 + 2:", t4 + 2)

t6 = np.arange(100, 124).reshape((4, 6))
print("t4 + t6:", t4 + t6)

t7 = np.arange(0, 6)
print("t4 - t7:", t4 - t7)

t8 = np.arange(4).reshape(4, 1)
print("t4 - t8:", t4 - t8)

# 文件的读取
'''
    np.loadtxt(frame.dtype = np.float.delimiter = None,skiprows = 0,usecols = None,unpack = False)
        frame:文件、字符串或产生器，可以是.gz或bz2压缩文件
        dtype:数据类型，可选，CSV的字符串以什么数据类型读入数组中，默认np.float
        delimiter:分隔字符串，默认是任何空格，改为逗号
        skiprows:跳过前x行，一般跳过第一行表头
        usecols:读取指定的列，索引，元组类型
        unpack:如果True，读入属性将分别写入不同数组变量，False读入数据只写入一个数组变量，默认False
'''
