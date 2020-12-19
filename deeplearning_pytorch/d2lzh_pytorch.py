import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

def data_iter(batch_size, features, labels): # 小批量数据样本
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i+batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)

def linreg(X, w, b): # 线性回归矢量计算表达式
    return torch.mm(X, w) + b

def squared_loss(y_hat, y): # 平方损失来定义损失函数
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size): # 优化算法：小批量随机梯度下降法
    for param in params:
        param.data -= lr * param.grad / batch_size
