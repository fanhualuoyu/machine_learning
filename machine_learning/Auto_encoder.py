import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# 定义一些超参数
epoch = 10
batch_size = 64
lr = 0.005
download_mnist = True  # 下过的数据的话，可以设置成False
n_test_img = 5  # 显示5张图片看效果

# 模型由两部分组成，编码器(encoder)和解码器(decoder)，它们是两个相反的流程。encoder将28×28的数据压缩成3个数据，decoder将3个数据扩展为28×28个数据。
# MNIST 手写数字数据集
train_data = torchvision.datasets.MNIST(
    root=r"D:\web\machine\machine_learning\Datasets\MNIST",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=download_mnist,
)

train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


# 定义模型
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3)  # 压缩成3个特征
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encodered = self.encoder(x)
        decodered = self.decoder(encodered)
        return encodered, decodered


autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
loss_func = nn.MSELoss()

# 初始化图像1
f, a = plt.subplots(2, n_test_img, figsize=(5, 2))
plt.ion()
view_data = train_data.train_data[:n_test_img].view(-1, 28 * 28).type(torch.FloatTensor) / 255
for i in range(n_test_img):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

# 模型训练
for e in range(epoch):
    for step, (x, b_label) in enumerate(train_loader):
        b_x = x.view(-1, 28 * 28)
        b_y = x.view(-1, 28 * 28)
        encoded, decoded = autoencoder(b_x)
        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('Epoch:', e, '| train loss: %.4f' % loss.data.numpy())
            _, decoded_data = autoencoder(view_data)
            for i in range(n_test_img):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.draw()
            plt.pause(0.05)
plt.ioff()
plt.show()

# 3D 可视化的展示
view_data = train_data.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.
encoded_data, _ = autoencoder(view_data)   # 提取压缩的特征值
fig = plt.figure(2); ax = Axes3D(fig)      # 3D 图
X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train_data.train_labels[:200].numpy()  # 标签值

for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()