# 5 steps
# 1. load the data
# 2. build the model
# 3. train the model
# 4. test the model
# 5. save the model
#    load the model

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt

from util import plot_image, plot_curve, one_hot

# 1. load the data
batch_size = 512
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True,  # 使用train data
                               download=True,  # 下载数据集
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))  # mean and std, for normalization
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

x, y = next(iter(train_loader))  # 得到一个batch的数据
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'image sample')


# 2. build the model
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)  # 输入[batch_size,28*28]，输出[batch_size,256]
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # h1 = relu(xw1+b1)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w2+b2)
        x = F.relu(self.fc2(x))
        # h3 = h2w3+b3
        x = self.fc3(x)

        # 输出[batch_size,10]
        return x


# 3. train the model
net = Net()

# [w1, b1, w2, b2, w3, b3]
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)  # 使用SGD优化器,lr是学习率

train_loss = []
for epoch in range(3):  # 训练3个epoch

    for batch_idx, (x, y) in enumerate(train_loader):  # enumerate是枚举，batch_idx是batch的编号，x是图片，y是标签
        # 此循环是遍历train_loader中的每一个batch

        # x:[512, 1, 28, 28], y:[512]
        # [b, 1, 28, 28] => [b, 784]
        x = x.view(x.size(0), 28 * 28)  # 将x的形状变为[batch_size,28*28]
        # => [b, 10]
        out = net(x)
        # [b, 10]
        y_onehot = one_hot(y)
        # loss = mse(out, y_onehot)
        loss = F.mse_loss(out, y_onehot)

        # backprop
        optimizer.zero_grad()  # 梯度清零
        # grad w.r.t the loss
        loss.backward()  # 反向传播
        # update w1, w2, w3, b1, b2, b3
        optimizer.step()  # 更新参数

        train_loss.append(loss.item())

        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())  # loss.item()是将loss转换为标量

# draw the loss curve
plot_curve(train_loss)

# 4. test the model
total_correct = 0
for (x, y) in test_loader:  # 遍历test_loader中的每一个batch
    x = x.view(x.size(0), 28 * 28)
    out = net(x)
    # out = [b, 10] => pred = [b]
    pred = out.argmax(dim=1)  # 返回每一行最大值的索引,dim=0表示每一列（在列上（纵向）使用函数），dim=1表示每一行（在行上（横向）使用函数）
    # dim=1表示在第二个维度上使用函数，即在10个one-hot编码的维度上使用函数，返回每一行最大值的索引
    # axis=0在行上（横向）使用函数,axis=1在列上（纵向）使用函数
    correct = pred.eq(y).sum().float().item()
    # pred.eq(y)是比较pred和y是否相等，相等返回1，不相等返回0，sum()是求和，float()是转换为浮点数，item()是将tensor转换为标量
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)

x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28 * 28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')

"""
# 5. save the model
torch.save(net.state_dict(), 'net.pth')

# 6. load the model
net2 = Net()
net2.load_state_dict(torch.load('net.pth'))

"""
