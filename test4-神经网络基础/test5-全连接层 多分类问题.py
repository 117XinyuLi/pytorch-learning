# 全连接层 多分类问题实例

import torch
import torchvision
from torch.nn import functional as F
from torch import nn

# 1.nn.Linear 与 relu
x = torch.randn(1, 784)
layer1 = nn.Linear(784, 200)  # 784->200 第一个参数是输入的维度，第二个参数是输出的维度
layer2 = nn.Linear(200, 200)
layer3 = nn.Linear(200, 10)
x = layer1(x)
x = F.relu(x, inplace=True)  # inplace=True 会改变x的值，不会创建新的tensor，节省内存
x = layer2(x)
x = F.relu(x, inplace=True)
x = layer3(x)
x = F.relu(x, inplace=True)

x = torch.randn(1, 28, 28, 3)
layer = nn.Linear(3, 10)
x = layer(x)
print(x.shape)# torch.Size([1, 28, 28, 10]),Linear层会把输入的最后一个维度作为输入的维度，输出的维度是第二个参数

# 区分class-style和function-style API
# class-style API
# 一般是大写开头，如nn.Linear，nn.ReLU 要先实例化，然后调用实例化的对象
# function-style API
# 一般是小写开头，如F.relu，F.sigmoid，F.softmax，F.cross_entropy 调用的时候直接调用函数

# 2.实现Module
#  1.继承nn.Module
#  2.定义__init__函数
#  3.定义forward函数
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(  # 顺序执行,可以使用所有nn.Module中的函数,用class-style API
            nn.Linear(784, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 3.进行训练
#  1.数据加载
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

#  2.定义模型
model = MLP() # 实例化模型,不用考虑初始化参数，因为nn.Module会自动初始化参数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 优化器 model.parameters()是模型的参数
criterion = nn.CrossEntropyLoss()  # 损失函数

#  3.训练
epochs = 10
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} | Batch index: {batch_idx} | Loss: {loss.item()}')

#  4.测试
test_correct = 0
for data, target in test_loader:
    data = data.view(-1, 28 * 28)
    output = model(data)
    pred = output.argmax(dim=1)
    test_correct += pred.eq(target).sum().float().item()

total_num = len(test_loader.dataset)
print(f'Accuracy on test set: {test_correct / total_num:.6f}')

