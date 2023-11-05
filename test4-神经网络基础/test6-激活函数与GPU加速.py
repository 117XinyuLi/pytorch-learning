# 激活函数与GPU加速

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# 1.激活函数
# sigmoid和tanh激活函数在深度学习中已经不常用了(tanh在RNN中还有使用)，因为它们的梯度容易消失，导致训练不稳定
# ReLU激活函数在深度学习中使用最多，因为它的梯度不容易消失，训练稳定

# leaky ReLU激活函数是ReLU的改进版，当x<0时，它的梯度不是0，而是一个很小的值，这样可以避免ReLU梯度消失的问题
# leaky ReLU激活函数的公式为：f(x)=max(αx,x) α是一个很小的值，如0.01
# 使用leaky ReLU激活函数
x = torch.randn(1, 784)
activation = nn.LeakyReLU(0.01, inplace=True)  # 0.01是α,一个很小的值(α默认为0.01)
y1 = activation(x)
print(y1)
y2 = F.leaky_relu(x, 0.01, inplace=True)  # 0.01是α,一个很小的值(α默认为0.01)
print(y2)  # y1和y2是一样的

# SELU激活函数是两个函数的综合，在x<0时也有光滑的曲线
# SELU激活函数的公式为：f(x)=λ⋅max(0,x)+λ⋅min(0,α⋅(exp(x)−1)) λ和α是两个常数，λ=1.0507，α=1.6733
# 使用SELU激活函数
x = torch.randn(1, 784)
activation = nn.SELU(inplace=True)  # inplace=True表示是否进行原地操作
y1 = activation(x)
print(y1)
y2 = F.selu(x, inplace=True)  # inplace=True表示是否进行原地操作
print(y2)  # y1和y2是一样的

# softplus是一个在x=0处平滑的ReLU函数，它的公式为：f(x)=(1/β)*log(1+exp(β*x))
# 使用softplus激活函数
x = torch.randn(1, 784)
activation = nn.Softplus(beta=1, threshold=20)  # beta=1表示β=1，threshold=20表示x>20时，f(x)=x
y1 = activation(x)
print(y1)
y2 = F.softplus(x, beta=1, threshold=20)  # beta=1表示β=1，threshold=20表示x>20时，f(x)=x
print(y2)  # y1和y2是一样的

# ConvNeXt是一种新的卷积神经网络，它的激活函数是GELU

# 2.使用GPU加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# net = MLP().to(device) # 将网络转移到GPU上
# criterion = nn.CrossEntropyLoss().to(device) # 将损失函数转移到GPU上
# data, target = data.to(device), target.to(device) # 将数据转移到GPU上
# 注意：data1 = data.to(device)中data和data1是不同的，data1是data的副本，data1和data在内存中的位置不同，data1在GPU上，data在CPU上
#      对data1和data同一次forward再backward，会产生两个tensor，一个在GPU上，一个在CPU上
#      把网络搬到GPU上后，和原来的网络是一样的

# 使用GPU加速实例
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(  # 顺序执行,可以使用所有nn.Module中的函数,用class-style API
            nn.Linear(784, 200),
            nn.LeakyReLU(0.01, inplace=True),# 使用leaky ReLU激活函数
            nn.Linear(200, 200),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x

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
model = MLP().to(device)  # 实例化模型,不用考虑初始化参数，因为nn.Module会自动初始化参数,并且将模型转移到GPU上
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 优化器 model.parameters()是模型的参数
criterion = nn.CrossEntropyLoss().to(device)  # 损失函数,并且将损失函数转移到GPU上

#  3.训练
epochs = 10
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.to(device)  # 将数据转移到GPU上
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
    data, target = data.to(device), target.to(device)  # 将数据转移到GPU上
    output = model(data)
    pred = output.argmax(dim=1)
    test_correct += pred.eq(target).sum().float().item()

total_num = len(test_loader.dataset)
print(f'Accuracy on test set: {test_correct / total_num:.6f}')
