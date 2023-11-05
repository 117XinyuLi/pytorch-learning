# 逻辑回归 多元分类问题

# Q1 why not maximize accuracy?
# acc.=sum(pred==y)/len(y)
# issue1: gradient==0 if accuracy unchanged but weights changed
# issue2: gradient not continuous since the number of correct is not continuous
# so we use loss function to maximize loss

# Q2 why call logistic regression?
# use sigmoid
# 刚出现的时候使用MSE优化，而非交叉熵，很像回归问题，所以叫逻辑回归
# 但是，现在这个问题是分类问题

# Q3 why use cross entropy (not MSE)?
# cross entropy更快（计算的梯度更大）
# 在分类中cross entropy为凸函数，更容易优化，而MSE不是
# 但是有时候(meta-learning)用MSE更好

# 多元分类问题实例
import torch
import torchvision
from torch.nn import functional as F

# 1. 参数初始化
w1, b1 = torch.randn(200, 784, requires_grad=True), torch.zeros(200, requires_grad=True)
# w的形状为（ch_out,ch_in）
w2, b2 = torch.randn(200, 200, requires_grad=True), torch.zeros(200, requires_grad=True)
w3, b3 = torch.randn(10, 200, requires_grad=True), torch.zeros(10, requires_grad=True)

# 使用kaiming初始化(一定注意)
torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)

# 2. 数据加载
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


# 3. 训练
def forward(x):
    x = x @ w1.t() + b1
    x = F.relu(x)
    x = x @ w2.t() + b2
    x = F.relu(x)
    x = x @ w3.t() + b3
    x = F.relu(x)
    return x


optimizer = torch.optim.SGD([w1, b1, w2, b2, w3, b3], lr=1e-3)  # 优化器 优化w1,b1,w2,b2,w3,b3,学习率为1e-3
crition = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数,自带softmax，target不是one-hot

epochs = 100
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        logits = forward(data)
        loss = crition(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'epoch:{epoch},batch_idx:{batch_idx},loss:{loss.item()}')

# 4. 测试
test_correct = 0
for data, target in test_loader:
    data = data.view(-1, 28 * 28)
    logits = forward(data)
    pred = logits.argmax(dim=1)
    test_correct += pred.eq(target).sum().float().item()

total_num = len(test_loader.dataset)
print(f'Accuracy on test set: {test_correct / total_num:.6f}')
