# 测试
# Loss！= Accuracy
# loss小，accuracy大不一定代表模型好
# 有可能是模型过拟合了
# 需要validation set 来验证模型

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# 进行测试
logits = torch.rand(4, 10)
pred = F.softmax(logits, dim=1)
pred_label = pred.argmax(dim=1)
print(pred_label)
label = torch.tensor([3, 5, 7, 9])
correct = pred_label.eq(label).sum().float().item()
#                           eq()是判断相等，返回bool类型, sum()是求和，float()是转换为浮点数, item()是将tensor转换为python的数据类型
acc = correct / len(label)
print(acc)


# 如果只想要预测的标签，不要概率，也可以不用softmax，直接使用argmax

# when to test(使用validation set)
# 1. test once per epoch
# 2. test once per N epochs
# 3. test once per N batches
# 以上均可以

# 测试实例
# 模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(  # 顺序执行,可以使用所有nn.Module中的函数,用class-style API
            nn.Linear(784, 200),
            nn.LeakyReLU(0.01, inplace=True),  # 使用leaky ReLU激活函数
            nn.Linear(200, 200),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 测试函数(训练中测试可用loss和acc来评估,这里用acc)
def test(model, test_loader, device):
    test_correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.to(device)  # 将数据转移到GPU上
        output = model(data)
        pred = output.argmax(dim=1)
        test_correct += pred.eq(target).sum().float().item()

    total_num = len(test_loader.dataset)
    print(f'Accuracy on test set: {test_correct / total_num:.6f}')


# 加载数据
def data_loader():
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
    return train_loader, test_loader


# 设置模型
def init_model(device):
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss().to(device)
    return model, optimizer, criterion


# 训练模型
def train(model, optimizer, criterion, train_loader, test_loader, device, epochs=10):
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

        #  训练中测试
        test(model, test_loader, device)


def main():
    # 1.加载数据
    train_loader, test_loader = data_loader()
    # 2.初始化模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, optimizer, criterion = init_model(device)
    # 3.训练模型
    train(model, optimizer, criterion, train_loader, test_loader, device, epochs=10)
    # 4.测试模型
    test(model, test_loader, device)


if __name__ == '__main__':
    main()
