# MNIST实战
# 使用train-validation-test的方法
# CV:5-fold可在test9中查看

import torch
import torch.nn as nn
import torchvision
from visdom import Visdom


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(  # 顺序执行,可以使用所有nn.Module中的函数,用class-style API
            nn.Linear(784, 200),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.01, inplace=True),  # 使用leaky ReLU激活函数
            nn.Linear(200, 200),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 加载数据
def data_loader(batch_size=512):
    # 加载数据
    train_db = torchvision.datasets.MNIST('mnist_data', train=True,  # 使用train data
                                          download=True,  # 下载数据集
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
                                              torchvision.transforms.Normalize(
                                                  (0.1307,), (0.3081,))  # mean and std, for normalization
                                          ]))
    test_db = torchvision.datasets.MNIST('mnist_data/', train=False,
                                         download=True,  # 下载数据集
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.1307,), (0.3081,))
                                         ]))

    # 划分数据集
    train_set, dev_set = torch.utils.data.random_split(train_db, [50000, 10000])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_db, batch_size=batch_size, shuffle=True)
    return train_loader, dev_loader, test_loader


# 设置模型
def init_model(device, learning_rate=1e-2, beta=(0.9, 0.99), weight_decay=1e-4):
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=beta)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    return model, optimizer, criterion, scheduler


# 测试函数
def test(model, test_loader, device):
    model.eval()
    test_correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        test_correct += pred.eq(target).sum().float().item()
    total_num = len(test_loader.dataset)
    print(f'{test_correct / total_num:.6f}')


# 在训练时的测试函数
def test_while_training(model, val_loader, criterion, device):
    model.eval()
    loss = 0
    correct = 0
    for data, target in val_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss += criterion(output, target).item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().float().item()
    total_num = len(val_loader.dataset)

    return loss / len(val_loader), correct / total_num


# 训练模型
def train(model, optimizer, criterion, scheduler, train_loader, dev_loader, device, epochs=10, test_interval=25):
    viz = Visdom()
    for epoch in range(epochs):
        # 训练
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, 28 * 28)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [epoch * len(train_loader) + batch_idx], win='train_loss', update='append', opts=dict(title='train_loss'))

            if batch_idx % test_interval == 0 and batch_idx > 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')

        # 测试
        val_loss, val_acc = test_while_training(model, dev_loader, criterion, device)
        print(f'Epoch: {epoch}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}')
        scheduler.step()
        viz.line([val_loss], [epoch], win='val_loss', update='append', opts=dict(title='val_loss'))


def main():
    # 1.加载数据
    train_loader, dev_loader, test_loader = data_loader()
    # 2.初始化模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, optimizer, criterion, scheduler = init_model(device)
    # 3.训练模型
    train(model, optimizer, criterion, scheduler, train_loader, dev_loader, device)
    # 4.测试模型
    print('Train Acc:')
    test(model, train_loader, device)
    print('Dev Acc:')
    test(model, dev_loader, device)
    print('Test Acc:')
    test(model, test_loader, device) # 0.97
    # 5.保存模型
    torch.save(model.state_dict(), 'model.pth') # 保存模型参数
    # 6.加载模型
    # model.load_state_dict(torch.load('model.pth')) # 加载模型参数


if __name__ == '__main__':
    main()
