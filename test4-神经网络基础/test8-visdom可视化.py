# visdom可视化

# 先安装visdom
# pip install visdom
# 然后运行visdom
# 在命令行anaconda prompt中conda activate pytorch，然后
# Python -m visdom.server
# 然后在浏览器中输入http://localhost:8097/，就可以看到visdom的界面了
# 关闭visdom的命令是在命令行中ctrl+c

import torch
import torchvision
import torch.nn as nn
from visdom import Visdom

loss = torch.randn(1)
global_step = 10
acc = 0.95

# 画一条曲线
viz = Visdom()
viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))# 创建一个窗口
#               第一个参数是y,第二个是x,win是窗口的名字(唯一的ID信息)，opts是窗口的参数,env是环境的名字(默认是main)
viz.line([loss.item()], [global_step], win='train_loss', update='append')# 更新窗口
#            第一个参数是y,第二个是x,win是窗口的名字(唯一的ID信息)，update是更新方式，append是追加

# 画多条曲线
viz = Visdom()
viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.', legend=['loss', 'acc.']))# 创建一个窗口
#                       y传两个数值 legend是图标(y1,y2的label)
viz.line([[loss.item(), acc]], [global_step], win='test', update='append')# 更新窗口

# visual X
pred = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
viz = Visdom()
# 可视化图片只调用viz.images()一次，否则会对之前的输出覆盖，图片格式：[batch, channel, height, width]
viz.images(torch.randn(3, 1, 28, 28), win='x')# 可视化图片
#           第一个参数是需要可视化的图片tensor,win是窗口的名字(唯一的ID信息)
# 可视化文字也是调用viz.text()一次，否则会对之前的输出覆盖
viz.text(str(pred.detach().cpu().numpy()), win='text', opts=dict(title='text'))# 可视化文本
#          第一个参数是需要可视化的文本,win是窗口的名字(唯一的ID信息),opts是窗口的参数

# visdom实例
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
    return test_correct / total_num


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
def train(model, optimizer, criterion, train_loader, test_loader, device, epochs=10, test_interval=50):
    viz = Visdom()
    steps = 1
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, 28 * 28)
            data, target = data.to(device), target.to(device)  # 将数据转移到GPU上
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 可视化
            viz.line([loss.item()], [steps], win='train loss', update='append', opts=dict(title='train loss'))  # 更新窗口
            steps += 1

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch} | Batch index: {batch_idx} | Loss: {loss.item()}')

        # 每个epoch测试一次
        test_acc = test(model, test_loader, device)
        # 可视化
        viz.line([test_acc], [epoch], win='test acc', update='append', opts=dict(title='test acc'))  # 更新窗口


def main():
    # 1.加载数据
    train_loader, test_loader = data_loader()
    # 2.初始化模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, optimizer, criterion = init_model(device)
    # 3.训练模型
    train(model, optimizer, criterion, train_loader, test_loader, device, epochs=10, test_interval=50)
    # 4.测试模型
    test(model, test_loader, device)


if __name__ == '__main__':
    main()





