# train-dev(validation)-test split

# 若只有两个set，即train和test(或者train和dev)
# 可以一边训练一边测试，记录每个test error的点，用此时的模型

# 若有三个set，即train-dev(validation)-test
# 可以先训练train，然后用dev来调参(找使dev error最小的点)，最后客户用test来测试(真实开发中test set不会给我们)，test error是交给客户验收的结果
# 用test error作为模型的最终结果，若使用dev error作为模型的最终结果，有的人会直接用dev set来训练，这样就会导致过拟合

import torch
import torch.nn as nn
import torchvision
from visdom import Visdom

# split data
# 获取数据
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
print(len(train_db), len(test_db))  # 60000 10000
# 2.划分数据（train-dev）
train_set, val_set = torch.utils.data.random_split(train_db, [50000, 10000])
#                                 从train_db中随机抽取50000个作为train_set, 10000个作为val_set
print(len(train_set), len(val_set))  # 50000 10000
# 3.构建数据加载器
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_db, batch_size=128, shuffle=True)
print(len(train_loader), len(val_loader), len(test_loader))  # 782 79 79 batch的个数

# 使用train-dev(validation)-test进行训练

def load_data(batch_size):
    train_db = torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                  torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))
                              ]))
    test_db = torchvision.datasets.MNIST('mnist_data/', train=False, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]))
    test_loader = torch.utils.data.DataLoader(test_db,batch_size=batch_size, shuffle=True)

    train_db, val_db = torch.utils.data.random_split(train_db, [50000, 10000])
    train_loader = torch.utils.data.DataLoader(train_db,batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_db,batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, val_loader


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x


def training(train_loader, net, device, optimizer, criteon):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def validating(val_loader, net, device, criteon):
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()
    test_loss /= len(val_loader.dataset)
    print('\nVAL set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


def testing(test_loader, net, device, criteon):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


global net

if __name__ == '__main__':
    batch_size = 200
    learning_rate = 0.01
    epochs = 10

    train_loader, test_loader, val_loader = load_data(batch_size)
    print('train:', len(train_loader.dataset), 'test:', len(test_loader.dataset), 'val:', len(val_loader.dataset))

    device = torch.device('cuda:0')
    net = MLP().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    criteon = nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        training(train_loader, net, device, optimizer, criteon)
        validating(val_loader, net, device, criteon)

    testing(test_loader, net, device, criteon)


# K—fold cross validation
# train_db 60000个数据化为6份，每份10000个数据，每次取其中一份作为val_set，剩下的5份作为train_set，这样训练一次为一个epoch
# 训练6次，每次都有不同的val_set，val_loss是6次的平均值
# 可以使用torch.utils.data.Subset(set, indices)来划分数据集

# 实例
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


# 加载数据
def data_loader(k=5, batch_size=512):
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

    # 划分数据,分为k个fold
    num_val = len(train_db) // k
    cv_loader_list = []
    for i in range(k):
        val_set = torch.utils.data.Subset(train_db, range(i * num_val, (i + 1) * num_val))
        train_set = torch.utils.data.Subset(train_db,
                                            list(range(0, i * num_val)) + list(range((i + 1) * num_val, len(train_db))))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
        cv_loader_list.append((train_loader, val_loader))
    test_loader = torch.utils.data.DataLoader(test_db, batch_size=batch_size, shuffle=True)
    return cv_loader_list, test_loader


# 设置模型
def init_model(device, learning_rate=1e-3):
    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)
    return model, optimizer, criterion


# 测试函数
def test(model, test_loader, device):
    test_correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        test_correct += pred.eq(target).sum().float().item()
    total_num = len(test_loader.dataset)
    print(f'Accuracy on test set: {test_correct / total_num:.6f}')


# 在训练时的测试函数
def test_while_training(model, val_loader, criterion, device):
    loss = 0
    for data, target in val_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss += criterion(output, target).item()

    return loss / len(val_loader)


# 训练模型
def train(model, optimizer, criterion, cv_loader_list, device, epochs=10, test_interval=25):
    if epochs < len(cv_loader_list):
        print('epochs should be larger than k')
        return

    viz = Visdom()
    cv_index = 0
    cv_loss = 0
    k = len(cv_loader_list)
    # 一共训练epochs次，一个epoch对k-1个fold进行训练，在另一个fold上进行测试
    for epoch in range(epochs):
        train_loader, val_loader = cv_loader_list[cv_index]
        cv_index = (cv_index + 1) % k # 选择下一个需要训练和测试的划分

        # 训练
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, 28 * 28)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [epoch * len(train_loader) + batch_idx], win='train_loss', update='append')

            if batch_idx % test_interval == 0 and batch_idx > 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')

        # 记录当前val set的loss
        cv_loss += test_while_training(model, val_loader, criterion, device)

        # k个epoch后，计算k个划分的平均loss(类似于k个epoch后，进行一次训练中测试)
        if (epoch + 1) % k == 0:
            # 所有划分都训练过一次，计算平均loss
            print(f'Cross validation loss: {cv_loss / k:.6f}')
            viz.line([cv_loss / k], [epoch], win='cv_loss', update='append')
            cv_loss = 0

    if cv_loss != 0:
        print(f'Cross validation loss: {cv_loss / k:.6f}')


def main():
    # 1.加载数据
    cv_loader_list, test_loader = data_loader()
    # 2.初始化模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, optimizer, criterion = init_model(device)
    # 3.训练模型
    train(model, optimizer, criterion, cv_loader_list, device)
    # 4.测试模型
    test(model, test_loader, device)


if __name__ == '__main__':
    main()
