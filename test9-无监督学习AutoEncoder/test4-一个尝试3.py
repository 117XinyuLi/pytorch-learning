import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from visdom import Visdom
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np


# 一个尝试2
# 本文件用于无监督学习
# 降维效果比单纯AE好(和UMAP差不多)，效果不如AE+MLP(监督学习)，生成效果比AE好
# 降维后的数据用KMeans聚类，通过MLP分类进行效果评估

# 试试卷积，效果不错
class convResBlk(nn.Module):
    def __init__(self, in_ch, hidden1, hidden2, hidden3, out_ch):
        super(convResBlk, self).__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(in_ch, hidden1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
            nn.BatchNorm2d(hidden1),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
            nn.BatchNorm2d(hidden2),
            nn.Conv2d(hidden2, hidden3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh(),
            nn.Conv2d(hidden3, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.side = nn.Sequential()
        if in_ch != out_ch:
            self.side = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, x):
        return self.blk(x) + self.side(x)


class ResBlk(nn.Module):
    def __init__(self, in_ch, hidden1, hidden2, hidden3, out_ch):
        super(ResBlk, self).__init__()
        self.blk = nn.Sequential(
            nn.Linear(in_ch, hidden1),
            nn.Tanh(),
            nn.BatchNorm1d(hidden1),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.BatchNorm1d(hidden2),
            nn.Linear(hidden2, hidden3),
            nn.Tanh(),
            nn.Linear(hidden3, out_ch)
        )
        self.side = nn.Sequential()
        if in_ch != out_ch:
            self.side = nn.Sequential(
                nn.Linear(in_ch, out_ch)
            )

    def forward(self, x):
        return self.blk(x) + self.side(x)


class AutoEncoder1(nn.Module):
    def __init__(self):
        super(AutoEncoder1, self).__init__()
        # Encoder [batch, 1, 28, 28] => [batch, 8, 28, 28]
        self.encoder = nn.Sequential(
            convResBlk(1, 8, 8, 8, 8),
        )
        # Decoder [batch, 8, 28, 28] => [batch, 1, 28, 28]
        self.decoder = nn.Sequential(
            convResBlk(8, 8, 8, 8, 1),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_code(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x


class AutoEncoder2(nn.Module):
    def __init__(self):
        super(AutoEncoder2, self).__init__()
        # Encoder [batch, 8, 28, 28] => [batch, 100]
        self.encoder = nn.Sequential(
            convResBlk(8, 16, 16, 16, 16),
            nn.Flatten(),
            ResBlk(16 * 28 * 28, 100, 100, 100, 100),
        )
        # Decoder [batch, 100] => [batch, 8, 28, 28]
        self.decoder = nn.Sequential(
            ResBlk(100, 100, 100, 100, 16 * 28 * 28),
            nn.Unflatten(1, (16, 28, 28)),
            convResBlk(16, 16, 16, 16, 8),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_code(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x


class AutoEncoder3(nn.Module):
    def __init__(self):
        super(AutoEncoder3, self).__init__()
        # Encoder [batch, 100] => [batch, 2]
        self.encoder = nn.Sequential(
            ResBlk(100, 64, 32, 16, 2),
        )
        # Decoder [batch, 2] => [batch, 100]
        self.decoder = nn.Sequential(
            ResBlk(2, 16, 32, 64, 100),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_code(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x


class ResMLP(nn.Module):
    def __init__(self, in_ch, hidden1, hidden2, hidden3, out_ch):
        super(ResMLP, self).__init__()
        self.blk = nn.Sequential(
            nn.Linear(in_ch, hidden1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Linear(hidden2, hidden3),
            nn.LeakyReLU(),
            nn.Linear(hidden3, out_ch)
        )
        self.side = nn.Sequential()
        if in_ch != out_ch:
            self.side = nn.Sequential(
                nn.Linear(in_ch, out_ch)
            )

    def forward(self, x):
        return self.blk(x) + self.side(x)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            ResMLP(2, 16, 32, 16, 10),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


def get_data(test_loader, model1, model2, model3, device):
    model1.eval()
    model2.eval()
    model3.eval()
    code_datas = []
    labels = []
    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.to(device)
        code1 = model1.get_code(data)
        code2 = model2.get_code(code1)
        code3 = model3.get_code(code2)
        code_datas.append(code3.cpu().detach().numpy())
        labels.append(label.cpu().detach().numpy())
    code_datas = np.concatenate(code_datas, axis=0)
    labels = np.concatenate(labels, axis=0)
    return code_datas, labels


def AE_eval(code_data, label, clusters, title):
    pca = PCA(n_components=2)
    pca.fit(code_data)
    code_data = pca.transform(code_data)
    clusters = pca.transform(clusters)
    plt.figure()
    plt.scatter(code_data[:, 0], code_data[:, 1], c=label, cmap='rainbow', s=0.5)
    plt.title(title)
    plt.colorbar()
    # plt.xticks(())
    # plt.yticks(())
    plt.scatter(clusters[:, 0], clusters[:, 1], c='black', s=10)
    plt.savefig('pic3/' + title + '.png')
    plt.close()


def MLP_eval(train_code_data, train_label, code_datas, labels, device):
    epochs = 100

    train_data = torch.from_numpy(train_code_data).float()
    train_label = torch.from_numpy(train_label).long()
    test_data = torch.from_numpy(code_datas).float()
    test_label = torch.from_numpy(labels).long()

    train_dataset = TensorDataset(train_data, train_label)
    test_dataset = TensorDataset(test_data, test_label)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    mlp = MLP().to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs * len(train_loader))
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(epochs):
        mlp.train()
        loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = mlp(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        mlp.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = mlp(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            if correct / total > best_acc:
                best_acc = correct / total

        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Accuracy: {correct / total:.4f}')

    print(f'Best Accuracy: {best_acc:.4f}')  # 0.84


def train(model1, model2, model3, train_loader, optimizer, scheduler, criterion, device, epoch, beta1, beta2, beta3,
          beta4, beta5, beta6):
    model1.train()
    model2.train()
    model3.train()
    loss = 0
    loss1, loss2, loss3 = 0, 0, 0
    loss4, loss5, loss6 = 0, 0, 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        code1 = model1.get_code(data)
        code2 = model2.get_code(code1)
        code3 = model3.get_code(code2)
        recon3 = model3.decode(code3)
        recon2 = model2.decode(recon3)
        recon1 = model1.decode(recon2)
        output1 = model1(data)
        output2 = model2(code1)
        output3 = model3(code2)

        loss1 = criterion(recon1, data)
        loss2 = criterion(recon2, code1)
        loss3 = criterion(recon3, code2)
        loss4 = criterion(output1, data)
        loss5 = criterion(output2, code1)
        loss6 = criterion(output3, code2)

        # loss是单个AE输入输出的MSE，多个AE组合输入输出的MSE的加权和
        loss = beta1 * loss1 + beta2 * loss2 + beta3 * loss3 + beta4 * loss4 + \
               beta5 * loss5 + beta6 * loss6

        optimizer.zero_grad()

        loss.backward()

        # 梯度裁剪
        for p in model1.parameters():
            torch.nn.utils.clip_grad_norm_(p, max_norm=1)
        for p in model2.parameters():
            torch.nn.utils.clip_grad_norm_(p, max_norm=1)
        for p in model3.parameters():
            torch.nn.utils.clip_grad_norm_(p, max_norm=1)

        optimizer.step()

        scheduler.step()

    print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Loss1: {loss1.item():.4f}, '
          f'Loss2: {loss2.item():.6f}, Loss3: {loss3.item():.8f}, '
          f'Loss4: {loss4.item():.4f}, Loss5: {loss5.item():.6f}, '
          f'Loss6: {loss6.item():.8f}')


def visual_test(model1, model2, model3, viz, image):
    code1 = model1.get_code(image)
    code2 = model2.get_code(code1)
    code3 = model3.get_code(code2)
    output1 = model3.decode(code3)
    output2 = model2.decode(output1)
    output3 = model1.decode(output2)
    # 输出图片(图片会随着epoch的变化而变化)
    viz.images(image, nrow=8, win='input', opts=dict(title='input'))
    viz.images(output3, nrow=8, win='output', opts=dict(title='output'))


def test(model1, model2, model3, test_loader, criterion, device):
    total_loss = 0
    for batch_idx, (data, _) in enumerate(test_loader):
        data = data.to(device)
        code1 = model1.get_code(data)
        code2 = model2.get_code(code1)
        code3 = model3.get_code(code2)
        output1 = model3.decode(code3)
        output2 = model2.decode(output1)
        output3 = model1.decode(output2)
        loss = criterion(output3, data)
        total_loss += loss.item()
    total_loss /= len(test_loader)

    print(f'Test Loss: {total_loss:.4f}')

    return total_loss


def main():
    mnist_train = datasets.MNIST(root='data', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     # 加入噪声
                                     transforms.Lambda(lambda x: x + 0.01 * torch.rand_like(x))
                                 ]))
    mnist_test = datasets.MNIST(root='data', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor()
                                ]))
    train_loader = DataLoader(mnist_train, batch_size=512, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=32, shuffle=True)

    epochs = 50

    beta1 = 10
    beta2 = 5
    beta3 = 1
    beta4 = 10
    beta5 = 5
    beta6 = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model1 = AutoEncoder1().to(device)
    model2 = AutoEncoder2().to(device)
    model3 = AutoEncoder3().to(device)
    # model1.load_state_dict(torch.load('models/AE1-un.pth'))
    # model2.load_state_dict(torch.load('models/AE2-un.pth'))
    # model3.load_state_dict(torch.load('models/AE3-un.pth'))

    criterion = nn.MSELoss().to(device)

    lr1 = 1e-3
    lr2 = 1e-3
    lr3 = 1e-3
    weight_decay = 1e-5

    optimizer = torch.optim.Adam([
        {'params': model1.parameters(), 'lr': lr1},
        {'params': model2.parameters(), 'lr': lr2},
        {'params': model3.parameters(), 'lr': lr3},
    ], weight_decay=weight_decay)  # weight_decay不宜过大，lr不宜过大
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs*len(train_loader))

    viz = Visdom()
    min_loss = 100000
    for epoch in range(epochs):

        train(model1, model2, model3, train_loader, optimizer, scheduler, criterion, device, epoch, beta1, beta2, beta3, beta4, beta5, beta6)

        x = next(iter(test_loader))[0].to(device)  # [0] is data
        with torch.no_grad():
            visual_test(model1, model2, model3, viz, x)

            train_code_data, train_label = get_data(train_loader, model1, model2, model3, device)
            kmeans = KMeans(n_clusters=10, random_state=0).fit(train_code_data)
            clusters = kmeans.cluster_centers_
            name = 'train epoch' + str(epoch)
            AE_eval(train_code_data, train_label, clusters, name)

            code_datas, labels = get_data(test_loader, model1, model2, model3, device)
            name = 'AE3_PCA epoch' + str(epoch)
            AE_eval(code_datas, labels, clusters, name)

            loss = test(model1, model2, model3, test_loader, criterion, device)

            if min_loss > loss:
                min_loss = loss
                torch.save(model1.state_dict(), 'models/AE1-un.pth')
                torch.save(model2.state_dict(), 'models/AE2-un.pth')
                torch.save(model3.state_dict(), 'models/AE3-un.pth')

    model1.load_state_dict(torch.load('models/AE1-un.pth'))
    model2.load_state_dict(torch.load('models/AE2-un.pth'))
    model3.load_state_dict(torch.load('models/AE3-un.pth'))
    print(f'Best Test Loss: {min_loss:.4f}')

    train_code_data, train_label = get_data(train_loader, model1, model2, model3, device)

    # 用Kmeans聚类
    kmeans = KMeans(n_clusters=10, random_state=0).fit(train_code_data)
    clusters = kmeans.cluster_centers_

    name = 'AE3_PCA_train'
    AE_eval(train_code_data, train_label, clusters, name)

    code_datas, labels = get_data(test_loader, model1, model2, model3, device)
    name = 'AE3_PCA_test'
    AE_eval(code_datas, labels, clusters, name)

    del model1, model2, model3
    MLP_eval(train_code_data, train_label, code_datas, labels, device)


if __name__ == "__main__":
    main()
