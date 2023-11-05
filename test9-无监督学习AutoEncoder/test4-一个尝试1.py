import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import optuna
import numpy as np


# 一个尝试：多个自编码器的结合降维，将加密后的数据通过MLP进行分类，边降维边分类（监督降维?）
# 最后通过PCA进行线性变换，可视化（不用PCA也可以，可视化效果也不错）

# 本文件用于调参


# 激活函数用Tanh，不要用ReLU，否则导致输出的点都在一条直线上
# Dropout不宜使用，BN可以使用
# 使用ResNet使训练更加稳定，让所有loss均匀下降，防止网络过深导致梯度消失，使网络参数更新不统一，导致深度网络训练不稳定，loss/acc剧烈波动
# 这里没有尝试卷积，可以试试看，效果应该会更好
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
        # Encoder [batch, 784] => [batch, 100]
        self.encoder = ResBlk(784, 512, 256, 128, 100)
        # Decoder [batch, 100] => [batch, 784]
        self.decoder = nn.Sequential(
            ResBlk(100, 128, 256, 512, 784),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 1, 28, 28)
        return x

    def get_code(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        x = x.view(x.size(0), 1, 28, 28)
        return x


class AutoEncoder2(nn.Module):
    def __init__(self):
        super(AutoEncoder2, self).__init__()
        # Encoder [batch, 100] => [batch, 20]
        self.encoder = ResBlk(100, 64, 32, 24, 20)
        # Decoder [batch, 20] => [batch, 100]
        self.decoder = ResBlk(20, 24, 32, 64, 100)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_code(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x


class AutoEncoder3(nn.Module):
    def __init__(self):
        super(AutoEncoder3, self).__init__()
        # Encoder [batch, 20] => [batch, 2]
        self.encoder = ResBlk(20, 16, 8, 4, 2)
        # Decoder [batch, 2] => [batch, 20]
        self.decoder = ResBlk(2, 4, 8, 16, 20)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_code(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x


class ResMLP(nn.Module):
    def __init__(self,  in_ch, hidden1, hidden2, hidden3, out_ch):
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
            ResMLP(2, 16, 32, 64, 128),
            nn.Dropout(0.25),
            ResMLP(128, 64, 32, 16, 10)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


def cluster_activation(trial, dist, device):
    rc = trial.suggest_float('rc', 5, 10)
    k = trial.suggest_int('k', 1, 10)
    idx1 = (dist < rc).to(device)
    idx2 = (dist > rc).to(device)
    return F.relu(rc - dist) * idx1 + F.relu(dist - rc) * idx2 * k


def distance_activation(trial, dist, device):
    rd = trial.suggest_float('rd', 0.1, 5)
    idx = (dist > rd).to(device)
    return F.relu(dist - rd) * idx


def distance_loss(trial, code, label, device):
    distance = torch.tensor(0.0).to(device)
    cluster = torch.zeros(10, 2).to(device)
    flag = False
    for i in range(10):
        label_index = (label == i).to(device)
        valid_code = code[label_index]# [x, 2]
        cluster[i] = torch.mean(valid_code, dim=0)
        if valid_code.size(0) == 0:# 防止出现loss为nan的情况
            flag = True
            continue
        distance += distance_activation(trial, torch.cdist(valid_code, valid_code), device).sum() / (valid_code.size(0) * (valid_code.size(0) - 1))/2
    if flag:
        return distance

    distance += cluster_activation(trial, torch.cdist(cluster, cluster), device).sum()/(10 * (10 - 1))/2

    return distance


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


def AE_eval(code_data, label, title):
    pca = PCA(n_components=2)
    pca.fit(code_data)
    code_data = pca.transform(code_data)
    plt.figure()
    plt.scatter(code_data[:, 0], code_data[:, 1], c=label, cmap='rainbow', s=0.5)
    plt.title(title)
    plt.colorbar()
    # plt.xticks(())
    # plt.yticks(())
    plt.savefig('pic2/' + title + '.png')
    plt.close()


def main(trial):
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
    beta1 = trial.suggest_float('beta1', 0.1, 15)
    beta2 = trial.suggest_float('beta2', 0.1, 15)
    beta3 = trial.suggest_float('beta3', 0.1, 15)
    beta4 = trial.suggest_float('beta4', 0.1, 15)
    beta5 = trial.suggest_float('beta5', 0.1, 15)
    beta6 = trial.suggest_float('beta6', 0.1, 15)
    gamma1 = trial.suggest_float('gamma1', 0.1, 15)
    gamma2 = trial.suggest_float('gamma2', 0.1, 15)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model1 = AutoEncoder1().to(device)
    model2 = AutoEncoder2().to(device)
    model3 = AutoEncoder3().to(device)
    mlp = MLP().to(device)

    criterion1 = nn.MSELoss().to(device)
    criterion2 = nn.CrossEntropyLoss().to(device)

    lr1 = trial.suggest_float('lr1', 1e-5, 1e-1, log=True)
    lr2 = trial.suggest_float('lr2', 1e-5, 1e-1, log=True)
    lr3 = trial.suggest_float('lr3', 1e-5, 1e-1, log=True)
    lr4 = trial.suggest_float('lr4', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay1', 1e-5, 1e-1, log=True)

    optimizer = torch.optim.Adam([
        {'params': model1.parameters(), 'lr': lr1},
        {'params': model2.parameters(), 'lr': lr2},
        {'params': model3.parameters(), 'lr': lr3},
        {'params': mlp.parameters(), 'lr': lr4},
    ], weight_decay=weight_decay)  # weight_decay不宜过大，lr不宜过大
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs*len(train_loader))

    best_acc = 0
    for epoch in range(epochs):
        model1.train()
        model2.train()
        model3.train()
        mlp.train()
        loss = 0
        loss1, loss2, loss3 = 0, 0, 0
        loss4, loss5, loss6 = 0, 0, 0
        loss7, loss8 = 0, 0
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            code1 = model1.get_code(data)
            code2 = model2.get_code(code1)
            code3 = model3.get_code(code2)
            recon3 = model3.decode(code3)
            recon2 = model2.decode(recon3)
            recon1 = model1.decode(recon2)
            output1 = model1(data)
            output2 = model2(code1)
            output3 = model3(code2)
            mlp_output = mlp(code3)

            loss1 = criterion1(recon1, data)
            loss2 = criterion1(recon2, code1)
            loss3 = criterion1(recon3, code2)
            loss4 = criterion1(output1, data)
            loss5 = criterion1(output2, code1)
            loss6 = criterion1(output3, code2)
            loss7 = criterion2(mlp_output, label)
            loss8 = distance_loss(trial, code3, label, device)
            # loss是单个AE输入输出的MSE，多个AE组合输入输出的MSE，MLP监督学习的交叉熵，distance loss(同类点距离，不同cluster距离)的加权和
            loss = beta1 * loss1 + beta2 * loss2 + beta3 * loss3 + beta4 * loss4 + \
                   beta5 * loss5 + beta6 * loss6 + gamma1 * loss7 + gamma2 * loss8

            optimizer.zero_grad()

            loss.backward()

            # 梯度裁剪
            for p in model1.parameters():
                torch.nn.utils.clip_grad_norm_(p, max_norm=1)
            for p in model2.parameters():
                torch.nn.utils.clip_grad_norm_(p, max_norm=1)
            for p in model3.parameters():
                torch.nn.utils.clip_grad_norm_(p, max_norm=1)
            for p in mlp.parameters():
                torch.nn.utils.clip_grad_norm_(p, max_norm=1)

            optimizer.step()

            scheduler.step()

        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Loss1: {loss1.item():.4f}, '
              f'Loss2: {loss2.item():.6f}, Loss3: {loss3.item():.8f}, '
              f'Loss4: {loss4.item():.4f}, Loss5: {loss5.item():.6f}, '
              f'Loss6: {loss6.item():.8f}, Loss7: {loss7.item():.4f}, Loss8: {loss8.item():.4f}')

        with torch.no_grad():
            total = 0
            correct = 0
            for idx, (data, label) in enumerate(test_loader):
                data = data.to(device)
                label = label.to(device)
                code1 = model1.get_code(data)
                code2 = model2.get_code(code1)
                code3 = model3.get_code(code2)
                mlp_output = mlp(code3)
                pred = mlp_output.argmax(dim=1)
                total += label.size(0)
                correct += torch.eq(pred, label).sum().item()

            print(f'Accuracy: {correct / total:.4f}')  # 0.94

            if correct / total > best_acc:
                best_acc = correct / total

            trial.report(correct / total, epoch)
            if trial.should_prune():  # 判断是否需要剪枝
                raise optuna.exceptions.TrialPruned()

    print('best_acc:', best_acc)

    return best_acc


if __name__ == "__main__":
    storage_name = "sqlite:///optuna.db"  # 存储的数据库,与打开dashboard时的数据库一致
    sampler = optuna.samplers.TPESampler()  # 选择采样器
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
        # 选择剪枝器，这里是中位数剪枝器，n_warmup_steps为预热步数，即前n_warmup_steps步不进行剪枝
        direction="maximize",  # 选择优化方向
        study_name="mnist_torch",  # 选择study名称
        storage=storage_name,  # 选择数据库
        load_if_exists=True,  # 如果数据库中存在study,则加载
        sampler=sampler  # 选择采样器
    )  # 创建study,用于记录参数和结果
    study.optimize(main, n_trials=50)  # 调用objective函数，进行超参数搜索，n_trials为搜索次数

    best_params = study.best_params
    best_value = study.best_value
    print("\n\nbest_value = " + str(best_value))  # 打印最优准确率
    print("best_params:")
    print(best_params)  # 打印最好的超参数
