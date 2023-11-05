import optuna
import torch
import torch.nn as nn
import torchvision

# 打开optuna dashboard
# 打开anaconda prompt，输入：conda activate pytorch，进入pytorch环境，
# 使用cd定位到工作文件夹，输入：optuna-dashboard --host 0.0.0.0  --port 8083 sqlite:///optuna.db
# 打开浏览器，输入：http://localhost:8083，即可打开optuna dashboard
# 同时会在工作文件夹下生成optuna.db文件，用于保存optuna的数据
# 可以在dashboard中查看每次搜索的结果，每次搜索的超参数，还有操作删除study等
# 删除历史后需要重启网页,否则看不到删除后的结果

try_times = 0  # 全局变量，用于记录搜索的次数


def data_loader(batch_size=512):  # 正常的数据加载函数
    train_db = torchvision.datasets.MNIST('mnist_data', train=True,  # 使用train data
                                          download=True,  # 下载数据集
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.RandomRotation(15),  # 随机旋转15度
                                              torchvision.transforms.RandomCrop(28, padding=4),  # 随机裁剪
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


class Net(nn.Module):  # 网络结构
    def __init__(self, conv1_channels, conv2_channels, conv3_channels, conv4_channels, FC1_channels, FC2_channels, dropout_rate1, dropout_rate2, activation):
        super(Net, self).__init__()
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'SELU':
            self.activation = nn.SELU(inplace=True)
        self.model = nn.Sequential(
            nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv1_channels),
            nn.MaxPool2d(2),
            self.activation,

            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv2_channels),
            nn.MaxPool2d(2),
            self.activation,

            nn.Conv2d(conv2_channels, conv3_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv3_channels),
            nn.MaxPool2d(2),
            self.activation,

            nn.Conv2d(conv3_channels, conv4_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv4_channels),
            nn.MaxPool2d(2),
            self.activation,

            nn.Flatten(),

            nn.Linear(conv4_channels, FC1_channels),
            self.activation,
            nn.Dropout(dropout_rate1),

            nn.Linear(FC1_channels, FC2_channels),
            self.activation,
            nn.Dropout(dropout_rate2),

            nn.Linear(FC2_channels, 10)


        )

    def forward(self, x):
        return self.model(x)


def model_fn(trial, device):  # 定义模型，trial为optuna的trial对象，其中的内容需要进行超参数搜索
    conv1_channels = trial.suggest_int('conv1_channels', 32, 64)  # 定义超参数范围，这里是32到64的整数
    conv2_channels = trial.suggest_int('conv2_channels', 64, 128)  # 定义超参数范围，这里是64到128的整数
    conv3_channels = trial.suggest_int('conv3_channels', 128, 256)  # 定义超参数范围，这里是128到256的整数
    conv4_channels = trial.suggest_int('conv4_channels', 256, 512)  # 定义超参数范围，这里是256到512的整数
    dropout_rate1 = trial.suggest_float('dropout_rate1', 0.2, 0.5)  # 定义超参数范围，这里是0.2到0.5的浮点数
    dropout_rate2 = trial.suggest_float('dropout_rate2', 0.2, 0.5)  # 定义超参数范围，这里是0.2到0.5的浮点数
    activation = trial.suggest_categorical('activation', ['ReLU', 'LeakyReLU', 'SELU'])  # 定义超参数范围，这里是三个激活函数
    FC1_channels = trial.suggest_int('FC1_channels', 128, 256)  # 定义超参数范围，这里是128到256的整数
    FC2_channels = trial.suggest_int('FC2_channels', 64, 128)  # 定义超参数范围，这里是64到128的整数
    model = Net(conv1_channels, conv2_channels, conv3_channels, conv4_channels, FC1_channels, FC2_channels, dropout_rate1, dropout_rate2, activation).to(device)  # 实例化网络结构
    return model


def test(model, test_loader, device):  # 正常的test set上的测试函数
    model.eval()
    test_correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        test_correct += pred.eq(target).sum().float().item()
    total_num = len(test_loader.dataset)
    return test_correct / total_num


def test_while_training(model, val_loader, criterion, device):  # 正常的在训练过程中dev set上的测试函数
    model.eval()
    loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss += criterion(output, target).item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().float().item()
    total_num = len(val_loader.dataset)

    return correct / total_num


def train(model, optimizer, criterion, scheduler, train_loader, device, epoch, test_interval=25):  # 训练模型，正常的训练函数
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % test_interval == 0 and batch_idx > 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
    scheduler.step()


def objective(trial):  # 定义主函数，trial为optuna的trial对象，其中的内容需要进行超参数搜索
    train_loader, dev_loader, test_loader = data_loader()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model_fn(trial, device)  # 调用model_fn函数，定义模型

    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])  # 定义超参数范围，这里是两个优化器
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)  # 定义超参数lr范围，这里是1e-5到1e-1的浮点数，按对数均匀分布
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)  # 定义超参数weight_decay范围，这里是1e-5到1e-1的浮点数，按对数均匀分布
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)  # 实例化优化器

    epochs = 15

    scheduler_name = trial.suggest_categorical('scheduler', ['StepLR', 'CosineAnnealingWarmRestarts', 'OneCycleLR'])  # 定义超参数范围，这里是三个学习率调整策略
    if scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs*len(train_loader)//5, gamma=0.1)  # 定义可调节的学习率调整策略
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs*len(train_loader))  # 定义可调节的学习率调整策略
    elif scheduler_name == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=7*lr, total_steps=epochs*len(train_loader), three_phase=True) # 定义可调节的学习率调整策略

    criterion = nn.CrossEntropyLoss()  # 定义损失函数

    global try_times  # 获取全局变量
    try_times += 1

    for epoch in range(epochs):
        train(model, optimizer, criterion, scheduler, train_loader, device, epoch)
        accuracy = test_while_training(model, dev_loader, criterion, device)
        trial.report(accuracy, epoch)  # 报告当前epoch的准确率，根据报告来进行超参数搜索和剪枝
        if trial.should_prune():  # 判断是否需要剪枝
            raise optuna.exceptions.TrialPruned()

    test_acc = test(model, test_loader, device)  # 测试模型
    torch.save(model.state_dict(), 'models/model No.' + str(try_times) + ' ' + str(test_acc) + '.pth')  # 保存模型，提前剪枝的模型不会保存

    return test_acc # 结束一轮尝试后会返回 Trial ... finished with value: ... and parameters: {...} 其中value为objective的返回值，parameters为超参数
    # 最后best_value会在objective的各个返回值中选取(其实不应该使用test set进行模型选择，但是这里只是为了演示)


if __name__ == '__main__':
    storage_name = "sqlite:///optuna.db"  # 存储的数据库,与打开dashboard时的数据库一致
    sampler = optuna.samplers.TPESampler()  # 选择采样器
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        # 选择剪枝器，这里是中位数剪枝器，n_warmup_steps为预热步数，即前n_warmup_steps步不进行剪枝
        direction="maximize",  # 选择优化方向
        study_name="mnist_torch",  # 选择study名称
        storage=storage_name,  # 选择数据库
        load_if_exists=True,  # 如果数据库中存在study,则加载
        sampler=sampler  # 选择采样器
    )  # 创建study,用于记录参数和结果
    study.optimize(objective, n_trials=25)  # 调用objective函数，进行超参数搜索，n_trials为搜索次数

    best_params = study.best_params
    best_value = study.best_value
    print("\n\nbest_value = " + str(best_value))  # 打印最优准确率
    print("best_params:")
    print(best_params)  # 打印最好的超参数
