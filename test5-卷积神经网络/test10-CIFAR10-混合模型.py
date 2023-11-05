import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.models import googlenet, resnet50, densenet121
from InceptionResNet import IncResNet
from ResNet import ResNet18
import optuna

batchsz = 128

cifar_test = datasets.CIFAR10(root='CIFAR10', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.Resize([32, 32]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                ]))
# 这里直接使用了CIFAR10的测试集，作为验证集，进行调参
# 注：这里的模型输入的图片大小是32*32，而有些transfer learning的正常输入为224*224，导致有的模型训练效果不好
cifar_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=batchsz, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pre_trained = googlenet(pretrained=True)
model1 = nn.Sequential(
    *list(pre_trained.children())[:-1],
    nn.Flatten(),
    nn.Linear(1024, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
).to(device) # 1. googlenet test acc: 0.8730
model1.load_state_dict(torch.load('models/cifar10-GoogleNet-transfer.pth'))

model2 = IncResNet().to(device) # 2. InceptionResNet test acc: 0.9088
model2.load_state_dict(torch.load('models/cifar10-IncResNet-epoch100.pth'))

model3 = ResNet18().to(device) # 3. ResNet18 test acc: 0.8701
model3.load_state_dict(torch.load('models/cifar10-resnet18.pth'))

pre_trained = resnet50(pretrained=True) # 4. resnet50 test acc: 0.8965
model4 = nn.Sequential(
    *list(pre_trained.children())[:-1],
    nn.Flatten(),
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.BatchNorm1d(1024),
    nn.Dropout(0.5),
    nn.Linear(1024, 10)
).to(device)
model4.load_state_dict(torch.load('models/cifar10-resnet50-transfer.pth'))

pre_trained = densenet121(pretrained=True) # 5. densenet121 test acc: 0.8998
model5 = nn.Sequential(
    *list(pre_trained.children())[:-1],
    nn.Flatten(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
).to(device)
model5.load_state_dict(torch.load('models/cifar10-densenet121-transfer.pth'))


def get_result(model, device, loader):
    model.eval()
    prob = []
    labels = []
    with torch.no_grad():  # 不计算梯度
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            result = model(data)
            logit = F.softmax(result, dim=1)
            prob.append(logit)
            labels.append(label)
    prob = torch.cat(prob, dim=0)
    labels = torch.cat(labels, dim=0)
    return prob, labels


def get_acc(result, labels):
    pred = torch.argmax(result, dim=1)
    acc = torch.eq(pred, labels).float().mean().item()
    return acc


result1, label1 = get_result(model1, device, cifar_test_loader)
print(f'1. googlenet test acc: {get_acc(result1, label1)}')
result2, label2 = get_result(model2, device, cifar_test_loader)
print(f'2. InceptionResNet test acc: {get_acc(result2, label2)}')
result3, label3 = get_result(model3, device, cifar_test_loader)
print(f'3. ResNet18 test acc: {get_acc(result3, label3)}')
result4, label4 = get_result(model4, device, cifar_test_loader)
print(f'4. resnet50 test acc: {get_acc(result4, label4)}')
result5, label5 = get_result(model5, device, cifar_test_loader)
print(f'5. densenet121 test acc: {get_acc(result5, label5)}')


def objective(trial):
    coef1 = trial.suggest_float('coef1', 0, 1)
    coef2 = trial.suggest_float('coef2', 0, 1)
    coef3 = trial.suggest_float('coef3', 0, 1)
    coef4 = trial.suggest_float('coef4', 0, 1)
    coef5 = trial.suggest_float('coef5', 0, 1)
    result = coef1 * result1 + coef2 * result2 + coef3 * result3 + coef4 * result4 + coef5 * result5
    pred = result.argmax(dim=1)
    acc = torch.eq(pred, label1).float().mean().item()
    trial.report(acc, 0)
    return acc


if __name__ == '__main__':
    storage_name = "sqlite:///optuna.db"  # 存储的数据库,与打开dashboard时的数据库一致
    sampler = optuna.samplers.TPESampler()  # 选择采样器
    study = optuna.create_study(
        direction="maximize",  # 选择优化方向
        study_name="mix_model",  # 选择study名称
        storage=storage_name,  # 选择数据库
        load_if_exists=True,  # 如果数据库中存在study,则加载
        sampler=sampler  # 选择采样器
    )  # 创建study,用于记录参数和结果
    study.optimize(objective, n_trials=100)  # 调用objective函数，进行超参数搜索，n_trials为搜索次数

    best_params = study.best_params
    best_value = study.best_value
    print("\n\nbest_value = " + str(best_value))  # 打印最优准确率
    print("best_params:")
    print(best_params)  # 打印最好的超参数

    # 混合模型
    # coef1: 0.29065782932322237, coef2: 0.9506050567333223, coef3: 0.5031553435469092, coef4: 0.3476444817196356, coef5: 0.7612714367529596
    # test acc: 0.9307
