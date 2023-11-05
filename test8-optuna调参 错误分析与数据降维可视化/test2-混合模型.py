import torch
from torchvision import transforms, datasets
from torchvision.models import resnet18, resnet50
from torch import nn
from torch.nn import functional as F
import optuna

batch_size = 50

cifar_test = datasets.CIFAR10(root='CIFAR10', train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])
                              ]))
# 这里直接使用了CIFAR10的测试集，作为验证集，进行调参
cifar_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pre_trained = resnet18(pretrained=True)

model1 = nn.Sequential(
    *list(pre_trained.children())[:-1],
    nn.Flatten(),
    nn.Linear(512, 10)
).to(device)

model1.load_state_dict(torch.load('cifar10-resnet18-transfer2.pth'))

pre_trained = resnet50(pretrained=True)
model2 = nn.Sequential(
    *list(pre_trained.children())[:-1],
    nn.Flatten(),
    nn.Linear(2048, 10)
).to(device)

model2.load_state_dict(torch.load('cifar10-resnet50-transfer2.pth'))


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


result1, label1 = get_result(model1, device, cifar_test_loader)
result2, _ = get_result(model2, device, cifar_test_loader)


def objective(trial):
    coef1 = trial.suggest_float('coef1', 0, 1)
    coef2 = trial.suggest_float('coef2', 0, 1)
    result = coef1 * result1 + coef2 * result2
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

    # resnet18: test acc 0.967 resnet50: test acc 0.963
    # 混合模型: coef1: 0.8302555651972333, coef2: 0.6921207508341242 test acc 0.9730
