# CIFAR10
# 10个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
# 每个类别有6000张32*32的彩色图片，训练集有50000张，测试集有10000张

import torch
from torchvision import transforms, datasets
from LeNet5 import LeNet5

def main():
    batchsz = 64
    cifar_train = datasets.CIFAR10(root='CIFAR10', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize([32, 32]),
                                    transforms.ToTensor()
                                ]))
    print(len(cifar_train))
    cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10(root='CIFAR10', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.Resize([32, 32]),
                                    transforms.ToTensor()
                                ]))
    print(len(cifar_test))
    cifar_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=batchsz, shuffle=False)

    x, y = next(iter(cifar_train_loader))
    print(x.shape, y.shape)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LeNet5().to(device)
    # model.load_state_dict(torch.load('models/LeNet5-CIFAR10.pth'))
    criteon = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    for epoch in range(50):
        model.train()
        loss_mean = 0.
        for batch_idx, (x, label) in enumerate(cifar_train_loader):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)
            loss_mean += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch, loss_mean / len(cifar_train_loader))
        scheduler.step()

        model.eval()
        with torch.no_grad():# 不计算梯度
            total_correct = 0
            total_num = 0
            for x, label in cifar_test_loader:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)
            print(total_correct / total_num)

        if(epoch % 10 == 0 and epoch != 0):
            batchsz *= 2
            cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=batchsz, shuffle=True)
            cifar_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=batchsz, shuffle=False)


    model.eval()
    with torch.no_grad():  # 不计算梯度
        total_correct = 0
        total_num = 0
        for x, label in cifar_train_loader:
            x, label = x.to(device), label.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            total_correct += torch.eq(pred, label).float().sum().item()
            total_num += x.size(0)
        print(f'train acc: {total_correct / total_num}')

    model.eval()
    with torch.no_grad():  # 不计算梯度
        total_correct = 0
        total_num = 0
        for x, label in cifar_test_loader:
            x, label = x.to(device), label.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            total_correct += torch.eq(pred, label).float().sum().item()
            total_num += x.size(0)
        print(f'test acc: {total_correct / total_num}')

    # torch.save(model.state_dict(), 'models/LeNet5-CIFAR10.pth')


if __name__ == '__main__':
    main()
