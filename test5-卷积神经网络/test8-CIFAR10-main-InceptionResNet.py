import torch
from torchvision import transforms, datasets
from InceptionResNet import IncResNet


def main():
    batchsz = 32
    iter_num = 8
    cifar_train = datasets.CIFAR10(root='CIFAR10', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(15),
                                       transforms.RandomCrop(32, padding=4),
                                       transforms.Resize([32, 32]),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                   ]))

    cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10(root='CIFAR10', train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.Resize([32, 32]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                                  ]))

    cifar_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=batchsz, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = IncResNet().to(device)
    # model.load_state_dict(torch.load('models/cifar10-IncResNet-epoch50.pth'))
    criteon = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3*(0.5**5), weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(60):
        model.train()
        loss_mean = 0.
        batch_num = 0
        for batch_idx, (x, label) in enumerate(cifar_train_loader):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)
            loss = loss / iter_num # 注意这里的loss要除以iter_num，使梯度分到每个iter上
            loss.backward()

            loss_mean += loss.item() * iter_num
            batch_num += 1

            # 累计梯度，近似实现batch_size增大
            if (batch_idx + 1) % iter_num == 0 or batch_idx == len(cifar_train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()

            if batch_idx % (len(cifar_train_loader) // 4) == 0 and batch_idx != 0:
                print(f'epoch: {epoch}, processing: {batch_idx * 100 / len(cifar_train_loader):.2f}%, loss: {loss_mean / batch_num:.4f}')
                loss_mean = 0.
                batch_num = 0

        scheduler.step()

        # 打乱数据集
        cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=batchsz, shuffle=True)
        
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
            print(f'epoch: {epoch}, accuracy: {total_correct / total_num:.4f}')

        if (epoch + 1) % 3 == 0 and epoch > 0:
            torch.save(model.state_dict(), 'models/cifar10-IncResNet-epoch' + str(epoch) + '.pth')
            iter_num *= 2

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

    torch.save(model.state_dict(), 'models/cifar10-IncResNet.pth')


if __name__ == '__main__':
    main()
