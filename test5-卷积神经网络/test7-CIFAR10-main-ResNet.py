import torch
from torchvision import transforms, datasets
from ResNet import ResNet18


def main():
    batchsz = 72
    cifar_train = datasets.CIFAR10(root='CIFAR10', train=True, download=True,
                                   transform=transforms.Compose([
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
    model = ResNet18().to(device)
    # model.load_state_dict(torch.load('models/cifar10-resnet18.pth'))
    criteon = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3*(0.5**6), weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(50):
        model.train()
        loss_mean = 0.
        batch_num = 0
        for batch_idx, (x, label) in enumerate(cifar_train_loader):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)
            loss_mean += loss.item()
            batch_num += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % (len(cifar_train_loader) // 4) == 0 and batch_idx != 0:
                print(f'epoch: {epoch}, processing: {batch_idx * 100 / len(cifar_train_loader):.2f}%, loss: {loss_mean / batch_num:.4f}')
                loss_mean = 0.
                batch_num = 0

        scheduler.step()

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

        if (epoch + 1) % 10 == 0 and epoch > 0:
            torch.save(model.state_dict(), 'models/cifar10-resnet18-epoch' + str(epoch) + '.pth')

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

    torch.save(model.state_dict(), 'models/cifar10-resnet18.pth')


if __name__ == '__main__':
    main()
