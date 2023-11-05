import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import resnet18, resnet50


def evaluate(model, device, loader):
    model.eval()
    with torch.no_grad():  # 不计算梯度
        total_correct = 0
        total_num = 0
        for x, label in loader:
            x, label = x.to(device), label.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            total_correct += torch.eq(pred, label).float().sum().item()
            total_num += x.size(0)
        return total_correct / total_num


def main():
    batchsz = 128
    epochs = 30
    torch.backends.cudnn.benchmark = True
    cifar_train = datasets.CIFAR10(root='CIFAR10', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomRotation(15),
                                       transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                                       transforms.RandomCrop(224, padding=24),# Resnet的输入是224*224
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]),
                                       transforms.RandomErasing(p=0.5, scale=(0.2, 0.25), ratio=(0.8, 1)),
                                   ]))

    cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10(root='CIFAR10', train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                                  ]))

    cifar_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=batchsz, shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pre_trained = resnet18(pretrained=True)
    model = nn.Sequential(
        *list(pre_trained.children())[:-1],
        nn.Flatten(),
        nn.Linear(512, 10)
    ).to(device)

    """
    pre_trained = resnet50(pretrained=True)
    model = nn.Sequential(
        *list(pre_trained.children())[:-1],
        nn.Flatten(),
        nn.Linear(2048, 10)
    ).to(device)
    """

    model.load_state_dict(torch.load('models/cifar10-resnet18-transfer2.pth'))

    """
    model.load_state_dict(torch.load('models/cifar10-resnet50-transfer2.pth'))
    """

    criteon = torch.nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4) # 100epoch 到40epoch的时候CUDA报错了
    # optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=1e-3) # 30epoch
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-3) # 30epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs * len(cifar_train_loader))

    best_acc, best_epoch = 0, 0
    for epoch in range(epochs):
        model.train()
        loss_mean = 0.
        batch_num = 0
        for batch_idx, (x, label) in enumerate(cifar_train_loader):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)
            loss.backward()

            for p in model.parameters():
                torch.nn.utils.clip_grad_norm_(p, 1)

            loss_mean += loss.item()
            batch_num += 1

            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % (len(cifar_train_loader) // 4) == 0 and batch_idx != 0:
                print(f'epoch: {epoch}, processing: {batch_idx * 100 / len(cifar_train_loader):.2f}%, loss: {loss_mean / batch_num:.4f}')
                loss_mean = 0.
                batch_num = 0

            scheduler.step()

        acc = evaluate(model, device, cifar_test_loader)
        print(f'epoch: {epoch}, accuracy: {acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), 'models/cifar10-resnet18-transfer2.pth')

    print(f'best acc: {best_acc}, best epoch: {best_epoch}')
    model.load_state_dict(torch.load('models/cifar10-resnet18-transfer2.pth'))
    print('load from ckpt')

    train_acc = evaluate(model, device, cifar_train_loader)
    test_acc = evaluate(model, device, cifar_test_loader)
    print(f'train acc: {train_acc}, test acc: {test_acc}')# resnet18 0.99 0.967 resnet50 0.99 0.963


if __name__ == '__main__':
    main()