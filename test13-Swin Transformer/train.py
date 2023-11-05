import os
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torchvision.models import swin_t

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = 'model/swin_transformer.pth'


def main():
    cifar10 = datasets.CIFAR10(root='CIFAR10', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.RandomHorizontalFlip(),
                                   transforms.Resize((232, 232),
                                                     interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                                   transforms.RandomCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                 ]))
    cifar10_train, cifar10_val = torch.utils.data.random_split(cifar10, [45000, 5000])
    cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=128, shuffle=True)
    cifar10_val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=128, shuffle=True)

    epochs = 100
    model = swin_t(pretrained=True).to(device)# 使用预训练模型
    model = nn.Sequential(model, nn.Linear(1000, 10)).to(device)

    # 只训练最后一层
    for param in model.parameters():
        param.requires_grad = False
    for param in model[-1].parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs*len(cifar10_train_loader), T_mult=1, eta_min=1e-6)

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print('Load model from', save_path)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(cifar10_train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for data, target in cifar10_val_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                pred = logits.argmax(dim=1)
                total += target.size(0)
                correct += torch.eq(pred, target).sum().item()
            print(f'Epoch: {epoch}, Accuracy: {correct / total}')
            if correct / total > best_acc:
                best_acc = correct / total
                torch.save(model.state_dict(), save_path)

    print(f'Best accuracy: {best_acc}')


if __name__ == '__main__':
    main()
