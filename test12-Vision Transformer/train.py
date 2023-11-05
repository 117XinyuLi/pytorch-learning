import os
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from vit_pytorch import SimpleViT  # 使用vit_pytorch中的SimpleViT, ViT
from torchvision.models import vit_b_16  # 也可以使用torchvision中的ViT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = 'model/ViT.pth'


def main():
    cifar10 = datasets.CIFAR10(root='CIFAR10', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.RandomHorizontalFlip(),
                                   transforms.Resize((256, 256),
                                                     interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                                   transforms.RandomCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ]))
    cifar10_train, cifar10_val = torch.utils.data.random_split(cifar10, [45000, 5000])
    cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=128, shuffle=True)
    cifar10_val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=128, shuffle=True)

    epochs = 100

    # 自定义模型
    # model = SimpleViT(image_size=224, patch_size=16, num_classes=10,dim=1024, depth=6, heads=16, mlp_dim=2048).to(device)

    # 使用预训练模型
    model = vit_b_16(pretrained=True).to(device)
    model.heads = nn.Sequential(
        nn.Linear(768, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 10)
    ).to(device)

    # 只更新最后一层
    for param in model.parameters():
        param.requires_grad = False
    for param in model.heads.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs * len(cifar10_train_loader),
                                                                     T_mult=1, eta_min=1e-6)
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
