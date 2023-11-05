import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
from se_resnet import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = 'model/SENet.pth'


def main():
    cifar10 = datasets.CIFAR10(root='CIFAR10', train=False, download=True, transform=transforms.Compose([
        transforms.Resize((224, 224),
                          interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    cifar10_test_loader = torch.utils.data.DataLoader(cifar10, batch_size=16, shuffle=True)

    model = se_resnet50(pretrained=True).to(device)  # 使用预训练模型
    model = nn.Sequential(*list(model.children())[:-1],
                          nn.Flatten(),
                          nn.Linear(2048, 10),
                          ).to(device)
    model.load_state_dict(torch.load(save_path))

    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for data, target in cifar10_test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            pred = logits.argmax(dim=1)
            total += target.size(0)
            correct += torch.eq(pred, target).sum().item()
        print(f'Accuracy: {correct / total}')# 迁移学习 epoch:2 acc:0.8902


if __name__ == '__main__':
    main()
