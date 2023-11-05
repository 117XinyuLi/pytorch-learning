import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torchvision.models import swin_t

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = 'model/swin_transformer.pth'


def main():
    cifar10 = datasets.CIFAR10(root='CIFAR10', train=False, download=True, transform=transforms.Compose([
        transforms.Resize((224, 224),
                          interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    cifar10_test_loader = torch.utils.data.DataLoader(cifar10, batch_size=128, shuffle=True)

    model = swin_t(pretrained=True).to(device)  # 使用预训练模型
    model = nn.Sequential(model, nn.Linear(1000, 10)).to(device)
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
        print(f'Accuracy: {correct / total}') # 迁移学习 epoch=3, acc=0.9118


if __name__ == '__main__':
    main()
