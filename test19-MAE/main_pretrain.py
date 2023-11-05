# 用pytorch在cifar10上pretrain一个MAE

import os
import torchvision
from torchvision import datasets, transforms
from models_mae import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = 'model/MAE.pth'
torch.backends.cudnn.benchmark = True


def main():
    cifar10 = datasets.CIFAR10(root='CIFAR10', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.RandomHorizontalFlip(),
                                   transforms.Resize((256, 256), interpolation=3),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ]))
    cifar10_train, cifar10_val = torch.utils.data.random_split(cifar10, [45000, 5000])
    cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=12, shuffle=True)
    cifar10_val_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=12, shuffle=True)

    model = mae_vit_base_patch16().to(device)

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print('Load model from', save_path)

    epochs = 100
    lr = 5e-4
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                        T_0=epochs * len(cifar10_train_loader),
                                                                        T_mult=1, eta_min=1e-6)

    low_loss = 0
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, _) in enumerate(cifar10_train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            loss, pred, mask = model(data)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch_idx, (data, _) in enumerate(cifar10_val_loader):
                data = data.to(device)
                loss, pred, mask = model(data)
                total_loss += loss.item()
            total_loss = total_loss / len(cifar10_val_loader)
            print(f'Epoch: {epoch}, Val Loss: {total_loss}')
            if low_loss == 0 or total_loss < low_loss:
                low_loss = total_loss
                torch.save(model.state_dict(), save_path)
                print('Save model to', save_path)

    print(f'Lowest loss: {low_loss}')


if __name__ == '__main__':
    main()
