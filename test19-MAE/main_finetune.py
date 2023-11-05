# 用pytorch在cifar10上对MAE的encoder进行fine-tune

import torchvision
from torchvision import datasets, transforms
import util.lr_decay as lrd
from util.load_model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = 'model/MAE_FT.pth'
MAE_path = 'model/MAE.pth'
transfer_path = 'model/mae_pretrain_vit_base.pth'
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

    model = load_vit_base_patch16(num_classes=10, save_path=save_path, MAE_path=MAE_path, transfer_path=transfer_path, device=device)

    epochs = 100
    lr = 5e-4
    weight_decay = 0.05
    layer_decay = 0.75 # layer-wise lr decay
    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model, weight_decay=weight_decay,
                                        no_weight_decay_list=model.no_weight_decay(),
                                        layer_decay=layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                        T_0=epochs * len(cifar10_train_loader),
                                                                        T_mult=1, eta_min=1e-6)
    criterion = torch.nn.CrossEntropyLoss()

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
            lr_scheduler.step()
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
