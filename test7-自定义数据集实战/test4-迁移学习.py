import torch
from torch import nn
import visdom
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from pokemon import Pokemon

batchsz = 64
lr = 1e-3
epochs = 32

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
torch.manual_seed(1234)

train_db = Pokemon('pokemon', 224, mode='train')
val_db = Pokemon('pokemon', 224, mode='val')
test_db = Pokemon('pokemon', 224, mode='test')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=4)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=2)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=2)



def evalute(model, loader):
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        model.eval()
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total

def main():
    # 迁移学习
    pre_model = resnet18(pretrained=True)
    model = nn.Sequential(*list(pre_model.children())[:-1], # 去掉最后一层
                        nn.Flatten(),
                        nn.Linear(512, 5)).to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_acc, best_epoch = 0, 0
    global_step = 0
    viz = visdom.Visdom()
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            model.train()
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criteon(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 5 == 0:
                print(f'epoch: {epoch}, step: {step}, loss: {loss.item():.4f}')

            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        if (epoch % 2 == 0 and epoch > 0) or epoch == epochs - 1:
            val_acc = evalute(model, val_loader)
            print(f'epoch: {epoch}, val_acc: {val_acc}')
            viz.line([val_acc], [epoch], win='val_acc', update='append')
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                torch.save(model.state_dict(), 'best-transfer.mdl')

    print(f'best epoch: {best_epoch}, best val_acc: {best_acc}')# best epoch: 10, best val_acc: 0.97424
    model.load_state_dict(torch.load('best-transfer.mdl'))
    print('loaded from ckpt!')
    train_acc = evalute(model, train_loader)
    print(f'train acc: {train_acc}')# train acc: 0.9886
    test_acc = evalute(model, test_loader)
    print(f'test acc: {test_acc}')# test acc: 0.9872


if __name__ == '__main__':
    main()
