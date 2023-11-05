import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from AutoEncoder import AutoEncoder, ConvAutoEncoder
from visdom import Visdom
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def get_data(test_loader, model, device):
    model.eval()
    datas = []
    code_datas = []
    labels = []
    for batch_idx, (data, label) in enumerate(test_loader):
        data = data.to(device)
        code_data = model.get_code(data)
        datas.append(data)
        code_datas.append(code_data)
        labels.append(label)
    datas = torch.cat(datas, dim=0).cpu().detach().numpy()
    code_datas = torch.cat(code_datas, dim=0).cpu().detach().numpy()
    labels = torch.cat(labels, dim=0).cpu().detach().numpy()
    return datas, code_datas, labels


def t_SNE_eval(data, label, title):
    data = data.reshape(data.shape[0], -1)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(data)
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=label, cmap='rainbow')
    plt.title(title)
    plt.colorbar()
    plt.xticks(())
    plt.yticks(())
    plt.savefig('pic/' + title + '.png')


def AE_eval(code_data, label, title):
    plt.figure()
    plt.scatter(code_data[:, 0], code_data[:, 1], c=label, cmap='rainbow')
    plt.title(title)
    plt.colorbar()
    plt.xticks(())
    plt.yticks(())
    plt.savefig('pic/' + title + '.png')


def CAE_eval(code_data, label, title):
    plt.figure()
    plt.scatter(code_data[:, 0], code_data[:, 1], c=label, cmap='rainbow')
    plt.title(title)
    plt.colorbar()
    plt.xticks(())
    plt.yticks(())
    plt.savefig('pic/' + title + '.png')


def main():
    mnist_train = datasets.MNIST(root='data', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     # 加入噪声
                                     transforms.Lambda(lambda x: x + 0.01 * torch.rand_like(x))
                                 ]))
    mnist_test = datasets.MNIST(root='data', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor()
                                ]))
    train_loader = DataLoader(mnist_train, batch_size=512, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)

    epochs = 100# epoch大一点，网络深一点，效果好

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load('models/AutoEncoder.pth'))

    # model = ConvAutoEncoder().to(device)
    # model.load_state_dict(torch.load('models/ConvAutoEncoder.pth'))

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)# 加入weight_decay的效果不好, 加上BN层后效果好了
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs*len(train_loader))

    viz = Visdom()
    for epoch in range(epochs):
        model.train()
        loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

        x = next(iter(test_loader))[0].to(device)# [0] is data
        with torch.no_grad():
            out = model(x)
            # 输出图片(图片会随着epoch的变化而变化)
            viz.images(x, win='input', opts=dict(title='input'))
            viz.images(out, win='output', opts=dict(title='output'))

        # 保存模型
        if (epoch % 50 == 0 and epoch != 0) or epoch == epochs - 1:
            torch.save(model.state_dict(), 'models/AutoEncoder.pth')

    torch.save(model.state_dict(), 'models/AutoEncoder.pth')

    # torch.save(model.state_dict(), 'models/ConvAutoEncoder.pth')

    data, code_data, label = get_data(test_loader, model, device)

    # t_SNE_eval(data, label, 'data_t-SNE')# 对数据进行t-SNE降维

    AE_eval(code_data, label, 'data_AE')# 对数据进行AE降维

    # CAE_eval(code_data, label, 'data_CAE')# 对数据进行CAE降维


if __name__ == "__main__":
    main()
