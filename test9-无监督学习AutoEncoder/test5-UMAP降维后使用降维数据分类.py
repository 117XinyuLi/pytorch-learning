import torch
import torch.nn as nn
from torchvision import datasets, transforms
from umap import UMAP
from matplotlib import pyplot as plt

# 使用UMAP降维后的数据进行分类


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.LeakyReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.25),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MLP().to(device)

    data_numpy = mnist_train.data.numpy()
    data_numpy = data_numpy.reshape(data_numpy.shape[0], -1)
    umap = UMAP(n_components=2, n_neighbors=5, min_dist=0.1, metric='euclidean')
    data_umap = umap.fit_transform(data_numpy)

    plt.scatter(data_umap[:, 0], data_umap[:, 1], c=mnist_train.targets, cmap='rainbow', s=0.5)
    plt.title('data_UMAP')
    plt.colorbar()
    plt.savefig('pic/data_UMAP.png')

    train_umap = torch.from_numpy(data_umap).float().to(device)
    train_label = mnist_train.targets.to(device)

    test_numpy = mnist_test.data.numpy()
    test_numpy = test_numpy.reshape(test_numpy.shape[0], -1)
    test_umap = umap.transform(test_numpy)
    test_umap = torch.from_numpy(test_umap).float().to(device)
    test_label = mnist_test.targets.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        model.train()
        out = model(train_umap)
        loss = criterion(out, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch: {}, loss: {}'.format(epoch, loss.item()))

        model.eval()
        with torch.no_grad():
            out = model(test_umap)
            pred = torch.argmax(out, dim=1)
            acc = torch.mean((pred == test_label).float())
            print('test acc: {}'.format(acc.item()))# 0.9+


if __name__ == '__main__':
    main()