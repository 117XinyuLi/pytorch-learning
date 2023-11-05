import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from VariationalAutoEncoder import VAE
from visdom import Visdom


def main():
    mnist_train = datasets.MNIST(root='data', train=True, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                 ]))
    mnist_test = datasets.MNIST(root='data', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]))
    train_loader = DataLoader(mnist_train, batch_size=512, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)

    epochs = 1000
    beta = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE().to(device)
    model.load_state_dict(torch.load('models/VAE.pth'))
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs*len(train_loader))

    viz = Visdom()
    for epoch in range(epochs):
        model.train()
        loss = 0
        kld = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            x_hat, kld = model(data)
            loss = criterion(x_hat, data)
            if kld is not None:
                loss += beta * kld
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        print(f'Epoch: {epoch}, Loss: {loss.item()}, KLD: {kld.item()}')

        x = next(iter(test_loader))[0].to(device)# [0] is data
        with torch.no_grad():
            x_hat, _ = model(x)
            # 输出图片(图片会随着epoch的变化而变化)
            viz.images(x, nrow=8, win='x', opts=dict(title='x'))
            viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))

        torch.save(model.state_dict(), 'models/VAE.pth')
    """"""

    # 生成图片
    model.eval()
    with torch.no_grad():
        z = torch.randn(64, 10).to(device)
        x_hat = model.decoder(z)
        x_hat = x_hat.view(x_hat.size(0), 1, 28, 28)
        viz = Visdom()
        viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))

        # 保存图片
        import torchvision.utils as vutils
        vutils.save_image(x_hat, 'images/x_hat.png', nrow=8)



if __name__ == "__main__":
    main()
