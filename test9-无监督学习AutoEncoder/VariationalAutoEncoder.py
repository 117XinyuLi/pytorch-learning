import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder [batch, 784] => [batch, 20]
        # μ：[batch, 10], σ：[batch, 10]
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 20),
        )
        # Decoder [batch, 10] => [batch, 784]
        self.decoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # [batch, 20],including μ and σ
        h = self.encoder(x)
        # [batch, 20] => [batch, 10] and [batch, 10]
        mu, sigma = h.chunk(2, dim=1)
        # reparameterization trick
        h = mu + sigma * torch.randn_like(sigma)
        # [batch, 10] => [batch, 1, 28, 28]
        x_hat = self.decoder(h)
        x_hat = x_hat.view(batch_size, 1, 28, 28)
        # KL divergence 这里希望得到的(μ1,σ2)...(μn,σn)均是一个均值为0，方差为1的正态分布，那么对(μ1,σ2)...(μn,σn)分别和标准正态分布计算KL散度，然后求和
        kld = 0.5 * torch.sum(
                        torch.pow(mu, 2) +
                        torch.pow(sigma, 2) -
                        torch.log(1e-8 + torch.pow(sigma, 2)) - 1)/(batch_size*28*28)# 28*28是因为要和图片的像素点对应,因为MSE是对每个像素点求平均的

        return x_hat, kld
