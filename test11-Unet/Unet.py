import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.2),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel//2, kernel_size=1, stride=1, padding=0)

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        out = torch.cat([out, feature_map], dim=1)
        return out


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.c1 = ConvBlock(3, 64)
        self.d1 = DownSample(64)
        self.c2 = ConvBlock(64, 128)
        self.d2 = DownSample(128)
        self.c3 = ConvBlock(128, 256)
        self.d3 = DownSample(256)
        self.c4 = ConvBlock(256, 512)
        self.d4 = DownSample(512)
        self.c5 = ConvBlock(512, 1024)
        self.u1 = UpSample(1024)
        self.c6 = ConvBlock(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = ConvBlock(512, 256)
        self.u3 = UpSample(256)
        self.c8 = ConvBlock(256, 128)
        self.u4 = UpSample(128)
        self.c9 = ConvBlock(128, 64)
        self.out = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        R6 = self.c6(self.u1(R5, R4))
        R7 = self.c7(self.u2(R6, R3))
        R8 = self.c8(self.u3(R7, R2))
        R9 = self.c9(self.u4(R8, R1))
        out = self.sigmoid(self.out(R9))

        return out


if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    net = Unet()
    y = net(x)
    print(y.shape)


