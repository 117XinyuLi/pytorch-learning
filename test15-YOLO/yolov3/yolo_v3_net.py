import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualLayer, self).__init__()
        self.conv = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        return self.conv(x)+x


class ConvolutionalSet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSet, self).__init__()
        self.conv = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        return self.conv(x)


class DownSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleLayer, self).__init__()
        self.layer = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 3, 2, 1),
        )

    def forward(self, x):
        return self.layer(x)


class UpSampleLayer(nn.Module):
    def __init__(self):
        super(UpSampleLayer, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')


class YoloV3Net(nn.Module):
    def __init__(self, num_classes):
        super(YoloV3Net, self).__init__()
        self.trunk_52 = nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            DownSampleLayer(32, 64),

            ResidualLayer(64, 32),

            DownSampleLayer(64, 128),

            ResidualLayer(128, 64),
            ResidualLayer(128, 64),

            DownSampleLayer(128, 256),

            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
            ResidualLayer(256, 128),
        )

        self.trunk_26 = nn.Sequential(
            DownSampleLayer(256, 512),

            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
            ResidualLayer(512, 256),
        )

        self.trunk_13 = nn.Sequential(
            DownSampleLayer(512, 1024),

            ResidualLayer(1024, 512),
            ResidualLayer(1024, 512),
            ResidualLayer(1024, 512),
            ResidualLayer(1024, 512),
        )

        self.convset_13 = nn.Sequential(
            ConvolutionalSet(1024, 512),
        )

        self.detection_13 = nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 3*(5+num_classes), 1, 1, 0) #  3 anchors * (5 bbox params + num_classes)
        )

        self.up_13_to_26 = nn.Sequential(
            ConvolutionalLayer(512, 256, 3, 1, 1),
            UpSampleLayer(),
        )

        self.convset_26 = nn.Sequential(
            ConvolutionalSet(768, 256),
        )

        self.detection_26 = nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),
            nn.Conv2d(512, 3*(5+num_classes), 1, 1, 0) #  3 anchors * (5 bbox params + num_classes)
        )

        self.up_26_to_52 = nn.Sequential(
            ConvolutionalLayer(256, 128, 3, 1, 1),
            UpSampleLayer(),
        )

        self.convset_52 = nn.Sequential(
            ConvolutionalSet(384, 128),
        )

        self.detection_52 = nn.Sequential(
            ConvolutionalLayer(128, 256, 3, 1, 1),
            nn.Conv2d(256, 3*(5+num_classes), 1, 1, 0) #  3 anchors * (5 bbox params + num_classes)
        )

    def forward(self, x):
        x_52 = self.trunk_52(x)
        x_26 = self.trunk_26(x_52)
        x_13 = self.trunk_13(x_26)

        convset_13_out = self.convset_13(x_13)
        detection_13 = self.detection_13(convset_13_out)
        up_13_to_26 = self.up_13_to_26(convset_13_out)
        cat_13_26 = torch.cat([up_13_to_26, x_26], dim=1)

        convset_26_out = self.convset_26(cat_13_26)
        detection_26 = self.detection_26(convset_26_out)
        up_26_to_52 = self.up_26_to_52(convset_26_out)
        cat_26_52 = torch.cat([up_26_to_52, x_52], dim=1)

        convset_52_out = self.convset_52(cat_26_52)
        detection_52 = self.detection_52(convset_52_out)

        return detection_13, detection_26, detection_52
    # [batch_size, 3*(5+num_classes), 13, 13] 大目标
    # [batch_size, 3*(5+num_classes), 26, 26] 中目标
    # [batch_size, 3*(5+num_classes), 52, 52] 小目标


if __name__ == '__main__':
    net = YoloV3Net(10)
    x = torch.randn(1, 3, 416, 416)
    y = net(x)
    print(y[0].shape, y[1].shape, y[2].shape)

