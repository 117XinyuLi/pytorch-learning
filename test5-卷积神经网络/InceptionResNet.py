import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, dropout=0.2):
        """
        :param ch_in: input channel
        :param ch_out: output channel
        """
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.dropout = nn.Dropout(dropout)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        del x
        out = nn.functional.relu(out, inplace=True)
        return out


class IncResBlk(nn.Module):
    def __init__(self, ch_in, ch_out_1, ch_out_2, ch_out_3, ch_out4, dropout):
        """
        :param ch_in: input channel
        :param ch_out1: output channel of conv1
        :param ch_out2: output channel of conv2
        :param ch_out3: output channel of conv3
        :param ch_out4: output channel of conv4
        :param dropout: dropout rate
        """
        super(IncResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out_1, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(ch_out_1)

        mid2 = (ch_out_2 + ch_in) // 2
        self.conv2_1 = nn.Conv2d(ch_in, mid2, kernel_size=1, stride=1, padding=0)
        self.bn2_1 = nn.BatchNorm2d(mid2)
        self.conv2_2 = nn.Conv2d(mid2, ch_out_2, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(ch_out_2)

        mid3 = (ch_out_3 + ch_in) // 2
        self.conv3_1 = nn.Conv2d(ch_in, mid3, kernel_size=1, stride=1, padding=0)
        self.bn3_1 = nn.BatchNorm2d(mid3)
        self.conv3_2 = nn.Conv2d(mid3, ch_out_3, kernel_size=5, stride=1, padding=2)
        self.bn3_2 = nn.BatchNorm2d(ch_out_3)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(ch_in, ch_out4, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(ch_out4)

        self.dropout = nn.Dropout(dropout)

        self.extra = nn.Sequential()
        if ch_in != (ch_out_1 + ch_out_2 + ch_out_3 + ch_out4):
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out_1 + ch_out_2 + ch_out_3 + ch_out4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ch_out_1 + ch_out_2 + ch_out_3 + ch_out4)
            )

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out2 = F.relu(self.bn2_2(self.conv2_2(F.relu(self.bn2_1(self.conv2_1(x))))), inplace=True)
        out3 = F.relu(self.bn3_2(self.conv3_2(F.relu(self.bn3_1(self.conv3_1(x))))), inplace=True)
        out4 = F.relu(self.bn4(self.conv4(self.maxpool(x))), inplace=True)
        out = torch.concat([out1, out2, out3, out4], dim=1)
        del out1, out2, out3, out4
        out = self.extra(x) + out
        del x
        out = self.dropout(out)
        return out

class IncResNet(nn.Module):
    """
    InceptionResNet for CIFAR10
    epoch 150 test acc: 0.913
    """
    def __init__(self):
        super(IncResNet, self).__init__()
        self.res1 = ResBlk(3, 32, 0)

        self.blk1 = IncResBlk(32, 28, 28, 28, 28, 0)
        self.blk2 = IncResBlk(112, 64, 64, 64, 64, 0)
        self.blk3 = IncResBlk(256, 72, 72, 72, 72, 0)
        self.blk4 = IncResBlk(288, 128, 128, 128, 128, 0)

        self.res2 = ResBlk(512, 512, 0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(512*16*16, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.res1(x)
        out = self.blk1(out)
        out = self.blk2(out)
        out = self.blk3(out)
        out = self.blk4(out)
        out = self.res2(out)
        out = self.maxpool(out)
        out = self.flatten(out)
        out = self.dropout(out)
        out = F.relu(self.fc_bn1(self.fc1(out)), inplace=True)
        out = self.fc2(out)

        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IncResNet().to(device)
    x = torch.randn(2, 3, 32, 32).to(device)
    y = model(x)
    print(y.shape)

