import torch
import torch.nn as nn
import torch.nn.functional as F
# torchvison中有预训练的模型resnet18/resnet34/resnet50/resnet101/resnet152
# 也有可以自定义的ResNet，可以自行设置block的类型和block的数量，pytorch会自动帮你构建网络

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        """
        :param ch_in: input channel
        :param ch_out: output channel
        """
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.dropout = nn.Dropout(0.5)

        self.extra = nn.Sequential()
        if ch_out != ch_in or stride != 1:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        out = nn.functional.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet18 for CIFAR10
    epoch: 20  acc: 0.87
    """
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        # followed 4 blocks
        self.blk1 = ResBlk(64, 128)
        self.blk2 = ResBlk(128, 256)
        self.blk3 = ResBlk(256, 512)
        self.blk4 = ResBlk(512, 512)

        self.outlayer = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x












