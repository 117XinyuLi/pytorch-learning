import torch
from torch import nn

class LeNet5(nn.Module):
    """
    LeNet5 for CIFAR10 在原LeNet5的基础上，加了BN，加了激活函数，加了dropout，改了参数，加深了网络
    epoch: 50 test acc: 0.85
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv_unit = nn.Sequential(
            # X: [b, 3, 32, 32] => [b, 96, 16, 16]
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),# 原LeNet5没有BN
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),# 原LeNet5是隔断采样，这里用maxpooling代替
            nn.ReLU(inplace=True),# 原LeNet5没有激活函数
            nn.Dropout(0.5),# 原LeNet5没有dropout
            # X: [b, 96, 16, 16] => [b, 256, 8, 8]
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # X: [b, 256, 8, 8] => [b, 512, 4, 4] 原LeNet5没有以下卷积层
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # X: [b, 512, 4, 4] => [b, 1024, 2, 2]
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.fc_unit = nn.Sequential(
            nn.Linear(1024*2*2, 2048), # 这里要是不确定1024*2*2的值，可以用随便一个数据来跑一下，看看输出的shape
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 10)
        )

    def forward(self,x):# 调用网络的时候，会自动调用forward函数，不要手动调用forward，调用网络的时候会调用除了forward以外的其他内容，比如初始化，手动调用forward的话，只会调用forward
        """
        :param x: [b, 3, 32, 32]
        :return: [b, 10]
        """
        batchsz = x.size(0)
        # [b, 3, 32, 32] => [b, 1024, 2, 2]
        x = self.conv_unit(x)
        # [b, 1024, 2, 2] => [b, 1024*2*2]
        x = x.view(batchsz, 1024*2*2) # Flatten
        # [b, 1024*2*2] => [b, 10]
        logits = self.fc_unit(x)

        return logits




