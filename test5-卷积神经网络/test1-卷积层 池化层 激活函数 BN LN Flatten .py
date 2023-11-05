# 卷积

# 图片常常归一化到0-1之间
# 图片一般有RGB三个通道，所以是三维的

# 卷积有weight sharing的特点，即卷积核的参数是共享的，所以卷积层的参数量比较少

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# 卷积层
x = torch.randn(1, 3, 32, 32)# 输入图片大小(3, 32, 32) 一张图片
layer = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)# 输入通道数(3)，输出通道数(16)，卷积核大小(3*3)，步长，填充(1表示在图片边缘填充1圈0)
y = layer(x)
print(y.shape)# 输出图片大小(16, 32, 32) 一张图片

layer = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)# 输入通道数(3)，输出通道数(16)，卷积核大小(3*3)，步长，填充(1表示在图片边缘填充1圈0)
y = layer(y)
print(y.shape)# 输出图片大小(16, 16, 16) 一张图片

w = torch.randn(10, 16, 3, 3)# 定义卷积核 输入通道数(16)，输出通道数(10)，卷积核大小(3*3)
b = torch.randn(10)# 定义偏置
y = F.conv2d(y, w, b, stride=1, padding=1)# 输入图片，卷积核，偏置，步长，填充
print(y.shape)# 输出图片大小(10, 16, 16) 一张图片

# 使用layer.weight和layer.bias来访问卷积层的参数
print(layer.weight.shape)# 返回(16, 16, 3, 3 ) 卷积核形状为(3, 3) 输入通道数(第二个16)，输出通道数(第一个16)
print(layer.bias.shape)# 偏置大小(16)
# y = F.conv2d(y, layer.weight, layer.bias, stride=1, padding=1)# 输入图片，卷积核，偏置，步长，填充

# padding_mode='reflect'表示填充的值为填充点关于边界的对称位置的像素值
# 一般padding_mode='zeros'表示填充的值为0

# 反卷积
# 反卷积的作用是将图片放大，可以用来做分割任务
# 有nn.ConvTranspose2d()和F.conv_transpose2d()两种方式
# 将图片周围padding很多，然后用卷积核进行卷积，就可以实现图片放大的效果

# 群卷积
# 群卷积是将输入通道数分成若干组，每组分别与不同的卷积核进行卷积，然后将结果拼接起来，减少参数量
# depthwise是一种特殊的群卷积，即每组只有一个输入通道，其余通道数都为0，即每个卷积核只与一个通道进行卷积
# 注意：群卷积的输入输出通道数必须能被groups整除，groups是分组数
x = torch.randn(1, 3, 32, 32)# 输入图片大小(3, 32, 32) 一张图片
layer = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3)# 输入通道数(3)，输出通道数(3)，卷积核大小(3*3)，步长，填充(1表示在图片边缘填充1圈0)，分组数
y = layer(x)
print(y.shape)# 输出图片大小(3, 32, 32) 一张图片

# 池化层
# 下采样，减少参数量，减少计算量，将图片窗口变小(但是会丢失一些信息)
# 有max pooling和average pooling

x = torch.randn(1, 3, 32, 32)# 输入图片大小(3, 32, 32) 一张图片
layer = nn.MaxPool2d(kernel_size=2, stride=2)# 卷积核大小(2*2)，步长2
y = layer(x)
print(y.shape)# 输出图片大小(3, 16, 16) 一张图片

y = F.avg_pool2d(y, kernel_size=2, stride=2)# 卷积核大小(2*2)，步长2
print(y.shape)# 输出图片大小(3, 8, 8) 一张图片
z = F.adaptive_max_pool2d(y, (1, 1))# 将图片变成(1, 1)
print(z.shape)# 输出图片大小(3, 1, 1) 一张图片

# 上采样
# 将图片窗口变大
# 将小的图片投影到大的图片，没有对应的点用差值计算出相应的值
y = F.interpolate(y, scale_factor=2, mode='nearest')# 将图片放大2倍，最近邻插值
print(y.shape)# 输出图片大小(3, 16, 16) 一张图片
y = F.interpolate(y, scale_factor=2, mode='bilinear')# 将图片放大2倍，双线性插值
print(y.shape)# 输出图片大小(3, 32, 32) 一张图片
y = F.interpolate(y, scale_factor=2, mode='bicubic')# 将图片放大2倍，双三次插值
print(y.shape)# 输出图片大小(3, 64, 64) 一张图片

# Relu层
# conv2d=>batch norm2d=>pooling=>relu(后三个的顺序可以调换)
# relu层的作用是将负数变成0，正数不变
x = torch.randn(1, 3, 32, 32)# 输入图片大小(3, 32, 32) 一张图片
layer = nn.ReLU(inplace=True)# inplace=True表示在原地修改，不会创建新的内存空间
y = layer(x)
print(y.shape)# 输出图片大小(3, 32, 32) 一张图片

y = F.relu(y, inplace=True)# inplace=True表示在原地修改，不会创建新的内存空间
print(y.shape)# 输出图片大小(3, 32, 32) 一张图片

# BatchNorm层
# 将图片的每个通道的像素值进行归一化，加快收敛速度，更加稳定，防止梯度消失
x = torch.randn(1, 3, 32, 32)# 输入图片大小(3, 32, 32) 一张图片
# 归一化
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])# 图片归一化,分别对应R,G,B三个通道
x = normalize(x)
print(x.shape)# 输出图片大小(3, 32, 32) 一张图片
# BatchNorm2d层:对某个batch，取每个通道的均值和方差，然后进行归一化接近N(0,1),然后乘以gamma加上beta(两个参数用于学习)，得到不同的分布
# 不同的batch之间的均值和方差是不一样的，均值和方程也随着训练的进行而变化
layer = nn.BatchNorm2d(3)# 归一化通道数(3)
y = layer(x)
print(y.shape)# 输出图片大小(3, 32, 32) 一张图片
print(layer.weight.shape)# gamma 参数大小(3),每个通道的gamma参数
print(layer.bias.shape)# beta 参数大小(3),每个通道的beta参数
print(vars(layer))# 查看所有参数,runtime_mean和running_var是每个通道的均值和方差,affine=True表示学习gamma和beta两个参数

x = torch.randn(100, 16, 784)# 输入图片大小(16, 784) 100张图片
layer = nn.BatchNorm1d(16)# 归一化通道数(16)
y = layer(x)
print(y.shape)# 输出图片大小(16, 784) 100张图片

# 对于BN，在训练时，需要
# model.train()
# 对于BN，在测试时，需要
# model.eval()

# LayerNorm层
# NLP Example
batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim) # pytorch中的LayerNorm层,对最后一个维度进行归一化
# Activate module
layer_norm(embedding)
# Image Example
N, C, H, W = 20, 5, 10, 10
input = torch.randn(N, C, H, W)
# Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
# as shown in the image below
layer_norm = nn.LayerNorm([C, H, W])# pytorch中的LayerNorm层,对最后三个维度进行归一化
output = layer_norm(input)
# pytorch中的都是对后面的维数做LN，channel_first的layerNorm见ConvNeXt，对[C,H,W]的C做LN

# Flatten层
# 将卷积层的输出转换成全连接层的输入
x = torch.randn(1, 3, 32, 32)# 输入图片大小(3, 32, 32) 一张图片
layer = nn.Flatten()
y = layer(x)
print(y.shape)# 输出图片大小(1, 3072) 一张图片

# Unflatten层
# 将全连接层的输出转换成卷积层的输入
x = torch.randn(1, 3, 32, 32)# 输入图片大小(3, 32, 32) 一张图片
layer = nn.Flatten()
y = layer(x)
print(y.shape)# 输出图片大小(1, 3072) 一张图片
layer = nn.Unflatten(1, (3, 32, 32))# 第一个参数表示将原张量的第几个维度展开，第二个参数表示展开后的维度
y = layer(y)
print(y.shape)# 输出图片大小(1, 3, 32, 32) 一张图片












