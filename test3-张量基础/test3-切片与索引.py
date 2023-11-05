# 切片与索引

import torch
import numpy as np

# indexing
a = torch.rand(4, 3, 28, 28)  # 4张图片，3个通道，28*28像素
print(a[0].shape)  # torch.Size([3, 28, 28])
print(a[0, 0].shape)  # torch.Size([28, 28])
print(a[0, 0, 0].shape)  # torch.Size([28])  一维
print(a[0, 0, 0, 0].shape)  # torch.Size([]) 0维张量

print(a[:2].shape)  # torch.Size([2, 3, 28, 28]) 前两个 [:2]=>[0,2)
print(a[1:].shape)  # torch.Size([3, 3, 28, 28]) 从第二个开始 [1:]=>[1,4)
print(a[1:2].shape)  # torch.Size([1, 3, 28, 28]) 第二个 第[1,2)个
print(a[1, 1:3].shape)  # torch.Size([2, 28, 28]) 第二个通道的第[1,3)个
print(a[:2, :1, :, :].shape)  # torch.Size([2, 1, 28, 28]) 前两个，前一个通道，所有行列
print(a[:2, -1, :, :].shape)  # torch.Size([2, 28, 28]) 前两个，最后一个通道，所有行列

print(a[:, :, 0:28:2, 0:28:2].shape)  # torch.Size([4, 3, 14, 14]) 间隔取值 从0到28,[0,28)，步长为2，取14个
print(a[:, :, ::2, ::2].shape)  # torch.Size([4, 3, 14, 14]) 间隔取值 从0到28,[0,28)，步长为2，取14个
# 用:索引的维度会被保留，用数字索引的维度会被去掉

b = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
label = torch.tensor([0, 1, 2])
index = label == 1
print(b[index].shape)  # torch.Size([1, 3]) 选出label为1的行
index = torch.LongTensor([0, 2])
print(b[index].shape)  # torch.Size([2, 3]) 选出第0行和第2行

# index_select
print(a.index_select(0, torch.tensor([0, 2])).shape)  # 第一个参数是维度，第二个参数是索引值（传入tensor） torch.Size([2, 3, 28, 28]) 选取第0，2个
print(a.index_select(2, torch.arange(28)).shape)  # torch.Size([4, 3, 28, 28]) 选取第0-27个通道

# ...
print(a[...].shape)  # torch.Size([4, 3, 28, 28]) 省略号，表示所有维度
print(a[0, ...].shape)  # torch.Size([3, 28, 28]) 省略号，表示所有维度
print(a[..., 0].shape)  # torch.Size([4, 3, 28]) 省略号，表示所有维度

# mask
mask = a.ge(0.5)  # 大于等于0.5的为True，小于0.5的为False
print(mask.shape)  # torch.Size([4, 3, 28, 28])
print(a.masked_select(mask).shape)  # torch.Size([x]) 选取大于等于0.5的值, x为True的个数, 会把所有维度展开成一维
print(torch.masked_select(a, mask).shape)  # torch.Size([x]) 选取大于等于0.5的值, x为True的个数,会默认展平 1维

# take
print(a.shape)  # torch.Size([4, 3, 28, 28])
print(a.take(torch.tensor([0, 2, 28, 30])).shape)  # torch.Size([4]) 先展平，再取值，选取第0，2，28，30个元素
print(torch.take(a, torch.tensor([0, 2, 28, 30])).shape)  # torch.Size([4]) 先展平，再取值，选取第0，2，28，30个元素


