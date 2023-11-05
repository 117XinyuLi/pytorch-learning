# 维度变换
# 所有维数都是从0开始的

import torch
import numpy as np

# 1. torch.view/reshape，这两个是一样的
a = torch.rand(4, 1, 28, 28)
print(a.shape) # torch.Size([4, 1, 28, 28])
b = a.view(4, 784) # 4张图片，每张图片28*28=784
print(b.shape) # torch.Size([4, 784])
b = a.view(4, -1) # -1表示自动计算
print(b.shape) # torch.Size([4, 784])
b = a.view(4*1, 28, 28) # 4*1张图片，每张图片28*28
print(b.shape) # torch.Size([4, 28, 28])
b = a.view(4, 28, 28, 1) # 逻辑错误，但是不会报错
print(b.shape) # torch.Size([4, 28, 28, 1])
b = a.view(4, 1, 28, 28) # 这个才是恢复了原来的维度
print(b.shape) # torch.Size([4, 1, 28, 28])

# 2. torch.squeeze/unsqueeze
a = torch.rand(4, 1, 28, 28)
print(a.shape) # torch.Size([4, 1, 28, 28])
b = a.squeeze() # 去掉维度为1的维度
print(b.shape) # torch.Size([4, 28, 28])
b = a.squeeze(1) # 第一维为1的话，去掉，否则不变
print(b.shape) # torch.Size([4, 28, 28])
b = a.unsqueeze(0) # 在第0维度增加一个维度，在原来的0维度前面增加一个维度，把原来的第0维度变成了1维度
print(b.shape) # torch.Size([1, 4, 1, 28, 28])
b = a.unsqueeze(1) # 在第1维度增加一个维度
print(b.shape) # torch.Size([4, 1, 1, 28, 28])
b = a.unsqueeze(-1) # 在倒数第1维度增加一个维度，把原来的倒数第1维度变成了1维度
print(b.shape) # torch.Size([4, 1, 28, 28, 1])

# unsqueeze举例
a = torch.rand(4, 28, 28)
b = torch.rand(28)
b = b.unsqueeze(0).unsqueeze(2)
print(b.shape) # torch.Size([1, 28, 1])
c = a + b
print(c.shape) # torch.Size([4, 28, 28])

# 3. expand/repeat
# expand:broadcasting机制，扩展维度，但是不会复制数据,要满足能够broadcasting的条件
# repeat:复制数据，扩展维度
a = torch.rand(4, 32, 14, 14)
b = torch.rand(1, 32, 1, 1)
b = b.expand(4, 32, 14, 14) # 4张图片，每张图片32通道，每个通道14*14
print(b.shape) # torch.Size([4, 32, 14, 14])
b = torch.rand(1, 32, 1, 1)
b = b.expand(-1, 32, -1, -1) # -1表示不变
print(b.shape) # torch.Size([1, 32, 1, 1])

a = torch.rand(4, 3, 32, 32)
b = a.repeat(4, 3, 1, 1) # (4, 3, 32, 32)表示第0个维度复制4次，第1个维度复制3次，第2个维度不变，第3个维度不变
print(b.shape) # torch.Size([16, 9, 32, 32])

# 4. transpose/t
a = torch.rand(4, 3)
print(a.t().shape) # torch.Size([3, 4]),t只能转置2维(dim=2)的

a = torch.rand(4, 3, 32, 32)
b = a
print(a.transpose(1, 3).shape) # torch.Size([4, 32, 32, 3]),transpose可以转置任意维度
# transpose(1, 3)表示第1个维度和第3个维度交换
# transpose会打乱原来的维度顺序，会改变原来的存储方式
a = a.transpose(1, 3).contiguous() # contiguous会把数据重新排列为连续存储
print(a.shape) # torch.Size([4, 32, 32, 3])
a = a.view(4, 3*32*32).view(4, 32, 32, 3).transpose(1, 3) # 变回原来的维度
print(a.shape) # torch.Size([4, 3, 32, 32])

print(torch.all(torch.eq(a, b))) # True

# 5. permute
a = torch.rand(4, 3, 28, 32)
b = a.permute(0, 2, 3, 1) # 现在0,1,2,3维度分别放原来的0,2,3,1维度
print(b.shape) # torch.Size([4, 28, 32, 3])
# permute也会把内存打乱，可以使用contiguous()使内存连续

# 6. tile
# torch.tile()的作用是把一个tensor沿着某个维度复制多次，比如torch.tile(a, (2, 3))，就是把a沿着第0维复制2次，沿着第1维复制3次
# 传入二维张量，指定有三个维度(a, b, c)，是把张量复制a次，每个张量沿着第0维复制b次，沿着第1维复制c次，然后把这些张量stack(dim=0)起来








