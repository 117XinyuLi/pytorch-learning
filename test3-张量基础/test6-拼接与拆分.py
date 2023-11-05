# 拼接与拆分

import torch

# 1.Cat (只有一个维度不同，其他维度相同)
a = torch.rand(4, 32, 8)
b = torch.rand(5, 32, 8)
print(torch.cat([a, b], dim=0).shape)  # torch.Size([9, 32, 8])，dim=0表示在第0维拼接，cat的第一个参数是一个list，里面是要拼接的tensor

# 2.Stack (所有维度必须相同)
a = torch.rand(4, 32, 8)
b = torch.rand(4, 32, 8)
print(torch.stack([a, b], dim=0).shape)  # torch.Size([2, 4, 32, 8])，dim=0表示在第0维生成新的维度，stack的第一个参数是一个list，里面是要拼接的tensor

a = torch.rand(4, 3, 16, 32)
b = torch.rand(4, 3, 16, 32)
print(torch.stack([a, b], dim=1).shape)  # torch.Size([4, 2, 3, 16, 32])，dim=1表示在第1维生成新的维度

# 3.Split (通过指定维度和长度进行拆分)
a = torch.rand(4, 32, 8)
aa, bb = a.split([3, 1], dim=0)  # 指定在第0维拆分，长度分别为3和1
print(aa.shape, bb.shape)  # torch.Size([3, 32, 8]) torch.Size([1, 32, 8])
cc, dd = torch.split(a, 2, dim=0)  # 指定在第0维拆分，长度为2
print(cc.shape, dd.shape)  # torch.Size([2, 32, 8]) torch.Size([2, 32, 8])

# 4.Chunk (通过指定维度和数量进行拆分,数量必须能整除,平均拆分)
a = torch.rand(4, 32, 8)
aa, bb, cc, dd = a.chunk(4, dim=0)  # 指定在第0维拆分，拆分为4份
print(aa.shape, bb.shape, cc.shape, dd.shape)  # torch.Size([1, 32, 8]) torch.Size([1, 32, 8]) torch.Size([1, 32, 8]) torch.Size([1, 32, 8])
ee, ff = torch.chunk(a, 2, dim=0)  # 指定在第0维拆分，拆分为2份
print(ee.shape, ff.shape)  # torch.Size([2, 32, 8]) torch.Size([2, 32, 8])
# gg, hh, ii = a.chunk(3, dim=0)  # 报错，数量必须能整除

# 5. tile
# torch.tile()的作用是把一个tensor沿着某个维度复制多次，比如torch.tile(a, (2, 3))，就是把a沿着第0维复制2次，沿着第1维复制3次
# 传入二维张量，指定有三个维度(a, b, c)，是把张量复制a次，每个张量沿着第0维复制b次，沿着第1维复制c次，然后把这些张量stack(dim=0)起来













