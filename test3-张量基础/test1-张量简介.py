# 主要用torch.FloatTensor、torch.IntTensor、torch.ByteTensor（用于存储bool值，torch.ByteTensor已经被弃用，使用torch.BoolTensor代替）来创建张量
# GPU上张量有torch.cuda.FloatTensor等

import torch
import numpy as np

a = torch.randn(2, 3)  # 创建一个2*3的张量(满足N(0,1)的正态分布),默认为FloatTensor
print(a.type())  # 查看张量类型,返回torch.FloatTensor
print(type(a))  # 查看张量类型，返回<class 'torch.Tensor'>

print(isinstance(a, torch.FloatTensor))  # 判断张量类型,返回True
print(isinstance(a, torch.Tensor))  # 判断张量类型,返回True
print(isinstance(a, torch.IntTensor))  # 判断张量类型,返回False

# 使用cuda
a = a.cuda()  # 将张量转换为GPU上的张量
print(a.type())  # 查看张量类型,返回torch.cuda.FloatTensor
print(type(a))  # 查看张量类型，返回<class 'torch.Tensor'>
print(isinstance(a, torch.FloatTensor))  # 判断张量类型,返回False
print(isinstance(a, torch.cuda.FloatTensor))  # 判断张量类型,返回True

# dim size/shape的区分
# dim是shape/size的长度，shape/size是一个张量的大小

a = torch.ones(2, 3, 4)  # 创建一个2*3*4的张量
print(a.dim())  # 查看张量维度,返回3 x.ndim = 3
print(a.shape)  # 返回torch.Size([2, 3, 4])
print(a.size)  # 返回<built-in method size of Tensor object at 0x000001C74DD7B9F0>,即返回一个函数
print(a.size())  # 返回torch.Size([2, 3, 4])
print(a.size(0))  # 返回2
print(a.size(1))  # 返回3
print(a.size(2))  # 返回4
print(a.shape[0])  # 返回2
print(a.shape[1])  # 返回3
print(a.shape[2])  # 返回4
print(list(a.shape))  # 返回[2, 3, 4]

a = torch.rand(2, 3, 4)  # 创建一个2*3*4的张量,随机初始化(0-1之间的均匀分布)

# 三维张量在RNN中常用，如[batch_size, seq_len, embedding_dim]
# 四维张量在CNN中常用，如[batch_size, channel, height, width]

print(a.numel())  # 返回24，即2*3*4

# 标量
a = torch.tensor(1.0)  # 创建一个标量，维度为0
print(a)  # 返回tensor(1.)
print(a.dim())  # 返回0 x.ndim = 0
print(a.shape)  # 查看张量维度,返回torch.Size([])
print(len(a.shape))  # 查看张量维度,返回0

# 向量/张量
a = torch.tensor([1.0])  # 创建一个向量
print(a)  # 返回tensor([1.])
print(a.dim())  # 返回1
print(a.type())  # 查看张量类型,返回torch.FloatTensor
print(a.shape)  # 查看张量维度,返回torch.Size([1])

b = torch.FloatTensor([1.0])  # 创建一个向量
print(b)  # 返回tensor([1.])
c = torch.FloatTensor(1)  # 创建一个向量，随机初始化
print(c)  # tensor([-139052.2500])
d = torch.FloatTensor(2)  # 创建一个向量
print(d)  # 返回tensor([3.5873e-43, 0.0000e+00])
e = torch.FloatTensor(2, 3)  # 创建一个2*3的张量
print(e)  # tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00],[0.0000e+00, 1.2273e+12, 6.3759e-43]])

# 区分
a = torch.tensor([1.0])  # 创建一个向量
print(a.shape)  # 返回torch.Size([1])
a = torch.tensor([[1.0]])  # 创建一个1*1的张量
print(a.shape)  # 返回torch.Size([1, 1])
a = torch.FloatTensor(1) # 创建一个向量
print(a.shape)  # 返回torch.Size([1])
print(a.dim())  # 返回1 x.ndim = 1
a = torch.FloatTensor(1, 1)  # 创建一个1*1的张量
print(a.shape)  # 返回torch.Size([1, 1])
print(a.dim())  # 返回2 x.ndim = 2

data = np.ones(2)  # 创建一个向量
a = torch.from_numpy(data)  # 将numpy数组转换为张量
print(a)  # 返回tensor([1., 1.], dtype=torch.float64)

# 从tensor转换为numpy
a = torch.ones(2)  # 创建一个向量
b = a.numpy()  # 将张量转换为numpy数组
print(b)  # 返回[1. 1.]
# 若a在GPU上，需要先将a转换为CPU上的张量，若a需要梯度，则需要先将a.detach()，即a = a.detach().cpu()
b = a.detach().cpu().numpy()  # 将张量转换为numpy数组
print(b)  # 返回[1. 1.]

# 维数计算
# (2,) + (2, 3) = (2, 2, 3)
# (2,) + 3 * (1,) = (2, 1, 1, 1)

