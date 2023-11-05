# 创建张量

import torch
import numpy as np

# import from numpy
a = np.array([2, 3.3])
t1 = torch.from_numpy(a)
print(t1)

a = np.ones([2, 3])
t2 = torch.from_numpy(a)
print(t2)

# import from list
a = [2, 3.3]
t1 = torch.tensor(a)
print(t1)

# 区分tensor和Tensor(FloatTensor,DoubleTensor,LongTensor,ByteTensor)
# tensor是接受现有数据，Tensor是接收shape
# Tensor可以接收现有数据，但是要传入一个list
t2 = torch.Tensor(2, 3)
print(t2)  # 输出的是随机数tensor([[0.0000e+00, 0.0000e+00, 2.1019e-44], [0.0000e+00, 4.3258e+11, 9.7530e-43]])，默认为float32
t3 = torch.FloatTensor([[2, 3], [4, 5]])
print(t3)  # tensor([[2., 3.],[4., 5.]])
# 尽量使用tensor

# 生成未初始化张量,生成的是随机数
t = torch.empty(3, 3)  # 传入shape
print(t)
t1 = torch.FloatTensor(3, 3)  # 传入shape
# 不能直接使用未初始化的张量进行计算，因为里面的值是随机的，可能会导致计算错误(NAN,INF)

# set default type
print(torch.Tensor(2, 3).type())  # torch.FloatTensor
torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.Tensor(2, 3).type())  # torch.DoubleTensor

# 随机初始化
t = torch.rand(3, 3)  # 均匀分布[0,1]，默认为float32，shape为(3,3)
print(t)
t = torch.randn(3, 3)  # 正态分布N(0,1)，默认为float32，shape为(3,3)
print(t)
t = torch.randint(1, 10, [3, 3])  # 均匀分布[1,10)，默认为int64，shape为(3,3)
print(t)
t = torch.rand_like(t, dtype=torch.float)  # 生成和t相同shape的随机数（[0,1]均匀分布），但是类型为float
print(t)
t = torch.randint_like(t, 1, 10)  # 生成和t相同shape的随机数（[1,10)均匀分布），但是类型为int64
print(t)
t = torch.normal(mean=torch.full([10], 0.), std=torch.arange(1, 0, -0.1))  # 正态分布，生成10个值，均值为0.(注意这里的0.后面有个点)，
# 标准差为[1,0.9,0.8,0.7,0.6,0.5,0.4, 0.3,0.2,0.1]，shape为(10,)
print(t)
t = torch.normal(0, 1, size=[10])  # 正态分布，生成10个值，均值为0，方差为1，shape为(10,)
print(t)
t = t.reshape(2, 5)  # 改变形状
print(t)

# full
t = torch.full([2, 3], 7)  # 生成全为7的张量，shape为(2,3)
print(t)
t = torch.full([], 7)  # 生成全为7的标量
print(t)  # tensor(7.)
t = torch.full([1], 7)  # 生成全为7的张量，shape为(1,)
print(t)  # tensor([7.])

# arange
t = torch.arange(0, 10)  # 生产[0,10)的序列，步长为1
print(t)
t = torch.arange(1, 10, 2)  # 生成[1,10)的等差数列，步长为2，shape为(5,)
print(t)

# linspace
t = torch.linspace(0, 10, steps=10)  # 生成[0,10]的等差数列，数量为10，shape为(10,)
print(t)

# logspace
t = torch.logspace(0, -1, steps=10)  # 生成[10^0,10^-1]的等比数列，数量为10，shape为(10,)
print(t)

# ones
t = torch.ones(3, 3)  # 生成全为1的张量，shape为(3,3)
print(t)
t = torch.ones_like(t)  # 生成和t相同shape的全为1的张量
print(t)

# zeros
t = torch.zeros(3, 3)  # 生成全为0的张量，shape为(3,3)
print(t)
t = torch.zeros_like(t)  # 生成和t相同shape的全为0的张量
print(t)

# eye
t = torch.eye(3, 3)  # 生成对角线为1的方阵，shape为(3,3)
print(t)
t = torch.eye(4, 3)  # 生成对角线为1的矩阵，shape为(4,3),在(1,1)(2,2)(3,3)位置为1
print(t)

# randperm
t = torch.randperm(10)  # 生成[0,10)的随机整数排列，shape为(10,)
print(t)
# randperm的作用：用于打乱数据集，生成随机索引,然后根据索引取数据
a = torch.rand(4, 10)
print(a)
index = torch.randperm(10)
print(index)
print(a[:, index]) # 列按照index的顺序打乱
