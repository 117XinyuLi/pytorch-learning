# Broadcast: expand without copying data

import torch

# 1.Insert 1 dim ahead (在第一维前面插入1维扩张，直到两张量dim相同)
# 2.Expand dims with size 1 to same size (将维度为1(仅能将1维的拓展，其他的不能拓展，不能Broadcast)的扩展到相同的维度,
#                                         使张量形状相同, 但是不会复制数据)
# 3.否则不能Broadcast

x = torch.randn(4, 32, 14, 14)
bias = torch.randn(32)
print(x.size(), bias.size())
bias = bias.view(32, 1, 1)  # 让bias的维度变成(32,1,1)，bias的32对齐x的32
# bias:[32,1,1]=(在高维度上进行扩张)=>[1,32,1,1]=(拓展)=>[4,32,14,14],与x相加
x = x + bias

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
bias = torch.tensor([1, 2, 3])
print(x)
print(bias.size())
print(x + bias)# bias:[3]=(在高维度上进行扩张)=>[1,3]=(拓展)=>[2,3]，与x相加

x = torch.tensor([1, 2, 3, 4])
y = torch.tensor([[1], [2], [3]])
print(x.size(), y.size()) # x:[4],y:[3,1]
print(x + y)# x:[4]=(在高维度上进行扩张)=>[1,4]=(拓展)=>[3,4], y:[3,1]=(拓展)=>[3,4], 再相加

x = torch.randn(4, 32, 8)
y = torch.tensor([5.0])
# 不用broadcast，更加复杂
z1 = y.unsqueeze(0).unsqueeze(0).expand_as(x) + x
# 使用broadcast，更加简单
z2 = y + x
print(torch.all(torch.eq(z1, z2)))# True

y2 = torch.full([4, 32, 8], 5.0)# 不用broadcast，需要占用更多的内存(相对于y)
z3 = y2 + x
print(torch.all(torch.eq(z1, z3)))# True

# A：[4,32,8] B：[4], A+B不能Broadcast(B扩张为[1,1,4]后,不知道怎么拓展到A的维度)



