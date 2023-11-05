# 高阶OP

import torch

# 1. torch.where(condition, x, y)
# 作用：根据condition的值，选择x和y中的元素
# condition x y 为同形状的张量
# condition为True时，选择x中的元素，否则选择y中的元素
# 返回值：同形状的张量
a = torch.arange(0, 6).view(2, 3)
print(a)  # tensor([[0, 1, 2], [3, 4, 5]])
print(torch.where(a > 3, a, torch.zeros_like(a)))  # 选择a中大于3的元素，否则选择0 返回tensor([[0, 0, 0], [0, 4, 5]])
cond = torch.tensor([[True, False, True], [False, True, True]])
print(torch.where(cond, a, torch.zeros_like(a)))  # 选择cond中为True的元素，否则选择0 返回tensor([[0, 0, 2], [0, 4, 5]])

# 2.torch.gather(input, dim, index, out=None)
# 作用：沿着指定维度dim，从input中按照index的索引值，取出元素
# input：输入张量 dim：指定维度 index：索引张量,与input的dim维度相同,index类似于topk中的index
# 返回值：和index形状相同的张量
a = torch.arange(0, 8).view(2, 2, 2)
print(a)  # tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
print(torch.gather(a, dim=1, index=torch.tensor([[[0, 1], [1, 1]], [[0, 1], [1, 1]]])))
#                                                           tensor([[[0, 3], [2, 3]], [[4, 7], [6, 7]]])
# index的具体使用和test8中topk()的index规则类似，这里dim=1,index[i][j][k]=n表示取input[i][n][k]的元素,output[i][j][k]=input[i][n][k]

# gather中的index和topk中的index规则类似
a = torch.arange(24).view(2, 3, 4).float()
values, index = a.topk(2, dim=1)
b = torch.gather(a, dim=1, index=index)
print(torch.equal(values, b))  # True
# 举例
prob = torch.rand(4, 10)
val, idx = prob.topk(dim=1, k=3)
print(val)  # tensor([[0.9811, 0.9689, 0.9645], [0.9999, 0.9998, 0.9997], [0.9999, 0.9998, 0.9997], [0.9999, 0.9998, 0.9997]])
print(idx)  # tensor([[9, 8, 7], [9, 8, 7], [9, 8, 7], [9, 8, 7]])
label = torch.arange(10)+100
print(torch.gather(label.expand(4, 10), dim=1, index=idx)) # tensor([[109, 108, 107], [109, 108, 107], [109, 108, 107], [109, 108, 107]])

# 3.detach()
# 作用：返回一个新的Tensor，从当前计算图中分离出来的，不再需要梯度
# 返回值：新的Tensor
a = torch.ones(3, 3, requires_grad=True)
b = a.detach()
print(b.requires_grad)  # False

# 4.roll(input, shifts, dims)
# 作用：将input中的元素按照指定的维度dims，向(右)后滚动shifts个位置，如果shifts为负数，则向(左)前滚动，如果滚出了数组，则从另一端重新进入
# input：输入张量 shifts：滚动的个数 dims：指定维度
# 返回值：同形状的张量
a = torch.arange(0, 9).view(3, 3)
print(a)  # tensor([[0, 1, 2],
#                   [3, 4, 5],
#                   [6, 7, 8]])
print(torch.roll(a, shifts=(1, 1), dims=(0, 1)))  # tensor([[8, 6, 7],
#                                                           [2, 0, 1],
#                                                           [5, 3, 4]])








