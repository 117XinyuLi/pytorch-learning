import torch
import torch.nn.functional as F

# 感知机(一层一输出)求导举例
x = torch.randn(1, 10)
w = torch.randn(1, 10, requires_grad=True)
b = torch.randn(1, requires_grad=True)
o = torch.sigmoid(x@w.t() + b)
y = torch.ones(1, 1)
loss = F.mse_loss(o, y)
loss.backward()
# 进行梯度下降
w = w - 0.01 * w.grad
b = b - 0.01 * b.grad
print(w)
print(b)
# 重置梯度 因为梯度是累加的
w.grad.zero_()
b.grad.zero_()
# 重复上述过程

# 一层多输出求导举例
x = torch.randn(1, 10)
w = torch.randn(2, 10, requires_grad=True)
b = torch.randn(2, requires_grad=True)
o = torch.sigmoid(x@w.t()+b) # o.shape torch.Size([1, 2])
y = torch.tensor([[1, 0.]])
loss = F.mse_loss(o, y)
loss.backward()
# 进行梯度下降
w = w - 0.01 * w.grad
b = b - 0.01 * b.grad
print(w)
print(b)
# 重置梯度 因为梯度是累加的
w.grad.zero_()
b.grad.zero_()
# 重复上述过程

# 求loss对w和b的导数，使用链式法则
# 验证链式法则
x = torch.tensor(1.)
w1 = torch.tensor(2., requires_grad=True)
b1 = torch.tensor(1.)
w2 = torch.tensor(2., requires_grad=True)
b2 = torch.tensor(1.)
y1 = x * w1 + b1
y2 = y1 * w2 + b2
dy2_dy1 = torch.autograd.grad(y2, [y1], retain_graph=True)[0]
dy1_dw1 = torch.autograd.grad(y1, [w1], retain_graph=True)[0]
dy2_dw1 = torch.autograd.grad(y2, [w1], retain_graph=True)[0]
print(dy2_dy1 * dy1_dw1)# tensor(2.)
print(dy2_dw1)# tensor(2.)


