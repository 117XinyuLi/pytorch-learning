# optimizer performance
# 1. initialization status
# 2. learning rate (learning rate decay)
# 3. momentum (考虑过去的梯度)
# 4. etc.

import torch
from torch.nn import functional as F

# activation function 输出的形状与输入的形状一致
# 1. sigmoid y = 1 / (1 + exp(-x))
# 导数 y * (1 - y)
# 缺陷：梯度消失，sigmoid函数的导数在x很大或很小时，导数值很小，导致梯度消失，无法更新参数
a = torch.linspace(-100, 100, 10)
print(torch.sigmoid(a))# tensor([0.0000e+00, 1.6655e-34, 7.4564e-25, 3.3382e-15, 1.4945e-05, 9.9999e-01,
#                        1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00])
# print(F.sigmoid(a))# tensor([0.0000e+00, 1.6655e-34, 7.4564e-25, 3.3382e-15, 1.4945e-05, 9.9999e-01,
#                        1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00])
#                        F.sigmoid() is deprecated. Use torch.sigmoid instead.

# 2. tanh y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
# tanh(x) = 2sigmoid(2x) - 1
# 导数 1 - y^2
# RNN中使用tanh激活函数
a = torch.linspace(-1, 1, 10)
print(torch.tanh(a))# tensor([-0.7616, -0.6514, -0.5047, -0.3215, -0.1107,  0.1107,  0.3215,  0.5047,
#                        0.6514,  0.7616])

# 3. relu y = max(0, x)
# 导数 0 if x < 0 else 1
# 导数计算简单，计算速度快，不容易出现梯度消失或梯度爆炸
a = torch.linspace(-1, 1, 10)
print(torch.relu(a))# tensor([0.0000, 0.1111, 0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778, 0.8889,1.0000])
print(F.relu(a))# tensor([0.0000, 0.1111, 0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778, 0.8889,1.0000])

# Loss 及 Loss的梯度
# 1. MSE mean((y_hat - y)**2)
y = torch.randint(0, 2, [3]).float()
y_pred = torch.rand(3)
MSE1 = F.mse_loss(y_pred, y) # 参数，mes_loss(y_hat, y)
print(MSE1)# tensor(0.2502)
MSE2 = torch.mean((y_pred - y)**2)
print(MSE2)# tensor(0.2502)
print(MSE1 == MSE2)# tensor(True)

# 自动求导并梯度下降
# 1. 创建变量并设置requires_grad=True
# 2. 计算loss
# 3. torch.autograd.grad(loss, [w1, w2, w3, b1, b2, b3]) 计算loss对w1, w2, w3, b1, b2, b3的梯度 返回的是一个元组可用[0]~[n]取出
#    或者 loss.backward()
#    计算loss对所有requires_grad=True的变量的梯度，用于求梯度或者反向传播的一定是标量，
#    想要多次求导，需要设置retain_graph=True，否则需要重新计算图
# 4. 更新参数 w1 = w1 - lr * w1.grad / optimizer.step()
# 5. 清空梯度 w1.grad.zero_() / optimizer.zero_grad() 这一步也可以放在计算loss之后,backward()之前,但一定要有
# 6. 重复2~5
# 以上顺序不能变，否则会报错

# 使用autograd计算梯度
x = torch.ones(1)
w = torch.full([1], 2.)
w.requires_grad_()
# 上两行代码等价于 w = torch.full([1], 2., requires_grad=True)
mse = F.mse_loss(x*w, torch.ones(1))# tensor(1.)
dw = torch.autograd.grad(mse, [w])[0]# (tensor([2.]),)
print(dw)# tensor([2.])

# 也可以使用backward()函数
x = torch.ones(1)
w = torch.full([1], 2., requires_grad=True)
mse = F.mse_loss(x*w, torch.ones(1))# tensor(1.)
mse.backward()
dw = w.grad
print(dw)# tensor([2.])
print(dw.norm(2)) # tensor(2.)

# 2. softmax
# softmax(x) = exp(x) / sum(exp(x))
a = torch.rand(3, requires_grad=True)
p = F.softmax(a, dim=0) # tensor([0.2647, 0.3743, 0.3609], grad_fn=<SoftmaxBackward>)
p[1].backward(retain_graph=True) # retain_graph=True 保留计算图,否则无法再次backward, 用于backward或者计算梯度的一定是标量
da1 = a.grad
da2 = torch.autograd.grad(p[1], [a], retain_graph=True)[0]
print(da1) # tensor([-0.1144,  0.2488, -0.1344]) 对p[1]求各个参数(a[0]~a[2])的导数,对a[1]的导数为正,其余为负
print(da2) # tensor([-0.1144,  0.2488, -0.1344])

# 3.
# cross entropy
# cross_entropy = -sum(y * log(y_hat))/m
# H(p, q) = -sum(p * log(q)) = H(p) + D_KL(p||q) # H(p)为熵 D_KL(p||q)为KL散度 优化cross_entropy等价于优化KL散度,使其接近于0，
#                                                                                                 即真实和预测两个分布接近
# 交叉熵损失函数，一般用于分类问题，pytorch中的交叉熵损失函数包含了softmax操作，所以不需要再对输出进行softmax操作
# 交叉熵损失函数的梯度计算
# torch 中的cross_entropy(y_hat, y) = softmax(得到y_hat) => log(得到log_y_hat) => -sum(y_onehot * log_y_hat)/m(得到loss)

# 以下为对2个样本的交叉熵损失函数的计算
a = torch.rand(2, 3)
y = torch.tensor([2, 1], dtype=torch.long) # 不用one-hot,直接用类别标签标出第几类，注意是long类型
loss1 = F.cross_entropy(a, y) # 参数cross_entropy(y_hat, y) y_hat的shape为(m, n)，m为样本数，n为类别数
#                               y的shape为(m,)不用转为one-hot,直接表明第几类(0~n-1) 交叉熵损失函数的shape为(1,)
y_onehot = torch.tensor([[0, 0, 1], [0, 1, 0]], dtype=torch.float) # y的one-hot编码，用于计算loss2
pred = F.softmax(a, dim=1) # 预测值
pred_log = torch.log(pred) # 预测值的log
loss2 = -torch.sum(y_onehot * pred_log)/a.shape[0] # 交叉熵损失函数 = -sum(y_onehot * log_y_hat))/m
loss3 = F.nll_loss(pred_log, y) # cross_entropy(y_hat，y) = softmax(得到y_hat) => log(得到log_y_hat) => null_loss(输入log_y_hat, y)
#                                 null_loss(log_y_hat, y) = -sum(y_onehot * log_y_hat)/m
print(loss1)
print(loss2)
print(loss3)

# binary cross entropy
# binary_cross_entropy(y_hat, y) = -sum(y * log(y_hat) + (1-y) * log(1-y_hat))/m
# 二分类交叉熵损失函数

# 以下为对2个样本的二分类交叉熵损失函数的计算
x = torch.rand(2)
y = torch.tensor([1, 0], dtype=torch.float)# 二分类标签，注意是float类型，shape为(m,)
loss1 = F.binary_cross_entropy(x, y) # 参数binary_cross_entropy(y_hat, y) y_hat的shape为(m,)，m为样本数，y的shape为(m,)
loss2 = -torch.sum(y * torch.log(x) + (1-y) * torch.log(1-x))/x.shape[0] # 二分类交叉熵损失函数 = -sum(y * log(y_hat) + (1-y) * log(1-y_hat))/m
print(loss1)
print(loss2)

# BCEWithLogitsLoss（带sigmoid的二分类交叉熵损失函数）
# BCEWithLogitsLoss = sigmoid(得到y_hat) => binary_cross_entropy(得到loss)
# 二分类交叉熵损失函数的梯度计算
x = torch.rand(2)
y = torch.tensor([1, 0], dtype=torch.float)
loss1 = F.binary_cross_entropy_with_logits(x, y) # 参数BCEWithLogitsLoss(y_hat, y) y_hat的shape为(m,)，m为样本数，y的shape为(m,)
loss2 = F.binary_cross_entropy(torch.sigmoid(x), y) # 参数binary_cross_entropy(y_hat, y) y_hat的shape为(m,)，m为样本数，y的shape为(m,)
print(loss1)
print(loss2)
# BCE也可以用于在[0,1]之间拟合问题，e.g.设定y=0.3, y_hat会趋近于0.3

# 以上的loss函数可以在torch.nn中找到，如nn.CrossEntropyLoss() nn.BCELoss() nn.BCEWithLogitsLoss()等

# 4. 自定义损失函数
# 直接def一个函数即可，注意里面所有的参数都是tensor类型，不要用numpy类型或者python类型
# 如果有需要优化的参数，需要在函数中加入requires_grad=True，如w = torch.tensor(1.0, requires_grad=True)然后在backward()中传入w
# 或者用nn.Parameter()包装一下，如w = nn.Parameter(torch.tensor(1.0))，这样就不用在backward()中传入w了，nn.Parameter()中的参数默认requires_grad=True

# 4.KL散度
# KL散度的计算
# c = nn.KLDivLoss(reduction='batchmean')
# loss = c(input, target)
# input需要取log，input和target需要提前softmax(dim=-1)
# 即loss = c(F.log_softmax(y_hat, dim=-1), F.softmax(y, dim=-1))
