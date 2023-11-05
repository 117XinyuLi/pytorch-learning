# 正则化

import torch


# Occam's Razor
# More things should be not be used than are necessary.

# reduce overfitting
# 1.More data
# 2.constraint model complexity
#   shallow model
#   regularization
# 3.dropout
# 4.data augmentation
# 5.early stopping

# Regularization(weight decay)
# 迫使某些权重变小，从而降低模型复杂度

# L1正则化：权重向量的L1范数作为正则化项
# L1会将权重向量稀疏化，即部分元素为0，可用于特征选择
# pytorch中需要手动实现
def l1_regularization_loss(y_hat, y, l1_lambda, net, criterion):
    l1 = 0
    for param in net.parameters():
        l1 += torch.sum(torch.abs(param))
    return criterion(y_hat, y) + l1_lambda * l1 / (2 * len(y))


'''
y_hat = net(x)
loss = l1_regularization_loss(y_hat, y, l1_lambda, net, criterion)
optimizer.zero_grad()
loss.backward()
optimizer.step()
'''

# L2正则化：权重向量的L2范数作为正则化项
# L2会将某些权重向量变得更小，但不会使其为0
a = torch.ones(3, requires_grad=True)
optimizer = torch.optim.SGD([a], lr=0.1, weight_decay=0.01)  # weight_decay就是L2正则化的系数(lambda)
#                                                           当数据数量变化时，可以调整lambda
