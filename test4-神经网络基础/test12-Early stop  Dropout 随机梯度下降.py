# Early stop Dropout 随机梯度下降

import torch

# Early stopping
# 当验证集的损失函数值不再下降(甚至上升)时，停止训练
# 需要手动观察曲线，设置合适的阈值

# Dropout
# 在训练过程中，随机让某些(p%)神经元的权重不工作，以此来防止过拟合
net_dropout = torch.nn.Sequential(
    torch.nn.Linear(784, 200),
    torch.nn.Dropout(p=0.4),# 随机让40%的神经元权重不工作,p=1时，所有神经元权重都不工作
    torch.nn.ReLU(),
    torch.nn.Linear(200, 200),
    torch.nn.Dropout(p=0.4),# 随机让40%的神经元权重不工作
    torch.nn.ReLU(),
    torch.nn.Linear(200, 10),
)
# 训练时有dropout，测试时没有dropout
# 测试时需要将所有神经元打开
# 所以训练时需要
net_dropout.train()
# 测试时需要
net_dropout.eval()

# 随机梯度下降(Stochastic Gradient Descent)(SGD)
# 从所有样本(60000个)中随机选取(并不是完全随机，符合某些分布)一个batch(128个/16个/1个)的样本进行训练
# 根据这个batch的样本计算loss，然后根据loss进行反向传播，更新参数
# 这样可以减少计算量，加快训练速度，把所有样本都用上计算量太大
# 但是每次选取的batch样本都不一样，所以每次训练的结果都不一样
# 之前使用分batch训练的模型，其实就是随机梯度下降（SGD）（手动分batch），DataLoader中有分batch的功能








