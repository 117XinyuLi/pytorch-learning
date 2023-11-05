# 动量与学习率衰减

import torch

# 1.Momentum(动量)
# 考虑之前的梯度，当前梯度的加权平均
# 可以加快收敛速度，可能可以跳过鞍点，打破局部最优
# β：考虑之前的梯度的比例，β越大，考虑之前的梯度越多，一般为0.9
a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
optimizer = torch.optim.SGD([a], lr=0.01, momentum=0.9)  # 动量为0.9
optimizer = torch.optim.Adam([a], lr=0.01, betas=(0.9, 0.99))  # 0.9为动量的β(β1)，0.99为RMSProp的β(β2)


# 2.Learning Rate Decay(学习率衰减)
# 一般在训练过程中，学习率会逐渐减小，使得模型收敛更加稳定
# 方法1
def train():
    pass
optimizer = torch.optim.SGD([a], lr=0.01, momentum=0.9, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
# factor：衰减因子，每次衰减的比例 patience：容忍的epoch数，即在patience个epoch内，如果loss没有下降，则衰减学习率 verbose：是否打印信息
# 当loss不再下降时，学习率减小
epochs = 10
for epoch in range(epochs):
    loss = train()
    # val_loss = validate()
    scheduler.step(loss)# 可以监听loss或者val_loss，建议监听val_loss，
    # 每个epoch结束后，调用scheduler.step(val_loss)，不要在每个batch后调用，防止导致学习率变化太快


# 方法2
def train():
    pass
optimizer = torch.optim.SGD([a], lr=0.01, momentum=0.9, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 每10个epoch，学习率乘以0.1
epochs = 100
for epoch in range(epochs):
    loss = train()
    # val_loss = validate()
    scheduler.step()
