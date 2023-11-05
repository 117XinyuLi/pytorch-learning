import torch

# 梯度爆炸
# 出现loss突然变大，或者nan的情况，说明梯度爆炸
# 使用clipping,将梯度的范围限制在一个范围内(将梯度的模长限制在一个范围内，梯度的方向不变)
# 一般使用torch.nn.utils.clip_grad_norm_()函数
model = torch.nn.Linear(2, 2)
for p in model.parameters():
    torch.nn.utils.clip_grad_norm_(p, max_norm=1.0)# 将梯度的模长限制在1.0以内

# 梯度消失
# 使用LSTM
