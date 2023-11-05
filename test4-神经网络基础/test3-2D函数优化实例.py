# 2D函数优化实例

import torch
import numpy as np
import matplotlib.pyplot as plt


# Himmelblau函数
# f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2 有四个极小值点，分别为(3,2),(−2.805118,3.131312),(−3.779310,-3.283186),(3.584428,-1.848126)
# 极小值为0，此函数可以用于判断优化算法性能

def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


# 画出Himmelblau函数的等高线图
x = np.arange(-6, 6, 0.1)  # x的范围
y = np.arange(-6, 6, 0.1)  # y的范围
X, Y = np.meshgrid(x, y) # 生成网格点坐标矩阵
print("X,Y maps:", X.shape, Y.shape) # (120, 120) (120, 120)
Z = himmelblau([X, Y]) # 计算每个网格点的高度值
fig = plt.figure('himmelblau') # 创建一个图形窗口
ax = fig.add_subplot(projection='3d') # 创建一个三维的绘图工程
ax.plot_surface(X, Y, Z) # 绘制三维曲面
ax.view_init(60, -30) # 设置视角
ax.set_xlabel('x') # 坐标轴
ax.set_ylabel('y') # 坐标轴
plt.show() # 显示

# 使用梯度下降法求解Himmelblau函数的最小值
x = torch.tensor([4., 0.], requires_grad=True) # 定义初始值
optimizer = torch.optim.Adam([x], lr=1e-3) # 定义优化器，使用Adam优化x，学习率为1e-3
for step in range(20000):
    pred = himmelblau(x) # 计算x的值
    pred.backward() # 反向传播
    optimizer.step() # 更新参数
    optimizer.zero_grad() # 清空梯度
    if step % 2000 == 0:
        print('step {}: x = {}, f(x) = {}'.format(step, x.tolist(), pred.item()))










