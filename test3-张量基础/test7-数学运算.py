# 数学运算

# 注：F(a,dim=n)表示对a的第n维进行F操作，dim=1表示对行操作，dim=0表示对列操作，dim=-1表示对最后一维操作
#    具体为对a[i,j,k,……,:(第n维),l,m,……]进行F操作,除了第n维为:，其他维数为具体的值,输出的维数为a去掉第n维的维数
#                                           输出数据a'[i,j,k,……,l,m,……]=F(a[i,j,k,……,:(第n维),l,m,……]),
#                                           维数会减少1，即去掉第n维，若使用keepdim=True，则维数不变，第n维维数变为1
# 比如：
# a = torch.tensor([[1,2,3],[4,5,6]])
# print(a.sum(dim=1)) # 对行操作 [6,15] 具体为对a[0,:]和a[1,:]进行操作,a：2*3=>a'：2*1,a'[0]=sum(a[0,:])=6,a'[1]=sum(a[1,:])=15

# torch.all(torch.eq(a,b)) # 判断a和b是否相等(全部相等),all()判断一个tensor中的所有元素是否都为True,是返回True,否返回False

import torch

# 1. 基本运算
a = torch.rand(3, 4)
b = torch.rand(4)
print(a+b)# broadcast
print(torch.all(torch.eq(a+b, torch.add(a, b))))# True
print(torch.all(torch.eq(a-b, torch.sub(a, b))))# True
print(torch.all(torch.eq(a*b, torch.mul(a, b))))# True
print(torch.all(torch.eq(a/b, torch.div(a, b))))# True
print(torch.all(torch.eq(a**2, torch.pow(a, 2))))# True
print(torch.all(torch.eq(a**2, a.pow(2))))# True
print(torch.all(torch.eq(a.sqrt(), a.pow(0.5))))# True
print(torch.all(torch.eq(a.sqrt(), a**0.5)))# True
print(torch.all(torch.eq(a.rsqrt(), 1/a.sqrt())))# True rsqrt()开平方根的倒数
print(torch.all(torch.eq(a//b, torch.floor_divide(a, b))))# True
print(torch.all(torch.eq(a % b, torch.remainder(a, b))))# True

a = torch.exp(torch.ones(2, 2)) # e指数
print(a)
b = torch.log(a) # e对数
print(b)
c = torch.log10(a) # 10对数
print(c)
d = torch.log2(a) # 2对数
print(d)
e = torch.log1p(a) # log(1+x) log为自然对数
print(e)

a = torch.tensor(-3.14)
print(a.floor()) # 向下取整 -4.
print(a.ceil()) # 向上取整 -3.
print(a.trunc()) # 截断(整数部分) -3.
print(a.frac()) # 小数部分 -0.14
print(a.round()) # 四舍五入 -3.
print(a.trunc().abs()) # 取绝对值 3.

a = torch.rand(2, 3)*15
print(a.max()) # 最大值
print(a.median()) # 中位数
print(a.mean()) # 均值
print(a.std()) # 标准差
print(a.clamp(10))# 小于10的数变为10
print(a.clamp(10, 12))# 小于10的数变为10，大于12的数变为12

# 2. 矩阵运算
# 2.1 矩阵乘法与转置
a = torch.tensor([[1, 2], [3, 4.]])
b = torch.tensor([[1, 1], [1, 1.]])
print(torch.all(torch.eq(a@b, torch.matmul(a, b))))# True
print(torch.all(torch.eq(a@b, a.mm(b))))# True, mm只能用于2维(dim=2)的tensor，bmm用于3维的tensor，后两维数进行矩阵乘法
print(torch.all(torch.eq(a@b, a*b)))# False
# 向量也可以用@运算
# 关注矩阵算结果是矩阵还是向量 @一个向量得到一个向量dim=1 @一个矩阵得到一个矩阵dim=2

print(torch.all(torch.eq(a.t(), torch.transpose(a, 0, 1))))# True t适用于2维(dim=2)的tensor，transpose适用于任意维度的tensor

# example1
x = torch.rand(4, 784)
w = torch.rand(512, 784)
print((x@w.t()).shape)# torch.Size([4, 512])

# example2
a = torch.rand(4, 3, 28, 64)
b = torch.rand(4, 3, 64, 32)
# torch.mm(a, b) # error 4维(dim=4)tensor不能用mm
# 高维tensor的矩阵乘法，只对最后两个维度做矩阵乘法，其他维度保持不变(其他维度的维数必须相等，除非其中一个维数为1(Broadcast))
print((a@b).shape)# torch.Size([4, 3, 28, 32])


# 2.2 矩阵的逆
a = torch.tensor([[1, 2], [3, 4.]])
print(torch.all(torch.eq(torch.inverse(a), a.inverse())))# True

# 2.3 矩阵的特征值和特征向量
a = torch.tensor([[1, 2], [3, 4.]])
eigenvalues, eigenvectors = torch.linalg.eig(a)
print(eigenvalues)
print(eigenvectors)

# 2.4 矩阵的奇异值分解
a = torch.tensor([[1, 2], [3, 4.]])
U, S, V = torch.linalg.svd(a)
print(U)
print(S)
print(V)

# 2.5 矩阵的QR分解
a = torch.tensor([[1, 2], [3, 4.]])
Q, R = torch.linalg.qr(a)
print(Q)
print(R)

# 2.6 矩阵的LU分解
a = torch.tensor([[1, 2], [3, 4.]])
P, L, U = torch.linalg.lu(a)
print(P)
print(L)
print(U)

# 2.7 矩阵的Cholesky分解
a = torch.tensor([[1, 2], [2, 5.]])
L = torch.linalg.cholesky(a)
print(L)

# 2.8 矩阵的行列式
a = torch.tensor([[1, 2], [3, 4.]])
print(torch.linalg.det(a))# tensor(-2.)

# 2.9 矩阵的范数(范数：a与0的距离(d(a,0)=||a||)，而d(a,b)=||a-b|| )
a = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
print(torch.linalg.norm(a, ord=1))# 1范数 tensor(12.) max(sum(abs(a), dim=0)) 12
print(torch.linalg.norm(a, ord=2))# 2范数 tensor(9.5255) sqrt(1^2+2^2+3^2+4^2+5^2+6^2)=9.5255
print(torch.linalg.norm(a, ord='fro'))# Frobenius范数 tensor(9.5394) sqrt(1^2+2^2+3^2+4^2+5^2+6^2)=9.5394
print(torch.linalg.norm(a, ord=float('inf')))# 无穷范数 tensor(11.) max(sum(abs(a), dim=1)) 11

# 2.10 矩阵的秩
a = torch.tensor([[1, 2], [3, 4.]])
print(torch.linalg.matrix_rank(a))# 2

# 2.11 矩阵的解线性方程组
a = torch.tensor([[1, 2], [3, 4.]])
b = torch.tensor([[1], [2.]])
print(torch.linalg.solve(a, b))

# 2.12 矩阵的伪逆
a = torch.tensor([[1, 2], [3, 4.]])
print(torch.linalg.pinv(a))

# 2.13 矩阵的对角线元素
a = torch.tensor([[1, 2], [3, 4.]])
print(torch.diag(a))# [1, 4]

# 2.14 矩阵的对角线元素之和
a = torch.tensor([[1, 2], [3, 4.]])
print(torch.trace(a))# 5

# 3.矩阵表示各点，求各点之间的距离
a = torch.tensor([[1, 2], [3, 4.]])
b = torch.tensor([[1, 2], [5, 6.]])
print(torch.cdist(a, b))# 2*2的矩阵 [[0.0000, 5.6569],[2.8284, 2.8284]]
#                                    [1, 2]    [5, 6]
#                            [1, 2]     0      5.6569
#                            [3, 4]   2.8284   2.8284






