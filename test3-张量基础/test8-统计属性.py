# 统计属性

import torch

# 1.norm(范数)
a = torch.full([8], 1.)  # 数据类型默要为float32 (1.而不是1),这里a是一个向量，张量也可以进行范数运算
b = a.view(2, 4)
c = a.view(2, 2, 2)
# a.norm(1)是求a中所有元素的绝对值之和
print(a.norm(1))  # 8.0
print(b.norm(1))  # 8.0
print(c.norm(1))  # 8.0
# a.norm(2)是求a中所有元素的平方和的平方根
print(a.norm(2))  # 2.8284270763397217
print(b.norm(2))  # 2.8284270763397217
print(c.norm(2))  # 2.8284270763397217

# 在dim上求范数,dim的使用方法如test7所示
print(b.norm(1, dim=0))  # tensor([2., 2., 2., 2.])
print(b.norm(1, dim=1))  # tensor([4., 4.])
print(c.norm(1, dim=0))  # tensor([[2., 2.], [2., 2.]])
print(c.norm(1, dim=1))  # tensor([[2., 2.], [2., 2.]])
print(c.norm(1, dim=2))  # tensor([[2., 2.], [2., 2.]])

# a.norm(1)和torch.linalg.norm(a, ord=1)是不一样的
a = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
print(a.norm(1))  # 21.0
print(torch.linalg.norm(a, ord=1))  # 12.0 max(sum(abs(a), dim=0))

# 2.max() min() mean() prod() sum() argmax() argmin()
a = torch.arange(8).view(2, 4).float()
print(a)  # tensor([[0., 1., 2., 3.], [4., 5., 6., 7.]])
print(a.max())  # tensor(7.)
print(a.min())  # tensor(0.)
print(a.mean())  # tensor(3.5000)
print(a.prod())  # tensor(0.) prod()是求所有元素的乘积
print(a.sum())  # tensor(28.)

print(a.argmax())  # tensor(7) 不带参数的argmax()会将a中所有元素展开成一维数组,然后返回最大值的索引
print(a.argmin())  # tensor(0) 不带参数的argmin()会将a中所有元素展开成一维数组,然后返回最小值的索引

a = torch.randn(4, 8)
print(a.argmax(dim=1))  # 求每行最大值的索引 比如tensor([4, 5, 5, 6]),dim使用方法见test7

# dim和keepdim test7中有介绍
a = torch.arange(8).view(2, 4).float()
values, index = a.max(dim=1)
print(values)  # tensor([3., 7.]),返回每行最大值
print(index)  # tensor([3, 3]),返回每行最大值的索引,在行中的位置
values, index = a.max(dim=1, keepdim=True)
print(values)  # tensor([[3.], [7.]]),返回每行最大值,并保留维度
print(index)  # tensor([[3], [3]]),返回每行最大值的索引,并保留维度
# 上面keepdim和index的具体细节类比下面topk和kthvalue的keepdim和index

# 3.topk() kthvalue()
# topk()返回最大(小)的k个值和对应的索引
a = torch.arange(24).view(2, 3, 4).float()
print(a)  # tensor([[[ 0.,  1.,  2.,  3.], [ 4.,  5.,  6.,  7.], [ 8.,  9., 10., 11.]], [[12., 13., 14., 15.],
#                    [16., 17., 18., 19.], [20., 21., 22., 23.]]])
values, index = a.topk(2, dim=1) # 输出的shape为(2, 2, 4), dim=1维度上取前2个最大值,所以dim=1的维度为2,没有keepdim参数
print(values)  # tensor([[[ 8.,  9., 10., 11.], [ 4.,  5.,  6.,  7.]], [[20., 21., 22., 23.], [16., 17., 18., 19.]]])
#               在dim=1上取最大2个就是在(i,:,j)上取最大2个,找到的最大值在输出中的位置为(i,0,j)和(i,1,j) value[i,0,j]>value[i,1,j]
print(index)  # tensor([[[2, 2, 2, 2], [1, 1, 1, 1]], [[2, 2, 2, 2], [1, 1, 1, 1]]]) 表示最大值的索引
#               在dim=1上取最大2个值的索引index[i,0(或1),j]=x,，表示values[i,0(或1),j]在原tensor中的位置为(i,x,j)
values, index = a.topk(2, dim=1, largest=False)  # largest=False表示返回每行最小的2个值
print(values)  # tensor([[[0., 1., 2., 3.], [4., 5., 6., 7.]], [[12., 13., 14., 15.], [16., 17., 18., 19.]]])
print(index)  # tensor([[[0, 0, 0, 0], [1, 1, 1, 1]], [[0, 0, 0, 0], [1, 1, 1, 1]]])

# kthvalue()找第k小的值,返回的是值和索引
values, index = a.kthvalue(2, dim=1, keepdim=True)  # 返回每行第2小的值,有keepdim参数
print(values)  # tensor([[[ 4.,  5.,  6.,  7.]], [[16., 17., 18., 19.]]]) 有keepdim参数,所以输出的shape为(2, 1, 4)
#               在dim=1上取第2小的值就是在(i,:,j)上取第2小的值,找到的第2小的值在输出中的位置为(i,0,j)
print(index)  # tensor([[[1, 1, 1, 1]], [[1, 1, 1, 1]]])
#              在dim=1上取第2小的值的索引index[i,0,j]=x,，表示values[i,0,j]在原tensor中的位置为(i,x,j)

values, index = a.kthvalue(2, dim=1)
print(values)  # tensor([[ 4.,  5.,  6.,  7.], [16., 17., 18., 19.]]) 没有keepdim参数,所以输出的shape为(2, 4)
#                在dim=1上取第2小的值,就是在(i,:,j)上取第2小的值,找到的第2小的值在输出中的位置为(i,j)
print(index)  # tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
#                在dim=1上取第2小的值的索引index[i,j]=x,，表示values[i,j]在原tensor中的位置为(i,x,j)

# 4.compare
a = torch.randn(3, 4)
print(a > 0, a >= 0, a < 0, a <= 0, a == 0, a != 0)  # 返回的是bool类型的tensor
print(torch.gt(a, 0))  # tensor([[False,  True,  True,  True], [ True,  True,  True,  True], [ True,  True,  True, False]])
#           gt()和>的功能一样,gt()可以比较两个tensor的大小,>只能比较一个tensor和一个数的大小
print(torch.ge(a, 0))  # tensor([[False,  True,  True,  True], [ True,  True,  True,  True], [ True,  True,  True,  True]])
#           ge()和>=的功能一样,ge()可以比较两个tensor的大小,>=只能比较一个tensor和一个数的大小
print(torch.lt(a, 0))  # tensor([[ True, False, False, False], [False, False, False, False], [False, False, False,  True]])
#           lt()和<的功能一样,lt()可以比较两个tensor的大小,<只能比较一个tensor和一个数的大小
print(torch.le(a, 0))  # tensor([[ True, False, False, False], [False, False, False, False], [False, False, False,  True]])
#           le()和<=的功能一样,le()可以比较两个tensor的大小,<=只能比较一个tensor和一个数的大小
print(torch.eq(a, 0))  # tensor([[False, False, False, False], [False, False, False, False], [False, False, False, False]])
#           eq()和==的功能一样,eq()可以比较两个tensor的大小,==只能比较一个tensor和一个数的大小
print(torch.ne(a, 0))  # tensor([[ True,  True,  True,  True], [ True,  True,  True,  True], [ True,  True,  True,  True]])
#           ne()和!=的功能一样,ne()可以比较两个tensor的大小,!=只能比较一个tensor和一个数的大小

a = torch.randn(3, 4)
b = torch.ones(3, 4)
print(torch.eq(a, b))# 返回的是bool类型的tensor
print(torch.equal(a, b))  # 返回单个False  equal()比较两个tensor是否相等,相等返回True,不相等返回False
print(torch.all(torch.eq(a, b)))  # 返回单个tensor(False)  all()判断一个tensor中的所有元素是否都为True,是返回True,否返回False





