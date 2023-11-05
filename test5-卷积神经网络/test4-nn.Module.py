import torch
import torch.nn as nn

# nn.Module
# 一个基本的父类，所有神经网络模块都应该继承这个类，这个类提供了一些基本的功能,也会参数初始化
# pytorch中的模块都是继承自nn.Module，这些模块可以相互嵌套，形成一个神经网络
# 好处：
# 1.有很多layer可用，比如卷积层，池化层，全连接层等等
# 2.Container
#   nn.Sequential(包含继承自nn.Module的模块的有序容器)
# 3.parameters
#   一个模块的参数可以通过调用parameters()方法来获取
#   list(model.parameters())[i]获取第i个参数矩阵(第i层的参数矩阵)
#   dict(model.named_parameters())获取参数矩阵的名字(0.weight/0.bias)和参数矩阵的字典
# 4.modules
class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.net = nn.Linear(4, 3)

    def forward(self, x):
        return self.net(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            BasicNet(),
            nn.ReLU(),
            nn.Linear(3, 2)
        )

    def forward(self, x):
        return self.net(x)

x = torch.randn(2, 4)
net = Net()
y = net(x)
para_list = list(net.parameters())
print(para_list[0])
print()
para_dict = dict(net.named_parameters())
print(para_dict['net.0.net.weight'])
print()
net_children = list(net.children())# 获取子模块
print(net_children)
print()
net_modules = list(net.modules())
print(net_modules) # 返回Net本身，Net的子模块，Net的子模块的子模块...
print()

# 以上例子是Module套Module的例子
# 可以查看net的parameters,children(包含哪些网络)等

# 5.to(device)
# 将模型转移到GPU上
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# 6.加载和保存模型

# 保存和加载模型的参数
# net.load_state_dict(torch.load('net_params.pkl'))
# torch.load中的map_location可以将模型从GPU加载到CPU,或者将模型从CPU加载到GPU
# load_state_dict中的strict参数可以控制是否严格按照字典加载参数，如果为False，可以加载部分参数
# torch.save(net.state_dict(), 'net_params.pkl')
# 可以保存模型的中间状态

# 保存和加载整个模型
# torch.save(net, 'net.pkl')
# net = torch.load('net.pkl')

# 7.train/test
# 用于切换模型的训练和测试模式
net.train()
net.eval()

# 8. implement your own layer
# 可以将Function中有的操作封装成Module
# 然后可以在nn.Sequential中使用
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        # nn.Parameter是一个特殊的Tensor，当作为Module的属性时，会自动被添加到参数列表里,以便于梯度更新时调用net.parameters()
        # nn.Parameter中requires_grad默认为True
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return x @ self.weight.t() + self.bias

# 9. nn.ModuleList
# 用于存储一些Module，可以通过索引来访问这些Module
# 但是这些Module不会被自动添加到参数列表中
# 需要手动将x = List[i](x)将x通过i个model，这样的操作添加到参数列表中
# 使用nn.ModuleList的好处是可以通过append来添加Module

# 10. self.apply()
# 用于对Module中的所有参数进行初始化
# 例如：
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(20, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)
        self.apply(self._init_weights) # 对model每个部分的参数都会调用apply中的函数_init_weights，self.apply()常用于初始化

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):# 只对Linear层进行初始化
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)







