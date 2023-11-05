# Data Augmentation
# 创造更多的数据，让模型更加健壮
# 对于limited data的情况，可以减小网络的复杂度，正则化，或者增加数据量，防止过拟合
# 本节介绍数据增强

# 将数据集中的图片进行变换，生成更多的数据
# 数据变多了，但是数据的分布没有变化，所以有帮助但是不是很大

import torch
import torchvision

# 1. Filp(翻转)
train_db1 = torchvision.datasets.MNIST('mnist_data', train=True,  # 使用train data
                                      download=True,  # 下载数据集
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.RandomVerticalFlip(p=0.5),  # 随机竖直翻转, p是概率
                                          torchvision.transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转, p是概率
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ]))
print(len(train_db1)) # 60000没有增大

# 2. Rotation(旋转)
train_db2 = torchvision.datasets.MNIST('mnist_data', train=True,  # 使用train data
                                      download=True,  # 下载数据集
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.RandomRotation(15),  # 随机旋转-15~15度
                                          torchvision.transforms.RandomRotation([0, 180]),  # 随机旋转0~180度
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ]))
print(len(train_db2)) # 60000没有增大

# 3. Scale(缩放)(图片大小变大时，会有0边，图片大小变小时，会有裁剪，常与RandomCrop配合使用)
train_db3 = torchvision.datasets.MNIST('mnist_data', train=True,  # 使用train data
                                        download=True,  # 下载数据集
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.Resize([32, 32]),  # 缩放到32*32
                                            torchvision.transforms.Resize([64, 64], interpolation=torchvision.transforms.InterpolationMode.BICUBIC),  # 缩放到64*64, 使用双三次插值
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))
                                        ]))
print(len(train_db3)) # 60000没有增大

# 4. Crop Part(裁剪)与中心裁剪
# 从图片中随机裁剪出一块
train_db4 = torchvision.datasets.MNIST('mnist_data', train=True,  # 使用train data
                                        download=True,  # 下载数据集
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.Resize([32, 32]),  # 缩放到32*32
                                            torchvision.transforms.RandomCrop([28, 28]),  # 随机裁剪28*28
                                            torchvision.transforms.CenterCrop([28, 28]),  # 中心裁剪28*28
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))
                                        ]))

# 5. 注意：若变换过程中有的像素缺失，会用0填充

# 6. Noise(噪声)
# 为图片添加噪声
train_db5 = torchvision.datasets.MNIST('mnist_data', train=True,  # 使用train data
                                        download=True,  # 下载数据集
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,)),
                                            torchvision.transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x))  # 添加噪声
                                        ]))
print(len(train_db5)) # 60000没有增大

# 7. ColorJitter(颜色抖动)
# 随机改变图片的亮度、对比度、饱和度、色相
train_db6 = torchvision.datasets.MNIST('mnist_data', train=True,  # 使用train data
                                        download=True,  # 下载数据集
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,)),
                                            torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)  # 随机改变亮度、对比度、饱和度、色相
                                        ]))
print(len(train_db6)) # 60000没有增大

# 8. Grayscale(灰度化)
# 将图片转换为灰度图
train_db7 = torchvision.datasets.MNIST('mnist_data', train=True,  # 使用train data
                                        download=True,  # 下载数据集
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,)),
                                            torchvision.transforms.Grayscale(num_output_channels=1)  # 转换为灰度图
                                        ]))
print(len(train_db7)) # 60000没有增大

# 9. RandomErasing(随机擦除)
# 随机擦除图片中的一块区域
train_db8 = torchvision.datasets.MNIST('mnist_data', train=True,  # 使用train data
                                        download=True,  # 下载数据集
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,)),
                                            torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)  # 参数：概率、比例、长宽比、填充值、是否原地操作
                                        ]))
print(len(train_db8)) # 60000没有增大

# 10. 将以上数据合并
train_db = train_db1 + train_db2 + train_db3 + train_db4 + train_db5 + train_db6 + train_db7 + train_db8
print(len(train_db)) # 480000


# 11.设置随机种子
import random
import numpy as np
import os
np.random.seed(1234)
random.seed(1234)
os.environ['PYTHONHASHSEED'] = str(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True # some cudnn methods can be random even after fixing the seed
#                                           unless you tell it to be deterministic
torch.backends.cudnn.enabled = False

# 12.小心ToTensor()的位置
#    而且ToTensor自带归一化，还原时需要乘以255




