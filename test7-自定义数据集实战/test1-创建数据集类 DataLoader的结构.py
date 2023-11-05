
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# 加载数据集
# 示例：读取数字的数据集
# 需要继承Dataset类，重写__init__和__getitem__方法
class NumbersDataset(Dataset):
    def __init__(self, training=True):
        super(NumbersDataset, self).__init__()
        if training:
            self.samples = list(range(1, 1001))
        else:
            self.samples = list(range(1001, 1501))

    def __getitem__(self, index):# 返回目标索引的数据
        return self.samples[index]

    def __len__(self):# 返回数据集的长度
        return len(self.samples)
# 写完之后，就可以直接使用DataLoader来加载数据集了

# DataLoader的结构
# DataLoader[i]返回的是一个batch的数据 比如在图片中，shape为[batch_size, 3, 224, 224]
# for data, target in enumerate(DataLoader):中
# data是一个batch的数据，target是一个batch的标签，比如在图片中，data的shape为[batch_size, 3, 224, 224]，target的shape为[batch_size]
# data[i]是第i个样本，target[i]是第i个样本的标签，比如在图片中，data[i].shape为[3, 224, 224]，target[i]是一个标量

# num_workers
# num_workers是指在加载数据时，使用多少个子进程来加速数据加载的过程
