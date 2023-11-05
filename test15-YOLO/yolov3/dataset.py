import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset
from utils import *
from config import *
from torchvision import transforms

tf = transforms.Compose([
    transforms.ToTensor()
])


def one_hot(cls_num, v):
    b = np.zeros(cls_num)
    b[v] = 1.
    return b


class YoloDataset(Dataset):
    def __init__(self, fitting_threshold=FITTING_THRESHOLD):
        f = open('data.txt', 'r')
        self.dataset = f.readlines()
        self.fitting_threshold = fitting_threshold

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 分割各个框
        temp_data = self.dataset[idx].split()
        _boxes = np.array([float(i) for i in temp_data[1:]])
        boxes = np.split(_boxes, len(_boxes) // 5)

        # 处理图片
        img = make_416_image(os.path.join('data/images', temp_data[0]))
        w, h = img.size
        img = img.resize((DATA_WIDTH, DATA_HEIGHT))
        case_w = DATA_WIDTH / w # 缩放比例
        case_h = DATA_HEIGHT / h # 缩放比例
        img_data = tf(img)

        # 处理各个框，如果标记的框和预设的框有较大重叠，就在某个点使用这个框
        labels = {}
        for feature_size, _anchor in anchors.items():
            labels[feature_size] = np.zeros((feature_size, feature_size, 3, 5 + CLASS_NUM)) # 3表示3个anchor

            for box in boxes:
                cls, cx, cy, w, h = box
                cx *= case_w
                cy *= case_h
                w *= case_w
                h *= case_h
                # math.modf() 返回浮点数的小数和整数部分，先小数后整数
                _x, _index_x = math.modf(cx * feature_size / DATA_WIDTH) # 在哪一格 = x坐标/(x总宽度/格子数)
                _y, _index_y = math.modf(cy * feature_size / DATA_HEIGHT) # 在哪一格 = y坐标/(y总宽度/格子数)

                for i, anchor in enumerate(_anchor):
                    fitting_rate = (w * h) / (anchor[0] * anchor[1]) # 框的面积/anchor的面积
                    fitting_rate = min(fitting_rate, 1 / fitting_rate) # 取最小值
                    p_w, p_h = w/anchor[0], h/anchor[1]

                    if fitting_rate > labels[feature_size][int(_index_y)][int(_index_x)][i][0] and fitting_rate > self.fitting_threshold:
                        labels[feature_size][int(_index_y), int(_index_x), i] = np.array([fitting_rate, _x, _y, np.log(p_w), np.log(p_h), *one_hot(CLASS_NUM, int(cls))])
                        # 网络输出可能为负数,需要取指数,这里对应的是取对数,注意这里是labels[feature_size][h][w][i]，不是labels[feature_size][w][h][i]
                        # 注意这里怎么表示到各个量的，后面要还原

        for feature_size in labels:
            # 将有类别的分量的置信度设为1
            labels[feature_size] = torch.from_numpy(labels[feature_size]).float()
            mask = labels[feature_size][..., 0] > 0# [h, w, 3]
            labels[feature_size][..., 0][mask] = 1

        return labels[13], labels[26], labels[52], img_data
        # [13, 13, 3, 5+CLASS_NUM], [26, 26, 3, 5+CLASS_NUM], [52, 52, 3, 5+CLASS_NUM], [3, 416, 416]


if __name__ == '__main__':
    dataset = YoloDataset()
    print(len(dataset))
    print(dataset[0][0].shape)
