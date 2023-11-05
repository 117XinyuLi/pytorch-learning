import csv
import glob
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# pokemon dataset
class Pokemon(Dataset):
    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()
        self.root = root
        self.resize = resize
        self.name2label = {}
        # 获取所有类别
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):#判断是否是文件夹
                continue
            self.name2label[name] = len(self.name2label.keys())# 把读进来时字典的长度作为此时的类别值
        # print(self.name2label)# {'bulbasaur': 0, 'charmander': 1, 'mewtwo': 2, 'pikachu': 3, 'squirtle': 4}

        # 生成/读取csv文件，返回所有图片的路径和标签
        self.images, self.labels = self.load_csv('images.csv')

        # 划分训练集和测试集
        if mode == 'train':# 训练集60%
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val':# 验证集20%
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:# 测试集20%
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    # 获取所有图片的路径，并写入/读取csv文件
    def load_csv(self, filename):
        # 如果csv文件不存在，就生成
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            # 读取所有图片的路径
            for name in self.name2label.keys():
                # 获取所有图片的路径
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))

            # print(len(images), images)# 1167 ['pokemon\\bulbasaur\\00000006.jpg',...]

            # 打乱图片的顺序
            random.shuffle(images)
            # 写入csv文件
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:# img = 'pokemon\\bulbasaur\\00000006.jpg'
                    name = img.split(os.sep)[-2]# 利用os.sep将字符串分割成列表，取倒数第二个元素
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print('writen into csv file:', filename)

        # 读取csv文件
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            # 将csv文件中的内容读取出来，放到images,label列表中
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)

        return images, labels

    # 将归一化的图片还原
    def denormalize(self, x_hat):
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        x = x_hat * std + mean
        return x

    # 返回数据集的大小
    def __len__(self):
        return len(self.images)

    # 返回数据集中第index个图片,最主要需要实现的函数
    def __getitem__(self, idx):
        # idx 0~len(self.images)
        # img: 'pokemon\\bulbasaur\\00000006.jpg'
        # label: 0
        img, label = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path= > image data
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),  # resize
            transforms.RandomRotation(15),  # rotate
            transforms.CenterCrop(self.resize),  # crop
            transforms.ToTensor(),  # numpy => tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize
        ])
        img = tf(img)
        label = torch.tensor(label)
        return img, label