import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from Unet import Unet
from data import VOC2007

net = Unet().cuda()
weight_path = 'params/Unet.pth'
if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))
    print('load weight')
else:
    print('no weight')

test_path = 'data/VOCtest'
dataset = VOC2007(test_path)
save_path = 'test'
if not os.path.exists(save_path):
    os.makedirs(save_path)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
for idx, (image, segment) in enumerate(dataloader):
    image, segment = image.cuda(), segment.cuda()
    output = net(image)
    _img = image[0]
    _seg = segment[0]
    _out = output[0]
    _image = torch.stack([_img, _seg, _out], dim=0)
    save_image(_image, save_path + f'/{idx}.png', nrow=3)


