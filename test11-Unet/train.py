import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from data import VOC2007
from Unet import Unet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/Unet.pth'
data_path = 'data/VOCtrain'
save_path = 'result'
torch.backends.cudnn.benchmark = True


def to_black_and_white(img):
    mask1 = img[:, 0, :, :] > 0
    mask2 = img[:, 1, :, :] > 0
    mask3 = img[:, 2, :, :] > 0
    mask = mask1 | mask2 | mask3
    mask = torch.stack([mask, mask, mask], dim=1)
    img[mask] = 1
    return img


if __name__ == '__main__':
    dataset = VOC2007(data_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    net = Unet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('load weight')

    epochs = 100
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs * len(dataloader), T_mult=1)

    for epoch in range(epochs):
        idx, image, segment, output = 0, None, None, None
        for idx, (image, segment) in enumerate(dataloader):
            image, segment = image.to(device), segment.to(device)
            output = net(image)
            segment = to_black_and_white(segment)# 转为黑白图，也可以不转，直接用BCELoss拟合原图色彩，但是容易颜色混乱
            loss = criterion(output, segment)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if idx % 10 == 0:
                print(f'epoch: {epoch}, idx: {idx}, loss: {loss.item()}')

            if idx % 50 == 0 and idx != 0:
                pass
                # torch.save(net.state_dict(), weight_path)
                # print('save weight')

        if epoch % 10 == 0:
            _img = image[0]
            _seg = segment[0]
            _out = output[0]

            _image = torch.stack([_img, _seg, _out], dim=0)
            save_image(_image, save_path + f'/{epoch}_{idx}.png', nrow=3)  # 网络输入输出都是归一化的，这里会自动反归一化
