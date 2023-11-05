import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img


def main():
    model = models.mobilenet_v3_large(pretrained=True)
    target_layers = [model.features[-1]]# 选取最后一个卷积层

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image，224x224, 网络训练的时候是这个尺寸，后面还需要归一化
    img_path = "both.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 224)# 将图片剪裁并缩放到224x224

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 281  # tabby, tabby cat 在imagenet中的类别
    # target_category = 254  # pug, pug-dog 在imagenet中的类别

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    # 一次也可以传入多张图片，[N, C, H, W]，target_category=None的话，会使用模型的预测分类类型

    grayscale_cam = grayscale_cam[0, :]# 选取第一张图片的结果
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)# 后处理，生成可视化热力图
    plt.imshow(visualization)
    plt.savefig("results/result-cnn.png")


if __name__ == '__main__':
    main()
