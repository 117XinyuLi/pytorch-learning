import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from nets.facenet import Facenet as facenet


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和backbone需要修改！
# --------------------------------------------#
class Facenet(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测要修改model_path，指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表准确度较高，仅代表该权值在验证集上泛化性能较好。
        # --------------------------------------------------------------------------#
        "model_path": "model_data/facenet_mobilenet.pth",
        # --------------------------------------------------------------------------#
        #   输入图片的大小。
        # --------------------------------------------------------------------------#
        "input_shape": [160, 160, 3],
        # --------------------------------------------------------------------------#
        #   所使用到的主干特征提取网络
        # --------------------------------------------------------------------------#
        "backbone": "mobilenet",
        # --------------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # --------------------------------------#
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Facenet
    # ---------------------------------------------------#
    def __init__(self):
        self.__dict__.update(self._defaults)# set up default values

        self.generate()

    def generate(self):
        # ---------------------------------------------------#
        #   载入模型与权值
        # ---------------------------------------------------#
        print('Loading weights into state dict...')
        model = facenet(backbone=self.backbone, mode="predict")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        self.net = model.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    def letterbox_image(self, image, size):
        if self.input_shape[-1] == 1:
            image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        if self.input_shape[-1] == 1:
            new_image = new_image.convert("L")
        return new_image

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image_1, image_2):
        # ---------------------------------------------------#
        #   图片预处理，归一化
        # ---------------------------------------------------#
        with torch.no_grad():
            image_1 = self.letterbox_image(image_1, [self.input_shape[1], self.input_shape[0]])
            image_2 = self.letterbox_image(image_2, [self.input_shape[1], self.input_shape[0]])

            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(np.array(image_1, np.float32) / 255, (2, 0, 1)), 0))
            photo_2 = torch.from_numpy(np.expand_dims(np.transpose(np.array(image_2, np.float32) / 255, (2, 0, 1)), 0))

            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            output1 = self.net(photo_1).cpu().numpy()
            output2 = self.net(photo_2).cpu().numpy()

            # ---------------------------------------------------#
            #   计算二者之间的距离
            # ---------------------------------------------------#
            l1 = np.linalg.norm(output1 - output2, axis=1)

        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image_1))

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(image_2))
        plt.text(-12, -12, 'Distance:%.3f' % l1, ha='center', va='bottom', fontsize=11)
        plt.show()
        return l1
