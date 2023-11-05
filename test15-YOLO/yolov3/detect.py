import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolo_v3_net import YoloV3Net
from config import *
from utils import *
from PIL import Image, ImageDraw
from torchvision import transforms

tf = transforms.Compose([
    transforms.ToTensor()
])


class Detector(nn.Module):
    def __init__(self, class_num):
        super(Detector, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights_path = 'params/yolo_v3_net.pth'
        self.net = YoloV3Net(class_num).to(self.device)
        if os.path.exists(self.weights_path):
            self.net.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            print('load params success')
        else:
            raise FileNotFoundError('params file not found')

        self.eval()# 设置为测试模式

    def get_index_and_bias(self, output, find_threshold=FIND_THRESHOLD):
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(output.size(0), output.size(1), output.size(2), 3, -1)
        # output: [batch_size, feature_size, feature_size, 3, 5 + class_num],表示[n,h,w,anchor,5+class_num]

        mask = torch.sigmoid(output[..., 0]) > find_threshold
        # mask: [batch_size, feature_size, feature_size, 3]

        index = mask.nonzero()# 返回非零元素的索引
        # index: [n, 4] n表示有多少个大于阈值的点, 4表示batch_size, feature_size, feature_size, 3, 这4个维度的索引
        # 这里index[n, a, b, i]表示第n个大于阈值的点在图中的a、b位置，即a、b是框的中心点的位置的整数部分，i表示第i个anchor
        bias = output[mask]# bias: [n, 5+class_num]
        # bias: [n, 5 + class_num]

        return index, bias

    def get_true_position(self, index, bias, t1, t2, _anchors, case1, case2):
        _anchors = torch.tensor(_anchors, dtype=torch.float32).to(self.device)
        a = index[:, 3]# a: [n]
        cx = (index[:, 2].float() + bias[:, 1].float())*t1/case1 # cx: [n]
        cy = (index[:, 1].float() + bias[:, 2].float())*t2/case2 # cy: [n]
        w = torch.exp(bias[:, 3]).float()*_anchors[a, 0]/case1 # w: [n]
        h = torch.exp(bias[:, 4]).float()*_anchors[a, 1]/case2 # h: [n]
        p = torch.sigmoid(bias[:, 0]).float() # p: [n]
        c = F.softmax(bias[:, 5:], dim=1) # c: [n, class_num]
        p = p*c.max(dim=1)[0] # p: [n]
        cls = torch.argmax(c, dim=1)# cls: [n]

        return torch.stack([p, cx, cy, w, h, cls], dim=1)

    def compute_iou(self, box1, box2):
        # box1: [n, 4], 4表示[cx, cy, w, h]
        # box2: [1, 4], 4表示[cx, cy, w, h]
        # 计算box2与box1各个框的iou

        # 克隆避免修改原始数据
        _box1 = box1.clone()
        _box2 = box2.clone()

        # 1.变为左上角和右下角的坐标
        _box1[:, 0] = _box1[:, 0] - _box1[:, 2]/2
        _box1[:, 1] = _box1[:, 1] - _box1[:, 3]/2
        _box1[:, 2] = _box1[:, 0] + _box1[:, 2]
        _box1[:, 3] = _box1[:, 1] + _box1[:, 3]
        _box2[:, 0] = _box2[:, 0] - _box2[:, 2]/2
        _box2[:, 1] = _box2[:, 1] - _box2[:, 3]/2
        _box2[:, 2] = _box2[:, 0] + _box2[:, 2]
        _box2[:, 3] = _box2[:, 1] + _box2[:, 3]

        # 2.计算box2与box1各个框的iou
        n = _box1.size(0)
        _box2 = torch.tile(_box2, (n, 1))# box2: [n, 4]

        # 计算交集
        left_up = torch.max(_box1[:, :2], _box2[:, :2])# left_up: [n, 2]
        right_down = torch.min(_box1[:, 2:], _box2[:, 2:])# right_down: [n, 2]
        inter = torch.clamp(right_down - left_up, min=0)# inter: [n, 2] # clamp函数是将小于0的数变为0，用于处理没有交集的情况
        inter = inter[:, 0]*inter[:, 1]# inter: [n]

        # 计算并集
        area1 = (_box1[:, 2] - _box1[:, 0])*(_box1[:, 3] - _box1[:, 1])# area1: [n]
        area2 = (_box2[:, 2] - _box2[:, 0])*(_box2[:, 3] - _box2[:, 1])# area2: [n]
        union = area1 + area2 - inter# union: [n]

        return inter/union

    def non_max_suppression(self, boxes, iou_threshold=IOU_THRESHOLD, detect_threshold=DETECT_THRESHOLD):
        # boxes: [n, 6], 6表示[p, cx, cy, w, h, cls]
        # threshold: 阈值
        # 非极大值抑制
        if boxes.size(0) == 0:
            return torch.zeros((0, 6), dtype=torch.float32).to(self.device)
        # 1.根据置信度排序
        boxes = boxes[boxes[:, 0].argsort(descending=True)]
        boxes = boxes[boxes[:, 0] > detect_threshold*boxes[:, 0].max()]# 置信度大于阈值的框

        # 2.根据iou进行非极大值抑制
        keep = []
        while boxes.size(0) > 0:
            keep.append(boxes[0])
            if boxes.size(0) == 1:
                break
            iou = self.compute_iou(boxes[1:, 1:5], boxes[0, 1:5].unsqueeze(0))
            # 将类别不同的iou置为0并排除iou大于阈值的框
            index = boxes[1:, 5] != boxes[0, 5]
            boxes = boxes[1:][index][iou[index] < iou_threshold]
        keep = torch.stack(keep, dim=0)

        return keep

    def forward(self, x, _anchors, case1, case2):
        output_13, output_26, output_52 = self.net(x.to(self.device))
        index_13, bias_13 = self.get_index_and_bias(output_13)
        index_26, bias_26 = self.get_index_and_bias(output_26)
        index_52, bias_52 = self.get_index_and_bias(output_52)
        boxes_13 = self.get_true_position(index_13, bias_13, DATA_WIDTH/13, DATA_HEIGHT/13, _anchors[13], case1, case2)
        boxes_26 = self.get_true_position(index_26, bias_26, DATA_WIDTH/26, DATA_HEIGHT/26, _anchors[26], case1, case2)
        boxes_52 = self.get_true_position(index_52, bias_52, DATA_WIDTH/52, DATA_HEIGHT/52, _anchors[52], case1, case2)
        boxes = torch.cat([boxes_13, boxes_26, boxes_52], dim=0)
        boxes = self.non_max_suppression(boxes)

        return boxes


if __name__ == '__main__':

    test_path = 'data/images/000022.jpg'
    save_path = 'result'

    img = Image.open(test_path)
    detector = Detector(CLASS_NUM)
    case1 = DATA_WIDTH/img.width
    case2 = DATA_HEIGHT/img.height

    _img = make_416_image(test_path)
    _img = _img.resize((DATA_WIDTH, DATA_HEIGHT))
    _img = tf(_img)
    result = detector(_img.unsqueeze(0), anchors, case1, case2)
    print(result)
    draw = ImageDraw.Draw(img)

    for box in result:
        p, cx, cy, w, h, cls = box
        x1 = torch.clamp((cx-w/2), min=0)
        y1 = torch.clamp((cy-h/2), min=10)
        x2 = torch.clamp((cx+w/2), max=img.width)
        y2 = torch.clamp((cy+h/2), max=img.height)
        draw.rectangle((x1, y1, x2, y2), outline='red', width=2)
        draw.rectangle((x1, y1-10, x1+75, y1), fill='red')
        draw.text((x1+2, y1-11), num2class[cls.item()]+':'+f'{p:.2f}', fill='white')

    img.show()
    img.save(os.path.join(save_path, os.path.basename(test_path)))





