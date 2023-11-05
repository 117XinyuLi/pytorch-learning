"""
    混淆矩阵
    Recall、Precision、MIOU计算
"""
import numpy as np
from sklearn.metrics import confusion_matrix
from utils import keep_image_size_open
import cv2

# 一般用于黑白图，黑白二分类，一般图片的像素值为0~255又三个通道，MIOU不好计算，不同通道可能MIOU不同
# labels为你的像素值的类别
def get_miou_recall_precision(label_image, pred_image, labels):
    label = label_image.reshape(-1)
    pred = pred_image.reshape(-1)
    out = confusion_matrix(label, pred, labels=labels)
    r, l = out.shape
    iou_temp = 0
    recall = {}
    precision = {}
    for i in range(r):
        TP = out[i][i]
        temp = np.concatenate((out[0:i, :], out[i + 1:, :]), axis=0)
        sum_one = np.sum(temp, axis=0)
        FP = sum_one[i]
        temp2 = np.concatenate((out[:, 0:i], out[:, i + 1:]), axis=1)
        FN = np.sum(temp2, axis=1)[i]
        TN = temp2.reshape(-1).sum() - FN
        iou_temp += (TP / (TP + FP + FN))
        recall[i] = TP / (TP + FN)
        precision[i] = TP / (TP + FP)
    MIOU = iou_temp / len(labels)
    return out, MIOU, recall, precision


if __name__ == '__main__':
    # 读取图片并缩放
    label = keep_image_size_open('data/VOCtrain/SegmentationClass/000039.png')
    pred = keep_image_size_open('data/VOCtrain/SegmentationClass/000039.png')
    # 将图片转换为灰度图，灰度图和一般图的像素值均为为0~255
    l, p = np.array(label).astype(int), np.array(pred).astype(int)# [256, 256, 3] np的channel在最后
    l, p = cv2.cvtColor(l.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.cvtColor(p.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    print(l.shape, p.shape)# (256, 256) (256, 256)
    l = np.where(l > 64, 255, 0)# 这里把l中大于64的赋为255，小于等于64的赋为0，阈值64可以自己调整
    p = np.where(p > 64, 255, 0)
    # 显示图片
    cv2.imshow('label', l.astype(np.uint8))
    cv2.imshow('pred', p.astype(np.uint8))
    cv2.waitKey(0)
    # 进行评价
    out, MIOU, recall, precision = get_miou_recall_precision(l, p, [0, 255])
    print(out)
    print(MIOU)
    print(recall)
    print(precision)
