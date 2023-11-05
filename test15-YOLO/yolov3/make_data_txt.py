# 将图片使用labelImg VOC模式标注后的xml文件转换为yolo需要的txt文件

import os
import xml.etree.ElementTree as et  # 使用ElementTree解析xml文件

class_num = {
    'person': 0,
    'horse': 1,
    'bicycle': 2,
}

xml_path = 'data/image_voc/'  # xml文件路径
xml_filenames = os.listdir(xml_path)  # 获取xml文件列表
with open('data.txt', 'a') as f:
    for xml_filename in xml_filenames:
        xml_filename_path = os.path.join(xml_path, xml_filename)
        tree = et.parse(xml_filename_path)
        root = tree.getroot()
        filename = root.find('filename')
        names = root.findall('object/name')
        boxs = root.findall('object/bndbox')
        data = [filename.text]
        for name, box in zip(names, boxs):
            cls = class_num[name.text]
            cx, cy, w, h = (int(box[0].text) + int(box[2].text)) // 2, (int(box[1].text) + int(box[3].text)) // 2, \
                           int(box[2].text) - int(box[0].text), int(box[3].text) - int(box[1].text)
            data.append(cls)
            data.append(cx)
            data.append(cy)
            data.append(w)
            data.append(h)
        # data = [filename.text, cls, cx, cy, w, h, ...]
        _str = ''
        for i in data:
            _str = _str + ' ' + str(i)
        f.write(_str + '\n')

f.close()
