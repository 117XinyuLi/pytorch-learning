DATA_WIDTH = 416
DATA_HEIGHT = 416

CLASS_NUM = 3

anchors = {
    13: [[270, 254], [291, 179], [162, 304]],
    26: [[175, 222], [112, 235], [175, 140]],
    52: [[81, 118], [53, 142], [44, 28]]
}

ANCHORS_AREA = {
    13: [x*y for x, y in anchors[13]],# [68620, 52539, 49248]
    26: [x*y for x, y in anchors[26]],# [38850, 26360, 24500]
    52: [x*y for x, y in anchors[52]]# [9468, 7546, 1232]
}

num2class = {
    0: 'person',
    1: 'horse',
    2: 'bicycle'
}

# dataset
FITTING_THRESHOLD = 0.01

# train
C = 0.4

# detect
FIND_THRESHOLD = 0.2
IOU_THRESHOLD = 0.7
DETECT_THRESHOLD = 0.1
