from PIL import Image


# 取图的最长边做正方形，然后将原图放在正方形上，后面再resize就不会变形
def make_416_image(path):
    image = Image.open(path)# 读取图片
    temp = max(image.size)# 取最长边
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))# 生成一个黑色的正方形
    mask.paste(image, (0, 0))# 将原图放在正方形上，沾到黑色的正方形左上角
    return mask




