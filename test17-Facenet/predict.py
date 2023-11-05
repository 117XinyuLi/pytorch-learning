from PIL import Image

from facenet import Facenet

# 输出结果为两张图片的距离（越小越相似）
if __name__ == "__main__":
    model = Facenet()
        
    while True:
        image_1 = input('Input image_1 filename:')
        try:
            image_1 = Image.open(image_1)
        except:
            print('Image_1 Open Error! Try again!')
            continue

        image_2 = input('Input image_2 filename:')
        try:
            image_2 = Image.open(image_2)
        except:
            print('Image_2 Open Error! Try again!')
            continue
        
        probability = model.detect_image(image_1, image_2)
        print(probability)
