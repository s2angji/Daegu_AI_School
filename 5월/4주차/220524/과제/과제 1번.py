import random
import numpy as np
import os
import cv2
import glob
from PIL import Image
import PIL.ImageOps
# PIL -> pip install Pillow
# cv2 -> pip install opencv-python

# 새로만들 이미지 갯수를 정합니다.
num_augmented_images = 50

# 원본 사진 폴더 경로
file_path = "./data"

# 위의 폴더 내부에 있는 이미지 이름의 배열이 저장 되는 형태
file_name = os.listdir(file_path)
print(file_name)

# file_name 길이를 가져오겠습니다.
total_origin_image_num = len(file_name)
print("total image number >> ", total_origin_image_num)
# total image number >>  3

augment_cnt = 1

for i in range(1, num_augmented_images):
    # image = [image01 , image02 , image03]
    change_picture_index = random.randint(0, total_origin_image_num-1)
    # print(change_picture_index)
    file_names = file_name[change_picture_index]
    # print(file_names)

    os.makedirs("./custom_data", exist_ok=True)
    origin_image_path = "./data/" + file_names
    # print(origin_image_path)

    image = Image.open(origin_image_path)

    # 랜덤 값이 1~4 사이으 값이 나오도록 1 2 3
    random_augment = random.randrange(4, 7)
    print(random_augment)

    if (random_augment == 1):
        # 이미지 좌우 반전
        inverted_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        inverted_image.save("./custom_data/" + "inverted_" +
                            str(augment_cnt) + ".png")
        pass
    elif (random_augment == 2):
        rotated_image = image.rotate(random.randrange(-20, 20))
        rotated_image.save("./custom_data/" + "rotated_" +
                           str(augment_cnt) + ".png")
        # 이미지 기울기
    elif (random_augment == 3):
        resize_image = image.resize(size=(224, 224))
        resize_image.save("./custom_data/" + "resize_" +
                          str(augment_cnt) + ".png")
        # 이미지 리사이즈
    # 3가지 추가 해주시면 됩니다.
    # 상하, 컬러지터, 그레이, 센터 크롭, 크롭, 랜덤 크롭
    elif (random_augment == 4): # 상하
        inverted_top_bottom_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        inverted_top_bottom_image.save("./custom_data/" + "inverted_top_bottom_" +
                            str(augment_cnt) + ".png")
    elif (random_augment == 5): # 색상 변환
        np_image = np.array(image)
        cvt_color_image = cv2.cvtColor(np_image, (cv2.COLOR_BGR2GRAY, cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2RGB, cv2.COLOR_BGR2LAB)[random.randint(0, 3)])
        pil_image = Image.fromarray(cvt_color_image)
        pil_image.save("./custom_data/" + "cvt_color_" +
                          str(augment_cnt) + ".png")
    elif (random_augment == 6): # 크롭
        x = random.randint(0, image.size[0] // 3)
        y = random.randint(0, image.size[1] // 3)
        width = random.randint(x + image.size[0] // 3, image.size[0] - 1)
        height = random.randint(y + image.size[1] // 3, image.size[1] - 1)
        crop_image = image.crop((x, y, width, height))
        crop_image.save("./custom_data/" + "crop_image_" +
                          str(augment_cnt) + ".png")

    # 제출 : 코드만 주시면 됩니다.
    augment_cnt += 1