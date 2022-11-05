import os
import cv2

image_root = r'C:\Users\user\ai_school\220516\steel_masking\image'
mask_root = r'C:\Users\user\ai_school\220516\steel_masking\mask'

for filename in os.listdir(image_root):
    # 파일명이 매칭되는 이미지/마스킹 가져오기
    image_path = os.path.join(image_root, filename)
    mask_path = os.path.join(mask_root, filename)
    
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    image_masked = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('image', img)
    cv2.imshow('mask', image_masked)
    if cv2.waitKey(0) & 0xff == ord('q'):
        cv2.destroyAllWindows()
        exit()