import os
import json
import cv2
import numpy as np
from math import log10, floor
import random

json_path = '.\\json'
img_path = '.\\img'
classes = []

if not os.listdir(json_path) == os.listdir(img_path):
    print('img, json 폴더 안의 class들이 같지 않습니다.')
    exit()
else:
    classes = os.listdir(json_path)

train_rate = 0.6
valid_rate = 0.2
test_rate = 0.2
train_path = '.\\dataset\\train'
valid_path = '.\\dataset\\valid'
test_path = '.\\dataset\\test'
os.makedirs(os.path.join(train_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_path, 'labels'), exist_ok=True)
os.makedirs(os.path.join(valid_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(valid_path, 'labels'), exist_ok=True)
os.makedirs(os.path.join(test_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(test_path, 'labels'), exist_ok=True)

with open('.\\data.yaml', 'w') as f:
    f.write('train: ./dataset/train/images' + '\n')
    f.write('val: ./dataset/valid/images' + '\n\n')
    f.write(f'nc: {len(classes)}' + '\n')
    f.write('names: [')
    for i in classes:
        if i == classes[-1]:
            f.write(f"'{i}']")
        else:
            f.write(f"'{i}', ")

for cls in classes:
    base_path = os.path.join(json_path, cls)
    with open(os.path.join(base_path, os.listdir(base_path)[0])) as f:
        j = json.load(f)

    assert len(j) > 0, '파일 읽기 실패'

    random.seed(7777)
    set_all = set(range(len(j)))
    idx_train = random.sample(set_all, int(len(j) * train_rate))
    set_train = set_all - set(idx_train)
    idx_valid = random.sample(set_train, int(len(set_all) * valid_rate))
    set_valid = set_all - set(idx_train) - set(idx_valid)
    idx_test = list(set_valid)

    zero_fill = floor(log10(len(j))) + 1
    print(cls, '의 총 이미지 수 : ', len(j))
    print(cls, '의 숫자 번호 매길 때 총 길이 : ', zero_fill)
    print('train 총 수 및 인덱스들', len(idx_train), idx_train)
    print('valid 총 수 및 인덱스들', len(idx_valid), idx_valid)
    print('test  총 수 및 인덱스들', len(idx_test), idx_test)

    for i, obj in enumerate(j):
        filename = obj['filename']
        bboxes = obj['annot_A']['boxes']
        minX, minY, maxX, maxY = int(bboxes['minX']), int(bboxes['minY']), int(bboxes['maxX']), int(bboxes['maxY'])

        img = np.fromfile(os.path.join(f'.\\img\\{cls}', filename), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        # img = cv2.rectangle(img, (minX, minY), (maxX, maxY), (207, 137, 8), 2)
        # img = cv2.putText(img, cls, (minX, minY - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (227, 220, 209), 2, cv2.LINE_AA)
        # cv2.imshow('test', img)
        # cv2.waitKey(0)

        yolo_filename = cls + str(i).zfill(zero_fill)
        yolo_width = (maxX - minX) / 2
        yolo_height = (maxY - minY) / 2
        yolo_centerX = minX + yolo_width
        yolo_centerY = minY + yolo_height
        # 정규화
        yolo_width, yolo_centerX = np.array([yolo_width, yolo_centerX]) / img.shape[0]
        yolo_height, yolo_centerY = np.array([yolo_height, yolo_centerY]) / img.shape[1]
        # 이미지 크기 줄이기
        img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)

        if i in idx_train:
            cv2.imwrite(os.path.join(train_path, 'images', yolo_filename + '.jpg'), img)
            with open(os.path.join(train_path, 'labels', yolo_filename + '.txt'), 'w') as f:
                f.write(f'{classes.index(cls)} {yolo_centerX:.6f} {yolo_centerY:.6f} {yolo_width:.6f} {yolo_height:.6f}')
        elif i in idx_valid:
            cv2.imwrite(os.path.join(valid_path, 'images', yolo_filename + '.jpg'), img)
            with open(os.path.join(valid_path, 'labels', yolo_filename + '.txt'), 'w') as f:
                f.write(f'{classes.index(cls)} {yolo_centerX:.6f} {yolo_centerY:.6f} {yolo_width:.6f} {yolo_height:.6f}')
        elif i in idx_test:
            cv2.imwrite(os.path.join(test_path, 'images', yolo_filename + '.jpg'), img)
            with open(os.path.join(test_path, 'labels', yolo_filename + '.txt'), 'w') as f:
                f.write(f'{classes.index(cls)} {yolo_centerX:.6f} {yolo_centerY:.6f} {yolo_width:.6f} {yolo_height:.6f}')

        if i % 100 == 0:
            print('filename : ', yolo_filename)
            print('x, y : %.6f %.6f' % (yolo_centerX, yolo_centerY))
            print('w, h : %.6f %.6f' % (yolo_width, yolo_height))
