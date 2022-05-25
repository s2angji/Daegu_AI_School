import os
import glob

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import json
import cv2
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import numpy as np
import torch

class MyCustomDatasetImage(Dataset):
    def __init__(self, data_path, json_path):
        # data
        self.all_data = sorted(glob.glob(os.path.join(data_path, '*.jpg')))

        # json
        with open(json_path, 'r') as file:
            self.json = json.load(file)
        assert len(self.json) > 0, '파일 읽기 실패'

        # categories
        self.cate_info = {}
        for category in self.json['categories']:
            self.cate_info[category['id']] = category['name']

        # annotations
        self.ann_info = {}
        annotation = self.json['annotations']
        for annotation in self.json['annotations']:
            self.ann_info[annotation['image_id']] = {
                'boxes'         : annotation['bbox'],
                'category_id'   : annotation['category_id']
            }

        # image
        self.image_info = {}
        for image in self.json['images']:
            self.image_info[image['id']] = {
                'file_name' : image["file_name"],
                'height'    : image["height"],
                'width'     : image["width"]
            }

        # font
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1
        self.color = (255, 0, 0)
        self.thickness = 2

    def __getitem__(self, index):
        x, y, w, h = self.ann_info[index]['boxes']

        org_img = cv2.imread(self.all_data[index])
        org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

        # 바운드 박스 그리기
        copy_img = org_img.copy()
        text_img = cv2.putText(copy_img, self.cate_info[self.ann_info[index]['category_id']],
                               (x, y - 10), self.font, self.fontScale, self.color, self.thickness, cv2.LINE_AA)
        rec_img = cv2.rectangle(text_img, (x, y), (int(x + w), int(y + h)), (255, 0, 255), 2)

        # 바운드 박스 크기 만큼 Crop
        copy_img = org_img.copy()
        crop_img = copy_img[y:int(y + h), x:int(x + w)]
        crop_img = cv2.warpAffine(crop_img, np.float32([[1, 0, x], [0, 1, y]]), (org_img.shape[:2][0], org_img.shape[:2][1]))
        crop_img = cv2.resize(crop_img, (255, 255))

        # if index in (0, 1, 2, 3):
        #     cv2.imshow("test", crop_img)
        #     cv2.waitKey(0)

        return rec_img, crop_img

    def __len__(self):
        return len(self.all_data)

# 2가지 구성 필요 -> dataset dataloader
dataset = MyCustomDatasetImage(data_path=".\\data\\", json_path=".\\anno\\raccoon_annotations.coco.json")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

os.makedirs(".\\rec", exist_ok=True)
os.makedirs(".\\crop", exist_ok=True)
for i, (rec_img, crop_img) in enumerate(dataloader):
    rec_img = ToPILImage()(torch.squeeze(rec_img).numpy())
    crop_img = ToPILImage()(torch.squeeze(crop_img).numpy())
    # plt.imshow(rec_img)
    # plt.show()
    rec_img.save(f'.\\rec\\rec_{i}.png', 'png')
    crop_img.save(f'.\\crop\\crop_{i}.png', 'png')
