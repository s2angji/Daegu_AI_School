""" bbox aug 적용되는 지 확인 코드"""
from PIL import Image
import cv2
import numpy as np
import time
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from matplotlib import pyplot as plt
import os
import random
import data_exploration as data_ex
import albumentations as albumentations
from albumentations.pytorch import ToTensorV2


'''
ip install albumentations 설치 후 CV2 최신버전으로 올라감.. 특정 버전에서 ToTensorV2 안되는것을 확인 

최신버전 올라간 cv2 삭제 후 아래 버전으로 다시 설치 필수

cv2 버전 
pip uninstall opencv-contrib-python opencv-python opencv-python-headless
pip install opencv-contrib-python==4.5.5.64 opencv-python==4.5.5.62 opencv-python-headless==4.5.5.64
'''


class TorchvisionDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.imgs = list(sorted(os.listdir(self.path)))
        self.transform = transform

    def __getitem__(self, index):
        file_image = self.imgs[index]
        file_label = self.imgs[index][:-3] + 'xml'
        img_path = os.path.join(self.path, file_image)

        if 'test' in self.path:
            label_path = os.path.join("test_annotations\\", file_label)
        else:
            label_path = os.path.join("annotations\\", file_label)

        img = Image.open(img_path).convert("RGB")

        target = data_ex.generate_target(label_path)

        start_t = time.time()
        if self.transform:
            img = self.transform(img)

        total_time = (time.time() - start_t)

        return img, target, total_time

    def __len__(self):
        return len(self.imgs)


class AlbumentationsDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.imgs = list(sorted(os.listdir(self.path)))
        self.transform = transform

    def __getitem__(self, index):
        file_image = self.imgs[index]
        file_label = self.imgs[index][:-3] + 'xml'
        img_path = os.path.join(self.path, file_image)

        if 'test' in self.path:
            label_path = os.path.join("test_annotations\\", file_label)
        else:
            label_path = os.path.join("annotations\\", file_label)

        """Read an images with OpenCV"""
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = data_ex.generate_target(label_path)

        start_t = time.time()
        # if self.transform:
        #     img = self.transform(img)
        to_tensor = torchvision.transforms.ToTensor()

        if self.transform:
            transformed = self.transform(
                image=image, bboxes=target['boxes'], labels=target['labels'])
            image = transformed['image']
            target = {'boxes': transformed['bboxes'],
                      'labels': transformed['labels']}

        total_time = (time.time() - start_t)

        # image = to_tensor(img)

        return image, target

    def __len__(self):
        return len(self.imgs)


torchvision_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomCrop(224),
    transforms.ColorJitter(brightness=0.15, contrast=0.15,
                           saturation=0.2, hue=0.18),
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.ToTensor()
])

bbox_transform = albumentations.Compose([
    albumentations.HorizontalFlip(p=1),
    albumentations.Rotate(p=1),
    ToTensorV2(),
], bbox_params=albumentations.BboxParams(
    format='pascal_voc', label_fields=['labels']))

bbox_transform_dataset = AlbumentationsDataset(
    path=".\\images\\", transform=bbox_transform
)

torchvision_dataset = TorchvisionDataset(
    path=".\\images\\", transform=torchvision_transform)


only_totensor = transforms.ToTensor()

torchvision_dataset_no_transform = TorchvisionDataset(
    path=".\\images\\", transform=only_totensor)

img, annot, transform_time = torchvision_dataset_no_transform[0]

print("transform 적용 전 ")
data_ex.plot_image(img, annot)

img, annot, transform_time = torchvision_dataset[0]
print("transform 적용 후 ")
data_ex.plot_image(img, annot)

img, annot = bbox_transform_dataset[0]
print(img)


print("bbox 적용")
data_ex.plot_image(img, annot)
