import torch.nn as nn
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import cv2
import numpy as np
import torch
import torchvision
# import albumentations as albumentations
import time

"""프로세스 상태정보 보여 주는것과 동일"""
from tqdm import tqdm
from math import gamma
from bs4 import BeautifulSoup
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
# from albumentations.pytorch import ToTensorV2


def generate_box(obj):
    """
    <xmin>79</xmin>
    <ymin>105</ymin>
    <xmax>109</xmax>
    <ymax>142</ymax>
    """
    xmin = float(obj.find("xmin").text)
    ymin = float(obj.find('ymin').text)
    xmax = float(obj.find('xmax').text)
    ymax = float(obj.find('ymax').text)

    return [xmin, ymin, xmax, ymax]


def generate_label(obj):
    # <name>with_mask</name>
    """label info -> mask_weared_incorrect -> 2, with_mask -> 1, without_mask -> 0 """

    if obj.find("name").text == 'with_mask':
        return 1
    elif obj.find("name").text == 'mask_weared_incorrect':
        return 2

    return 0


def generate_target(file):
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, "html.parser")
        objects = soup.find_all("object")
        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))

        """fix code """
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        return target


def plot_image_from_output(img, annotation):

    # img = mping.imread(img_path)

    # 텐서 이미지 -> 이미지 화 처리
    img = img.cpu().permute(1, 2, 0)

    """img show"""
    # fig, ax = plt.subplots(1)
    # ax.imshow(img)

    rects = []

    for idx in range(len(annotation['boxes'])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]

        if annotation['labels'][idx] == 0:
            rect = patches.Rectangle(
                (xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r', facecolor='none'
            )
        elif annotation['labels'][idx] == 1:
            rect = patches.Rectangle(
                (xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='g', facecolor='none'
            )
        else:
            rect = patches.Rectangle(
                (xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='b', facecolor='none'
            )
        rects.append(rect)
    """image show"""
    #     ax.add_patch(rect)

    # plt.show()

    return img, rects


class MaskDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.img = list(sorted(os.listdir(self.path)))
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        file_image = self.img[index]

        file_label = self.img[index][:-3] + 'xml'
        img_path = os.path.join(self.path, file_image)

        # print("file_image ", file_image)
        # print("file_label ", file_label)
        # print("img_path ", img_path)

        if 'test' in self.path:
            label_path = os.path.join(".\\test_annotations\\", file_label)

        else:
            label_path = os.path.join(".\\annotations\\", file_label)

        """cv2 image read"""
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = generate_target(label_path)

        to_tensor = torchvision.transforms.ToTensor()

        if self.transform:
            transformed = self.transform(
                image=image, bboxes=target['boxes'], labels=target['labels'])
            image = transformed['image']
            # target = {'boxes': transformed['bboxes'],
            #           'labels': transformed['labels']}
        else:
            image = to_tensor(image)
        """image -> tensor"""

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# bbox_transform = albumentations.Compose([
#     albumentations.HorizontalFlip(),
#     albumentations.Rotate(p=0.8),
#     ToTensorV2()
# ], bbox_params=albumentations.BboxParams(
#     format='pascal_voc', label_fields=['labels']
# ))
# bbox_transform_test = albumentations.Compose([
#     ToTensorV2()
# ], bbox_params=albumentations.BboxParams(
#     format='pascal_voc', label_fields=['labels']
# ))


train_dataset = MaskDataset(".\\images\\")
test_dataset = MaskDataset(".\\test_images\\")

train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)


"""모델 호출"""
retina = torchvision.models.detection.retinanet_resnet50_fpn(
    num_classes=3, pretrained=False, pretrained_backbone=True
)

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

num_epochs = 2
retina.to(device)

"""gradinet calculation 이 필요한 params 만 추출"""
params = [p for p in retina.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params, lr=0.0025, momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.1)

len_dataloader = len(train_loader)

"""train loop"""
# for epoch in range(num_epochs):
#     start = time.time()
#     retina.train()
#
#     i = 0
#     epoch_loss = 0
#     for index, (images, targets) in enumerate(train_loader):
#         images = list(image.to(device) for image in images)
#
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#         outputs = retina(images, targets)
#         optimizer.zero_grad()
#         losses = sum(loss for loss in outputs.values())
#
#         i += 1
#
#         losses.backward()
#         optimizer.step()
#
#         if index % 10 == 0:
#             print("loss >>", losses.item(), "epoch >> ", epoch, "index >> ", index,
#                   f"time : {time.time() - start}")
#
#             torch.save(retina.state_dict(),
#                        f".\\retina_{num_epochs}_{epoch}.pt")
#
#     if lr_scheduler is not None:
#         lr_scheduler.step()
# torch.save(retina.state_dict(), f".\\retina_last.pt")

"""eval loop"""
retina.load_state_dict(torch.load(f'retina_last.pt', map_location='cpu'))

"""
make_prediction 함수에는 학습된 딥러닝 모델을 활용해 예측하는 알고리즘이 저장돼 있습니다. 
threshold 파라미터를 조정해 신뢰도가 일정 수준 이상의 바운딩 박스만 선택합니다. 
보통 0.5 이상인 값을 최종 선택합니다. 
"""


def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)):
        idx_list = []
        for idx, score in enumerate(preds[id]['scores']):
            if score > threshold:  # threshold 넘는 idx 구함
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    print("pred info ", preds)
    return preds


labels = []
preds_adj_all = []
annot_all = []

for im, annot in tqdm(test_loader, position=0, leave=True):
    im = list(img.to(device) for img in im)
    #annot = [{k: v.to(device) for k, v in t.items()} for t in annot]

    for t in annot:
        labels += t['labels']

    with torch.no_grad():
        preds_adj = make_prediction(retina, im, 0.5)
        preds_adj = [{k: v.to(torch.device('cpu'))
                      for k, v in t.items()} for t in preds_adj]

        print("make_prediction function 다음 처리 되는 값 >> ", preds_adj)

        preds_adj_all.append(preds_adj)
        annot_all.append(annot)


"""바운딩 박스 시각화 그리는 곳"""
nrows = 10
ncols = 2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4, nrows*4))

batch_i = 0
for im, annot in test_loader: # DataLoader의 배치 사이즈에 주의할 것, batch_i
    pos = batch_i * 4 + 1
    for sample_i in range(len(im)):

        img, rects = plot_image_from_output(im[sample_i], annot[sample_i])
        axes[(pos)//2, 1-((pos) % 2)].imshow(img)
        for rect in rects:
            axes[(pos)//2, 1-((pos) % 2)].add_patch(rect)

        img, rects = plot_image_from_output(
            im[sample_i], preds_adj_all[batch_i][sample_i])
        axes[(pos)//2, 1-((pos+1) % 2)].imshow(img)
        for rect in rects:
            axes[(pos)//2, 1-((pos+1) % 2)].add_patch(rect)

        pos += 2

    batch_i += 1
    if batch_i == 4:
        break

# xtick, ytick 제거
for idx, ax in enumerate(axes.flat):
    ax.set_xticks([])
    ax.set_yticks([])

colnames = ['True', 'Pred']

for idx, ax in enumerate(axes[0]):
    ax.set_title(colnames[idx])

plt.tight_layout()
plt.show()
