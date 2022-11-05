import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import seaborn as sns

from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from timeit import default_timer as timer

classes = ['Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior2']


# 데이터 비율 확인 체크
def data_check(path=".\\dataset\\YogaPoses"):
    count_dit = {}
    for root, dirs, files in os.walk(path):
        # print(root)
        # print(root.split('\\')[-1])
        if files != [] and str(root.split("\\")[-1]) in classes:
            count_dit[str(root.split('\\')[-1])] = len(files)
    return count_dit


# counts = data_check()
# plt.bar(list(counts.keys()), list(counts.values()))
# plt.show()


# 데이터 train val
def data_split(path=".\\dataset\\YogaPoses", split_predictions=0.1):
    train_dict = {}
    val_dict = {}
    counts = data_check(path)
    for root, dirs, files in os.walk(path):
        if files != [] and str(root.split('\\')[-1]) in classes:
            file_paths = [os.path.join(root, files[i])
                          for i in range(len(files))]

            valid_index = np.random.randint(low=0, high=len(files), size=int(len(files)*split_predictions))
            train_index = list(set(range(0, len(files))) - set(valid_index))

            train_dict[str(root.split('\\')[-1])] = [file_paths[idx]
                                                     for idx in train_index]
            val_dict[str(root.split('\\')[-1])] = [file_paths[idx]
                                                   for idx in valid_index]

    return train_dict, val_dict


train_split_data, val_split_data = data_split()
print("Train data size : ", [len(l) for l in train_split_data.values()])
print("Val data size : ", [len(l) for l in val_split_data.values()])


# 3. custom dataset
class YogaPosesData(Dataset):

    # chess Piece dataset class

    def __init__(self, data_dict, transform=None):
        # Args : data_dict (dict)
        self.data_dict = data_dict
        self.transform = transform

    def __getitem__(self, idx):
        counts = [len(l) for l in self.data_dict.values()]
        # cumsum은 배열에서 주어진 축에 따라 누적되는 원소들의 누적 합을 계산하는 함수
        sum_counts = list(np.cumsum(counts))
        # 0, Downdog의 데이터 수(196), Goddess의 데이터 수(196 + 199), ... ,Infinitie (무한)
        sum_counts = [0] + sum_counts + [np.inf]

        for c, v in enumerate(sum_counts):
            if idx < v:
                i = (idx - sum_counts[c-1]) - 1
                break

        label = list(self.data_dict.keys())[c-1]
        img = Image.open(self.data_dict[str(label)][i]).convert("RGB")

        # data augmentation
        if self.transform:
            img = self.transform(img)

        return img, classes.index(str(label))

    def __len__(self):
        return sum([len(l) for l in self.data_dict.values()])


# 4. data augmentation
train_data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(),
    transforms.RandomAdjustSharpness(sharpness_factor=1.5),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor()
])
val_data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(),
    transforms.ToTensor()
])

# 5. data loader
# Custom dataset의 인스턴스 만들기
data_train = YogaPosesData(train_split_data, transform=train_data_transform)
data_val = YogaPosesData(val_split_data, transform=val_data_transform)
# Data loader의 인스턴스 만들기
train_loader = DataLoader(data_train, batch_size=10, shuffle=True)
val_loader = DataLoader(data_val, batch_size=10, shuffle=False)

# 6. train data val data check
t_idx = np.random.randint(0, len(data_train))
v_idx = np.random.randint(0, len(data_val))

print("Total number a train images >> ", len(data_train))
print("Val number a val images >> ", len(data_val))

t_img, t_label = data_train[t_idx]
v_img, v_label = data_val[v_idx]

# show train image check
plt.figure(figsize=(8, 5))
plt.subplot(121)
plt.imshow(t_img.numpy().transpose(1, 2, 0))
plt.title(f'Train data class = {classes[t_label]}')
plt.subplot(122)
plt.imshow(v_img.numpy().transpose(1, 2, 0))
plt.title(f'Val data class = {classes[v_label]}')
plt.show()

# Loss Function
criterion = nn.CrossEntropyLoss()

# device
# Windows, Linux 사용자
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Mac M1
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')
print("device info >> ", device)


# model chose
def base_model_build(device):

    # Load the pretrained model from pytorch
    vgg11 = models.vgg11(pretrained=True)
    # print(vgg11)
    for param in vgg11.features.parameters():
        param.requires_grad = False

    # (6): Linear(in_features=4096, out_features=1000, bias=True)
    # 총 5가지 라벨이므로, 마지막 6번째 Linear의 out_feautres가 5가 되는 Linear 인스턴스를 만듭니다.
    n_inputs = vgg11.classifier[6].in_features
    last_layer = nn.Linear(n_inputs, len(classes))

    # 마지막 6번째 classifier를 위에서 만들어진 인스턴스를 참조하도록 합니다.
    vgg11.classifier[6] = last_layer
    # print(vgg11)
    if device:
        print('training ...')
        vgg11.to(device)

    return vgg11


def loss_acc_visualize(history, optim, path):
    plt.figure(figsize=(20, 10))

    plt.suptitle(str(optim))

    plt.subplot(121)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.title("loss Curves")

    plt.subplot(122)
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    plt.title("ACC Curves")

    plt.savefig(str(path) + 'loss_acc.png')


def grad_visualize(history, optim, path, ylimit=10):

    # gradient norm distribution

    plt.figure(figsize=(20, 10))
    plt.suptitle(str(optim))
    plt.subplot(131)
    sns.kdeplot(weight_grads1, shade=True)
    sns.kdeplot(bias_grads1, shade=True)
    plt.legend(['weight', 'bias'])
    plt.title("Linear layer 1")
    plt.ylim(0, ylimit)

    plt.subplot(132)
    sns.kdeplot(weight_grads2, shade=True)
    sns.kdeplot(bias_grads2, shade=True)
    plt.legend(['weight', 'bias'])
    plt.title("Linear layer 2")
    plt.ylim(0, ylimit)

    plt.subplot(133)
    sns.kdeplot(weight_grads3, shade=True)
    sns.kdeplot(bias_grads3, shade=True)
    plt.legend(['weight', 'bias'])
    plt.title("Linear layer 3")
    plt.ylim(0, ylimit)

    plt.savefig(str(path) + "grad_norms.png")


def visual_predict(model, data=data_val):
    # 검증용 데이터 중에 하나 랜덤하게 선택
    c = np.random.randint(0, len(data))
    img, label = data[c]

    with torch.no_grad():
        model.eval()
        out = model(img.view(1, 3, 224, 224).to(device))
        out = torch.exp(out)
        print(out)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.title(str(classes[label]))
    plt.subplot(122)
    plt.barh(classes, out.cpu().numpy()[0])

    plt.show()


def class_accuracies(model, data_dict=val_split_data, classes=classes):
    accuracy_dic = {}
    with torch.no_grad():
        model.eval()

        for c in data_dict.keys():
            correct_count = 0
            total_count = len(data_dict[str(c)])
            gt = classes.index(str(c))

            for path in data_dict[str(c)]:
                im = Image.open(path).convert('RGB')

                im = transforms.ToTensor()(im)
                im = transforms.Resize((224, 224))(im)
                out = model(im.view(1, 3, 224, 244)).to(device)
                out = torch.exp(out)
                pred = list(out.cpu.numpy()[0])
                pred = pred.index(max(pred))

                if gt == pred:
                    correct_count += 1

            print(f"Acc for class {str(c)} : ", correct_count / total_count)
            accuracy_dic[str(c)] = correct_count / total_count

    return accuracy_dic
