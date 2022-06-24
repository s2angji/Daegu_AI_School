from cProfile import label
import glob
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np

from torch import optim, save
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import data_split

CLASS_NAME = {"O": 0, "R": 1}
device = torch.device("mps")
"""
CPU 사용자 
device = torch.device("cpu")

CUDA 사용자 
device = torch.device("cuda")
"""

"""Customdataset 구성"""


class Customdataset(Dataset):
    def __init__(self, data_path, transform=None):
        """정의"""
        self.data_path = data_path
        self.transform = transform

    def __getitem__(self, index):
        """데이터 경로에서 하나씩 데이터 가져오기"""
        path = self.data_path[index]
        """데이터 경로에서 폴더 명 가지고와서 라벨 변경"""
        path_split = path.split("/")
        label_temp = path_split[3]
        label = CLASS_NAME[label_temp]
        """이미지 오픈"""
        img = Image.open(path).convert("RGB")
        """Augmentation"""
        if self.transform is not None:
            img = self.transform(img)

        """라벨 이미지 리턴"""
        return img, label

    def __len__(self):
        return len(self.data_path)


"""augmentation 구성"""
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
])

"""데이터 셋 구성"""
train_data = Customdataset(
    data_path=data_split.x_train, transform=train_transform)
val_data = Customdataset(data_path=data_split.x_valid, transform=val_transform)

"""라벨 별 이미지 갯수"""
# train_one_label_cnt = 0  # O
# train_two_label_cnt = 0  # R

# for i in train_data:
#     image, labels = i
#     if labels == 0:
#         train_one_label_cnt += 1
#     elif labels == 1:
#         train_two_label_cnt += 1

# print(f"Train 라벨갯수  >> [{train_one_label_cnt}/{train_two_label_cnt}]")
"""데이터 로더 구성"""
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)


def train(num_epoch, model, train_loader, val_loader, criterion,
          optimizer, save_dir, val_every, device):
    print("String train... !! ")
    best_loss = 9999
    for epoch in range(num_epoch):
        for i, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device)
            """모델에 image & data 넣기 """
            output = model(image)
            """loss function"""
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(output, 1)
            acc = (label == argmax).float().mean()
            lr = optimizer.param_groups[0]['lr']
            print(
                "Epoch [{}/{}], lr [{}], Step [{}/{}], Loss : {:.4f}, accuracy : {:.2f}%"
                .format(
                    epoch + 1, num_epoch, lr, i +
                    1, len(train_loader), loss.item(),  acc.item() * 100
                ))

            """val loop call"""
            if (epoch + 1) % val_every == 0:
                avg_loss = validation(
                    epoch + 1, model, val_loader, criterion, device)

                if avg_loss < best_loss:
                    print("Best prediction at epoch {}".format(epoch + 1))
                    print("save model in", save_dir)
                    best_loss = avg_loss
                    save_model(model, save_dir)

    save_model(model, save_dir, file_name="last.pt")


def validation(epoch, model, val_loader, criterion, device):
    print("Start validation at epoch {}".format(epoch))
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0

        for i, (image, label) in enumerate(val_loader):
            image, label = image.to(device), label.to(device)
            """모델에 데이터를 넣어서 예측 값을 뽑기 """
            output = model(image)
            """loss function"""
            loss = criterion(output, label)
            total += image.size(0)
            _, argmax = torch.max(output, 1)
            correct += (label == argmax).sum().item()
            total_loss += loss
            cnt += 1

        """avg loss """
        avg_loss = total_loss / cnt
        print("Validation # {} Acc : {:.2f}% Average Loss : {:.4f}%".format(
            epoch, correct / total * 100, avg_loss
        ))

    model.train()

    return avg_loss


def save_model(model, save_dir, file_name="best.pt"):
    output = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), output)


def get_model(n_classes):
    model_ft = models.mobilenet_v2(pretrained=True)
    num_ft = model_ft.last_channel
    model_ft.classifier[1] = nn.Linear(num_ft, n_classes)

    return model_ft


""" 하이퍼 파라메타 지정 """
num_epoch = 20
val_every = 5
net = get_model(2)
net = net.to(device)


criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, net.parameters()), lr=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=4, gamma=0.1)

save_weights_dir = "./weights"
os.makedirs(save_weights_dir, exist_ok=True)

if __name__ == "__main__":
    train(num_epoch, net, train_loader, val_loader,
          criterion, optimizer, save_weights_dir, val_every, device)
