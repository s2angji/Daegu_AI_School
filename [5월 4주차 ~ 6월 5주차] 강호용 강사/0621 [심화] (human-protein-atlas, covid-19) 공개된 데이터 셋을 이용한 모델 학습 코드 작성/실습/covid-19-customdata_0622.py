"""data -> covid, normal, Viral Pneumonia"""
"""
train
 |
 Covid
 Normal
 Viral Peneumonia 
test
 |
 Covid
 Normal
 Viral Peneumonia 
"""


from h11 import Data
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch import mode, optim, save
import glob
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
device = torch.device("cpu")


class CustomDataset(Dataset):
    def __init__(self, data_path, mode, transform=None):
        """ init : 초기값 설정 """
        """데이터 가져오기 전체 데이터 경로 불러오기"""
        self.all_data = sorted(
            glob.glob(os.path.join(data_path, mode, "*", "*")))
        """data_path > ./dataset/ mode -> train * -> Downdog *-> 000000.jpg """
        self.transform = transform

    def __getitem__(self, index):
        data_path = self.all_data[index]
        # data_path info >>  ./Covid19-dataset/train/Covid/01.jpeg
        data_path_split = data_path.split("/")
        labels_temp = data_path_split[3]

        label = 0
        if "Covid" == labels_temp:
            label = 0
        elif "Normal" == labels_temp:
            label = 1
        elif "Viral Pneumonia" == labels_temp:
            label = 3

        image = Image.open(data_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.all_data)


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_data = CustomDataset(
    "./Covid19-dataset/", "train", transform=image_transform)
test_data = CustomDataset("./Covid19-dataset/", "test",
                          transform=image_transform)
train_load = DataLoader(train_data, batch_size=34, shuffle=True)
test_load = DataLoader(test_data, batch_size=34, shuffle=False)
# data loader
# 기울기 계산 함수, 옵티마이저, 활성함수
# 하이퍼 파라미터 값
# train loop val loop save

"""Custom data 까지 진행 나머지"""


def train(num_epoch, model, train_loader, test_loader, criterion, optimizer,
          save_dir, val_every, device):

    print("String... train !!! ")
    best_loss = 9999
    for epoch in range(num_epoch):
        for i, (imgs, labels) in enumerate(train_loader):
            print(imgs, labels)
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(output, 1)
            acc = (labels == argmax).float().mean()

            print("Epoch [{}/{}], Step [{}/{}], Loss : {:.4f}, Acc : {:.2f}%".format(
                epoch + 1, num_epoch, i +
                1, len(train_loader), loss.item(), acc.item() * 100
            ))

            if (epoch + 1) % val_every == 0:
                avg_loss = validation(
                    epoch + 1, model, test_loader, criterion, device)
                if avg_loss < best_loss:
                    print("Best prediction at epoch : {} ".format(epoch + 1))
                    print("Save model in", save_dir)
                    best_loss = avg_loss
                    save_model(model, save_dir)

    save_model(model, save_dir, file_name="last.pt")


def validation(epoch, model, test_loader, criterion, device):
    print("Start validation # {}".format(epoch))
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        for i, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total += imgs.size(0)
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()
            total_loss += loss
            cnt += 1
        avg_loss = total_loss / cnt
        print("Validation # {} Acc : {:.2f}% Average Loss : {:.4f}%".format(
            epoch, correct / total * 100, avg_loss
        ))

    model.train()
    return avg_loss


def save_model(model, save_dir, file_name="best.pt"):
    output_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), output_path)


def eval(model, test_loader, device):
    print("Starting evaluation")
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for i, (imgs, labels) in tqdm(enumerate(test_loader)):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            # 점수가 가장 높은 클래스 선택
            _, argmax = torch.max(outputs, 1)
            total += imgs.size(0)
            correct += (labels == argmax).sum().item()

        print("Test acc for image : {} ACC : {:.2f}".format(
            total, correct / total * 100))
        print("End test.. ")


def get_model(n_classes, image_channels=3):
    # resnet 18
    model = resnet18(pretrained=True)
    for p in model.parameters():
        p.requires_grad = True
    inft = model.fc.in_features
    """self.fc = nn.Linear(512 * block.expansion, num_classes)"""
    model.fc = nn.Linear(in_features=inft, out_features=n_classes)
    model.conv1 = nn.Conv2d(
        image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    return model
    # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
    # padding=3, bias=False)


model = get_model(3, 3)
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                       model.parameters()), lr=0.00025)

os.makedirs("./weights", exist_ok=True)
val_every = 1
save_weights_dir = "./weights"
num_epochs = 10

# """model load => model test"""
# model.load_state_dict(torch.load(
#     "./weights/best.pt"))


if __name__ == "__main__":
    train(num_epochs, model, train_load, test_load, criterion, optimizer, save_weights_dir,
          val_every, device)
    # eval(model, test_load, device)
