import customdata_0621

import pandas as pd
import torch
import os
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.models import resnet18
from torch import optim, save
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from statistics import mode

SEED_NUMBER = 7777
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AdjustedGamma(object):
    def __call__(self, img):
        return TF.adjust_gamma(img, 0.8, gain=1)


image_transform = transforms.Compose([
    AdjustedGamma(),
    transforms.ToTensor()
])

df = pd.read_csv('.\\data\\train.csv')
df_train, df_val = train_test_split(df, test_size=.2, random_state=SEED_NUMBER)
train_data = customdata_0621.MulitiBandMultiLabelDataset(
    df_train, base_path='.\\data\\train', image_transform=image_transform)
test_data = customdata_0621.MulitiBandMultiLabelDataset(
    df_val, base_path='.\\data\\train', image_transform=image_transform)
train_load = DataLoader(
    train_data, collate_fn=train_data.collate_func, batch_size=16, num_workers=6)
test_load = DataLoader(
    test_data, collate_fn=test_data.collate_func, batch_size=16, num_workers=6)


def train(num_epoch, model, train_loader, test_loader, criterion, optimizer,
          save_dir, val_every, device):

    print("String... train !!! ")
    best_loss = 9999
    for epoch in range(num_epoch):
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)

            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(output, 1)
            # acc = (labels == argmax).float().mean()

            print("Epoch [{}/{}], Step [{}/{}], Loss : {:.4f}".format(
                epoch + 1, num_epoch, i +
                1, len(train_loader), loss.item()
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
            # correct += (labels == argmax).sum().item()
            total_loss += loss
            cnt += 1
        avg_loss = total_loss / cnt
        print("Validation # {} Average Loss : {:.4f}%".format(
            epoch, correct / total * 100, avg_loss
        ))

    model.train()
    return avg_loss


def save_model(model, save_dir, file_name="best.pt"):
    output_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), output_path)


def get_model(n_classes, image_channels=4):
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


def evaluate(model, test_loader, threshold=0.2):
    all_preds = []
    true = []
    model.eval()

    for b in test_loader:
        image, target = b

        if torch.cuda.is_available():
            image, target = image.cuda(), target.cuda()
        else:
            image, target = image.to(device), target.to(device)

        pred = model(image)
        all_preds.append(pred.simoid().cpu().data.numpy())
        true.append(target.cpu().data.numpy())
    P = np.concatenate(all_preds)
    R = np.concatenate(true)

    f1 = f1_score(P > threshold, R, average='macro')

    return f1


model = get_model(28, 4)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                       model.parameters()), lr=0.00025)

os.makedirs('.\\weights', exist_ok=True)
val_every = 1
save_weights_dir = '.\\weights'
num_epochs = 1

if __name__ == '__main__':
    train(num_epochs, model, train_load, test_load, criterion, optimizer, save_weights_dir, val_every, device)
    customdata_0621.evaluator(model, test_load)

# net = get_model(28)
# net = net.to(device)
# criterion = nn.BCEWithLogitsLoss().to(device)
# optimizer = optim.Adam(filter(lambda  p: p.requires_grad, net.parameters()), lr=0.00005)
#
# os.makedirs('.\\weights', exist_ok=True)
# val_every = 1
# save_weights_dir = '.\weights'
# num_epochs = 10
#
# if __name__ == '__main__':
#
#     train(num_epochs, net, train_load, test_load, criterion, optimizer, save_weights_dir, val_every, device)
