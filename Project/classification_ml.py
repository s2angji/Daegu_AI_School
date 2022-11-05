import os
import glob
import time
import copy
from enum import Enum
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from PIL import Image


device = 'cuda' if torch.cuda.is_available() else 'cpu' # cuda 사용 유무
# 하이퍼파라메타 값 세팅
batch_size = 36
num_epochs = 10
val_every = 1   # 평가 간격
data_path = '.\\dataset\\kfood'
nc = len(os.listdir(os.path.join(data_path, os.listdir(data_path)[0]))) # num_classes
lr = 0.025 # learning rate
criterion = nn.CrossEntropyLoss().to(device)
# 모델 가중치 저장할 폴더 생성
save_weights_dir = '.\\weights'
os.makedirs(save_weights_dir, exist_ok=True)
# matplotlib 한글 폰트 사용을 위해서 세팅
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


# 훈련(train) 또는 테스트
class Mode(Enum):
    __dir = os.listdir(data_path)
    train = __dir[0] if 'train' in __dir[0].lower() else __dir[1]
    val = __dir[1 - __dir.index(train)]


class CustomDataset(Dataset):
    def __init__(self, data_path, mode: Mode, transform=None):
        self.labels = os.listdir(os.path.join(data_path, mode.value))
        self.all_data = sorted(glob.glob(os.path.join(data_path, mode.value, '*', '*.jpg')))
        self.all_data_dict = {label: sorted(glob.glob(os.path.join(data_path, mode.value, str(label), '*.jpg')))
                              for label in self.labels}
        self.transform = transform

    def __getitem__(self, index):
        images = Image.open(self.all_data[index]).convert('RGB')
        if self.transform is not None:
            images = self.transform(images)

        return images, self.labels.index(self.all_data[index].split('\\')[-2])

    def __len__(self):
        return len(self.all_data)


def initialize_model(model_name, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        # Resnet18
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        # Alexnet
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        # VGG11_bn
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        # Squeezenet
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        # Densenet
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        # Inception v3
        # Be careful, expects (299,299) sized images and has auxiliary output
        model_ft = models.inception_v3(pretrained=use_pretrained)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print(f"{model_name} Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# data augmentation 함수
def data_augmentation():
    data_transform = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(p=0.4),
            # transforms.RandomVerticalFlip(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
        ])
    }
    return data_transform


def train_model(model, data_loaders, criterion, optimizer, scheduler, num_epochs, signal):
    since = time.time()

    history = []
    train_loss = 0
    valid_loss = 0
    train_acc = 0
    valid_acc = 0

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    progress_step = 100 / num_epochs
    for epoch in range(num_epochs):
        signal.emit(epoch * progress_step, '\nEpoch {}/{}\n────────────────────────\n'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)

            signal.emit(epoch * progress_step,
                        '[{}] Loss: {:.4f} Acc: {:.4f}\n'.format(phase.rjust(4).ljust(5), epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                train_loss = epoch_loss
                train_acc = epoch_acc
            else:
                valid_loss = epoch_loss
                valid_acc = epoch_acc

        history.append([train_loss, valid_loss, train_acc, valid_acc])
        scheduler.step()

    time_elapsed = time.time() - since
    signal.emit(100, '\n\nTraining complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    signal.emit(100, 'Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])


def visualize_loss_acc(history, optim):
    plt.figure(figsize=(20, 10))

    plt.suptitle(str(optim))

    plt.subplot(121)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['valid_loss'], label='valid_loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(122)
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['valid_acc'], label='valid_acc')
    plt.legend()
    plt.title('Accuracy Curves')

    plt.show()
    # plt.savefig(str(path) + 'loss_acc.png')


def visualize_predict(model, data):
    c = np.random.randint(0, len(data))
    img, label = data[c]

    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        # out = model(img.view(1, 3, 224, 224).cuda())
        out = model(img.view(1, 3, 224, 224).cpu())
        out = torch.exp(out)
        # print(out)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img.numpy().transpose((1, 2, 0)).astype(np.uint8))
    plt.title(data.labels[label])
    plt.subplot(122)
    plt.barh(data.labels, out.cpu().numpy()[0])
    plt.show()


def visualize_class_accuracies(model, data):
    accuracy_dict = {}
    with torch.no_grad():
        model.eval()
        for c in data.all_data_dict.keys():
            total_count = len(data.all_data_dict[c])
            correct_count = 0
            for path in data.all_data_dict[c]:
                # print(path)
                im = Image.open(path).convert('RGB')
                # im.show()
                im = transforms.ToTensor()(im)
                im = transforms.Resize((224, 224))(im)
                if device == 'cuda':
                    out = model(im.view(1, 3, 224, 224).cuda())
                else:
                    out = model(im.view(1, 3, 224, 224).cpu())
                # print(out)
                out = torch.exp(out)
                pred = list(out.cpu().numpy()[0])
                # print(pred)
                pred = pred.index(max(pred))
                # print(pred, data.labels.index(c))

                if pred == data.labels.index(c):
                    correct_count += 1

            print(f"Accuracy for class {c} : ", correct_count / total_count)
            accuracy_dict[c] = correct_count / total_count

    plt.figure(figsize=(10, 5))
    plt.title('Class_accuracies')
    plt.barh(list(accuracy_dict.keys()), list(accuracy_dict.values()))
    plt.show()

    return accuracy_dict


def save_model(model, file_name="best.pt", save_dir=save_weights_dir):
    output_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), output_path)
