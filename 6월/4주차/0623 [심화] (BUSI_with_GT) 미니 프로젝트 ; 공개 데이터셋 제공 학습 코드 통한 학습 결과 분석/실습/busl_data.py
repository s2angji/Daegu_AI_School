import torchvision.models as models
from re import X
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm

device = torch.device("mps")
CLASS_NAME = {"benign": 0, "malignant": 1, "normal": 2}


device = torch.device("mps")

"""https://tutorials.pytorch.kr/beginner/fineturing_torchvision_models_tutorial.html"""


def initialize_model(model_name, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "mobilenet":
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        num_ftrs = model_ft.last_channel
        print("num_ftrs = ", num_ftrs)

        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


data_path = "./Dataset_BUSI_with_GT"
data_dir = os.listdir(data_path)

files = []  # save all images for each folder
labels = []  # set for each images for image the name of it

"""read files for each directory"""

for folder in data_dir:
    fileList = glob.glob(os.path.join(data_path, folder, "*"))
    labels.extend([folder for l in fileList])
    files.extend(fileList)

# print(len(files), len(labels))

"""create two list to hold only non-mask images and labels for each one"""
selected_files = []
selected_labels = []


for file, label in zip(files, labels):
    if 'mask' not in file:
        selected_files.append(file)
        selected_labels.append(label)

# print("select file len >> ", len(selected_files))
# print("select labels len >> ", len(selected_labels))

images = {
    'image': [],
    'target': []
}
print("Preparing the image ... ")

for i, (file, label) in enumerate(zip(selected_files, selected_labels)):
    images["image"].append(file)
    images["target"].append(label)


x_train, x_test, y_train, y_test = train_test_split(
    images["image"], images["target"], test_size=0.10)

train_data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
])

val_data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
])


class Mycustom(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]
        images = Image.open(data).convert("RGB")
        label_temp = CLASS_NAME[label]

        # augmentation 적용
        if self.transform is not None:
            images = self.transform(images)

        return images, label_temp

    def __len__(self):
        return len(self.x)


train_data = Mycustom(x_train, y_train, transform=train_data_transform)
test_data = Mycustom(x_test, y_test, transform=val_data_transform)

train_data_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=32, shuffle=False)


"""model 가져와야조 !!"""
model, _ = initialize_model(model_name="mobilenet", num_classes=3)
net = model.to(device)

"""하이퍼파레메타 값 가져오기"""
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=4, gamma=0.1)


num_epochs = 10
val_every = 1
save_weights_dir = "./weights_busl_moblienet"
os.makedirs(save_weights_dir, exist_ok=True)

"""학습 돌려보기"""


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


"""model load => model test"""
model.load_state_dict(torch.load(
    "./weights_busl_moblienet/best.pt"))
if __name__ == "__main__":
    # train(num_epochs, model, train_data_loader, test_data_loader,
    #       criterion, optimizer, save_weights_dir, val_every, device)
    eval(model, test_data_loader, device)
