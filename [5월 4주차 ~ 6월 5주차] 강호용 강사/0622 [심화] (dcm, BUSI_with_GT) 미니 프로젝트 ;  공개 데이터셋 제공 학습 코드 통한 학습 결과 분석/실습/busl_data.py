from re import X
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt


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


class Mycustom(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]
        images = Image.open(data).convert("RGB")

        label_temp = 0
        if label == "benign":
            label_temp = 0
        elif label == "malignant":
            label_temp = 1
        elif label == "normal":
            label_temp = 2

        return images, label_temp

    def __len__(self):
        return len(self.x)


train_data = Mycustom(x_train, y_train, transform=None)
