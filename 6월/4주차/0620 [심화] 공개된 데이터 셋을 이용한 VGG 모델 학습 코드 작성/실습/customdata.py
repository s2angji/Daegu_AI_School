import numpy as np
from sklearn.metrics import f1_score
import torch.nn as nn
from torchvision.models import resnet18

import matplotlib.pyplot as plt
import PIL
import torchvision.transforms as transforms
import pathlib
import torchvision.utils
import pandas as pd
import torch
from tkinter.messagebox import NO
from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.datapipes.iter import batch
from torch.utils.data.sampler import Sampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import PIL.Image
import torchvision.transforms.functional as TF


SEED_NUMBER = 7777
LABEL_MAP = {
    0: "Nuclepolasm",
    1: "Nuclear membrane",
    2: "Nucleoli",
    3: "Nucleoli fibrillar center",
    4: "Nuclear speckles",
    5: "Nuclear bodies",
    6: "Endoplasmic reticulum",
    7: "Golgi apparatus",
    8: "Peroxisomes",
    9: "Endosomes",
    10: "Lysosomes",
    11: "Intermediate filaments",
    12: "Action filaments",
    13: "Focal adhesion ends",
    14: "Microtubules",
    15: "Microtubules ends",
    16: "Cytokinetic bridge",
    17: "Mitotic spindle",
    18: "Microtubule organizing center",
    19: "Centrosome",
    20: "Lipid droplets",
    21: "Plasma membranc",
    22: "Cell junctions",
    23: "Mitochondria",
    24: "Aggresome",
    25: "Cytosol",
    26: "Cytoplasmic bodies",
    27: "Rods & rings"
}


class AdjustedGamma(object):
    def __call__(self, img):
        return TF.adjust_gamma(img, 0.8, gain=1)


image_transform = transforms.Compose([
    AdjustedGamma(),
    transforms.ToTensor()
])


class MulitiBandMultiLabelDataset(Dataset):
    BAND_NAMES = ["_red.png", "_green.png", "_blue.png", "_yellow.png"]

    def __init__(self, images_df, base_path, image_transform=image_transform, augmentation=None):
        if not isinstance(base_path, pathlib.Path):
            base_path = pathlib.Path(base_path)

        self.images_df = images_df.copy()
        self.image_transform = image_transform
        self.augmentation = augmentation
        self.images_df.Id = self.images_df.Id.apply(lambda x: base_path / x)
        self.mlb = MultiLabelBinarizer(classes=list(LABEL_MAP.keys()))

    def __getitem__(self, index):
        x = self._load_multiband_image(index)
        y = self._load_multiband_target(index)

        if self.augmentation is not None:
            x = self.augmentation(x)

        x = self.image_transform(x)

        return x, y

    def __len__(self):
        return len(self.images_df)

    def _load_multiband_image(self, index):
        row = self.images_df.iloc[index]
        image_bands = []
        for band_name in self.BAND_NAMES:
            p = str(row.Id.absolute()) + band_name
            pil_channel = PIL.Image.open(p)
            image_bands.append(pil_channel)

        # band3image = pil_channel.convert("RGB")
        band4image = PIL.Image.merge("RGBA", bands=image_bands)
        return band4image

    def _load_multiband_target(self, index):
        return list(map(int, self.images_df.iloc[index].Target.split(' ')))

    # def collate_func(self, batch):
    #     images = [x[0] for x in batch]
    #     labels = [x[1] for x in batch]

    #     labels_one_hot = self.mlb.fit_transform(labels)

    #     return torch.stack(images), torch.FloatTensor(labels_one_hot)


df = pd.read_csv("./data/train.csv")
df_train, df_val = train_test_split(df, test_size=.2, random_state=SEED_NUMBER)
train_data = MulitiBandMultiLabelDataset(
    df_train, base_path="./data/train", image_transform=image_transform)


# for i in range(1):
#     sample, _ = train_data[0]

# plt.figure(figsize=(10, 10))
# plt.imshow(transforms.ToPILImage()(sample))
# plt.show()
image_channels = 4
model = resnet18(predictions=True)
for p in model.parameters():
    p.requires_grad = True
inft = model.fc.in_features
model.fc = nn.Linear(in_features=inft, out_features=28)
model.avgpool = nn.AdaptiveAvgPool2d(1)
model.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7,
                        stride=2, padding=3, bias=False)

"""평가"""
# 함수 인자 값 : model, test_loader, threshold
threshold = 0.2
all_preds = []
true = []
model.eval()
for b in test_loader:
    x, y = b
    if torch.cuda.is_available():
        x, y = x.cuda(), y.cuda()
    pred = model(x)
    all_preds.append(pred.sigmoid().cpu().data.numpy())
    true.append(y.cpu().data.numpy)

p = np.concatenate(all_preds)
R = np.concatenate(true)

f1 = f1_score(P > threshold, R, average="macro")
print(f1)
