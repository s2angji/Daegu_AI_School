import glob
import os
from enum import Enum
from PIL import Image
from torch.utils.data import Dataset

from configer import *


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
