import os
import glob

from torch.utils.data import Dataset
from PIL import Image
import time


class TorchvisionDataset(Dataset):
    def __init__(self, data_path, labels, transform=None):
        self.data_paths = sorted(glob.glob(os.path.join(data_path, '*', '*.png')))
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        __label = self.labels.index(self.data_paths[index].split('\\')[1])

        # Image open
        __image = Image.open(data_path)

        start_t = time.time()
        if self.transform:
            __image = self.transform(__image)
        __total_time = (time.time() - start_t)

        return __image, __label, __total_time

    def __len__(self):
        return len(self.data_paths)
