import glob
import os
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_path, mode, transform=None):
        """ init : 초기값 설정 """
        """데이터 가져오기 전체 데이터 경로 불러오기"""
        self.all_data = sorted(
            glob.glob(os.path.join(data_path, mode, "*", "*.jpg")))
        """data_path > .\\dataset\\ mode -> train * -> Downdog *-> 000000.jpg """
        self.transform = transform

    def __getitem__(self, index):
        data_path = self.all_data[index]
        # print("data_path info >> ", data_path)
        data_path_split = data_path.split("\\")
        """data_path_split info >>  ['.\\dataset\\YogaPoses\\Downdog\\00000000.jpg']"""
        """data_path_split info >>  ['.', 'dataset', 'YogaPoses', 'Downdog', '00000000.jpg']"""
        labels_temp = data_path_split[3]
        """data_path_split info >>  Downdog"""
        # print("data_path_split info >> ", labels_temp)

        label = 0
        if "Downdog" == labels_temp:
            label = 0
        elif "Goddess" == labels_temp:
            label = 1
        elif "Plank" == labels_temp:
            label = 2
        elif "Tree" == labels_temp:
            label = 3
        elif "Warrior2" == labels_temp:
            label = 4

        images = Image.open(data_path).convert("RGB")

        if self.transform is not None:
            images = self.transform(images)

        # print(images, label)
        return images, label

    def __len__(self):
        return len(self.all_data)
