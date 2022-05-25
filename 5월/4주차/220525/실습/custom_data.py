import os
# from h11 import Data
import torch
import glob
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class MyCustomDatasetImage(Dataset):
    def __init__(self, path):
        # 정의
        # ./data/*/*.png
        self.all_data = sorted(glob.glob(os.path.join(path, '*', '*.png')))

    def __getitem__(self, index):
        data_path = self.all_data[index]
        # windows
        data_split = data_path.split('\\')
        # Mac
        # data_split = data_path.split('/')
        data_labels = data_split[1]

        labels = 0
        if data_labels == 'Ak':
            labels = 0
        elif data_labels == 'Ala_Idris':
            labels = 1
        elif data_labels == 'Buzgulu':
            labels = 2
        elif data_labels == 'Dimnit':
            labels = 3
        elif data_labels == 'Nazli':
            labels = 4

        # print(data_labels, labels)
        # cv2 PIL 이용해서 이미지 변경 해주시면 됩니다.
        return data_path, labels
        # 정의 작성된 내용을 구현

    def __len__(self):
        # 전체 길이를 반환 -> 리스트 [] len()
        return len(self.all_data)


# 2가지 구성 필요 -> dataset dataloader
dataset = MyCustomDatasetImage(path="Grapevine_Leaves_Image_Dataset/")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for path, label in dataloader:
    print(path, label)
