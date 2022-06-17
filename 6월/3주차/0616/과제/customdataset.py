import glob
import os
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_path, mode, transform=None):
        # init : 초기값 설정
        # 데이터 가져오기 전체 데이터 경로 불러오기
        self.all_data = sorted(glob.glob(os.path.join(data_path, mode, '*', '*.jpg')))
        # data_path > .\\dataset\\train * -> DownDog *.jpg -> 000000.jpg
        self.transform = transform

        paths = sorted(glob.glob(os.path.join(data_path, mode, '*')))
        self.labels = [path.split('\\')[-1] for path in paths]
        self.train_dict = {label: sorted(glob.glob(os.path.join(data_path, 'train', label, '*.jpg')))
                           for label in self.labels}
        self.valid_dict = {label: sorted(glob.glob(os.path.join(data_path, 'val', label, '*.jpg')))
                           for label in self.labels}

    def __getitem__(self, index):
        data_path = self.all_data[index]
        # print('data_path info >> ', data_path)
        data_path_split = data_path.split('\\')
        # data_path_split info >>  ['.\\dataset\\train\\Downdog\\00000000.jpg']
        # data_path_split info >>  ['.', 'dataset', 'train', 'Downdog', '00000000.jpg']
        labels_temp = data_path_split[3]
        # labels_temp info >> Downdog

        # label = 0
        # if 'Downdog' == labels_temp:
        #     label = 0
        # elif 'Goddess' == labels_temp:
        #     label = 1
        # elif 'Plank' == labels_temp:
        #     label = 2
        # elif 'Tree' == labels_temp:
        #     label = 3
        # elif 'Warrior2' == labels_temp:
        #     label = 4

        images = Image.open(data_path).convert('RGB')

        if self.transform is not None:
            images = self.transform(images)

        return images, self.labels.index(labels_temp)

    def __len__(self):
        return len(self.all_data)
