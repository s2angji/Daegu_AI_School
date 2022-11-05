from crawling import Crawling
from dataset import TorchvisionDataset

from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

search_keywords = ['망고', '용과', '리치', '두리안']
dir_names = ['mango', 'dragon_fruit', 'lychee', 'durian']

# 전처리: 크롤링 및 이미지 resize(mango, dragon_fruit, lychee, durian 폴더에 저장)
with Crawling() as crawling:
    for search_keyword, dir_name in zip(search_keywords, dir_names):
        crawling.do_crawling(search_keyword, dir_name)

# 후처리: 커스텀 Dataset 클래스를 만들어 이미지와 라벨을 가져옴
torchvision_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
dataset = TorchvisionDataset('.\\', dir_names, torchvision_transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
# torchvision_transform 처리 시간 확인
total_time = 0
for i in range(len(dataloader)):
    image, label, transform_time = dataset[i]
    total_time += transform_time
print(f"torchvision time : {total_time * 10} ms")
# 라벨 확인 및 이미지 확인
image, label, transform_time = dataset[0]
print(dir_names[label], label)
plt.imshow(transforms.ToPILImage()(image))
plt.show()
