import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import utils
import utils_file
from customdataset import CustomDataset
import configer
import models_build


# models.py -> 학습할 모델 build 파일
# utils_file.py -> 여러가지 잡동사니 ex) Image show 필요한 함수 구현 하는 곳
# customdataset.py -> 학습데이터를 가져오기 위한 데이터셋 구성
# configer.py -> 하이퍼파라메타 값 세팅 하는 곳

# 1. device 설정
device = configer.device

# 2. augmentation setting
data_transform = utils_file.data_augmentation()

# 3. data set setting
train_data = CustomDataset(data_path=configer.data_path, mode='train', transform=data_transform['train'])
test_data = CustomDataset(data_path=configer.data_path, mode='val', transform=data_transform['test'])
# 디버그
# for i in train_data:
#     pass

# 4. data loader setting
train_loader = DataLoader(train_data, batch_size=configer.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=configer.batch_size, shuffle=False, drop_last=True)
# 디버그
# for data, target in train_loader:
#     print(data, target)

# 5. model call
net, image_size = models_build.initialize_model('resnet', num_classes=configer.nc)
# print(net)

# 6. 하이퍼파라메타 값 call loss function 호출, optim, lr_scheduler
criterion = configer.criterion
optimizer = optim.SGD(net.parameters(), lr=configer.lr, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

# 7. train loop 함수 호출
# 8. test loop 함수 호출
# 과제 train loop 구성 하시면 됩니다.
model_ft, hist = utils_file.train_model(
    net, {'train': train_loader, 'val': test_loader}, criterion, optimizer, lr_scheduler, num_epochs=configer.num_epochs
)
utils_file.loss_acc_visualize(history=hist, optim=f'SGD ; lr={configer.lr}, step_size=4, gamma=0.1')
utils_file.visual_predict(model=model_ft, data=test_data)
sgd_lr025_dict = utils_file.class_accuracies(model=model_ft, data=test_data)
