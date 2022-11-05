import torch.cuda
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from zmq import device
import utils_file
import models_build
from customdataset import CustomDataset
import configer
import torch
"""
models.py -> 학습할 모델 build 파일 
utils_file.py -> 여려가지 잡동사니 ex) Image show 필요한 함수 구현 곳 
customdataset.py -> 학습데이터를 가져오기위한 데이터셋 구성 
config.py -> 하이퍼파라메타 값 세팅 하는곳 
"""

"""1. augmentation setting"""
data_transform = utils_file.data_augmentation()

"""2. data set setting"""
# data path, mode, transform
train_data = CustomDataset(data_path=configer.data_path,
                           mode="train", transform=data_transform['train'])
test_data = CustomDataset(data_path=configer.data_path,
                          mode="val", transform=data_transform['test'])
"""디버그"""
# for i in train_data:
#     pass

"""3. data loader setting"""
train_loader = DataLoader(
    train_data, batch_size=configer.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(
    test_data, batch_size=configer.batch_size, shuffle=False, drop_last=True)

# for data, target in test_loader:
#     print(data, target)
"""4. model call"""
net, image_size = models_build.initialize_model(
    "resnet", num_classes=configer.nc)

"""5. 하이퍼파레메타 값 call loss 함수 호출 optim, lr_scheduler"""
criterion = configer.criterion
optimizer = optim.SGD(net.parameters(), lr=configer.lr, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=4, gamma=0.1)
"""6. train loop 함수 호출"""
utils_file.train(configer.num_epochs, net, train_loader, test_loader, criterion, optimizer, configer.save_weights_dir,
                 configer.val_every, configer.device)

#

"""7. test loop 함수 호출"""
