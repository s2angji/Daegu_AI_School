from torch.utils.data import DataLoader
import torch.optim as optim

# configer.py -> 하이퍼파라메타 값 세팅 하는 곳
# utils_file.py -> 여러가지 잡동사니 ex) Image show 필요한 함수 구현 하는 곳
# models.py -> 학습할 모델 build 파일
# customdataset.py -> 학습데이터를 가져오기 위한 데이터셋 구성
from utils_file import *
from models_build import *
from customdataset import *

# 1. device 설정
# configer.py
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. augmentation setting
data_transform = data_augmentation()

# 3. data set setting
train_data = CustomDataset(data_path=data_path, mode=Mode.train, transform=data_transform['train'])
test_data = CustomDataset(data_path=data_path, mode=Mode.val, transform=data_transform['test'])
# 디버그
# for i in train_data:
#     pass

# 4. data loader setting
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
# 디버그
# for data, target in train_loader:
#     print(data, target)

# 5. model call
resnet, image_size = initialize_model('resnet', num_classes=nc)
# print(net)

# 6. 하이퍼파라메타 값 call loss function 호출, optim, lr_scheduler
criterion = criterion

optimizer = optim.SGD(resnet.parameters(), lr=lr, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

# 7. train loop 함수 호출
# 8. test loop 함수 호출
# 과제 train loop 구성 하시면 됩니다.
model_ft, hist = train_model(
    resnet, {'train': train_loader, 'val': test_loader}, criterion, optimizer, lr_scheduler, num_epochs=num_epochs
)
visualize_loss_acc(history=hist, optim=f'SGD ; lr={lr}, step_size=4, gamma=0.1')
visualize_predict(model=model_ft, data=test_data)
sgd_lr025_dict_resnet = visualize_class_accuracies(model=model_ft, data=test_data)
save_model(model_ft, 'resnet_best.pt')

# vgg
vgg, _ = initialize_model('vgg', num_classes=nc)
optimizer = optim.SGD(vgg.parameters(), lr=lr, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
model_ft, hist = train_model(
    vgg, {'train': train_loader, 'val': test_loader}, criterion, optimizer, lr_scheduler, num_epochs=num_epochs
)
visualize_loss_acc(history=hist, optim=f'SGD ; lr={lr}, step_size=4, gamma=0.1')
visualize_predict(model=model_ft, data=test_data)
sgd_lr025_dict_vgg = visualize_class_accuracies(model=model_ft, data=test_data)
save_model(model_ft, 'vgg_best.pt')
