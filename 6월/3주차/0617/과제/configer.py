import os
import torch
import torch.nn as nn

# device
# device = 'cuda' if torch.cuda.is_available() else 'mps'
# device = torch.device('mps')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 하이퍼파라메타 값 세팅
batch_size = 36
num_epochs = 2
val_every = 1 # 평가 간격
save_weights_dir = '.\\weights'
# 모델 가중치 저장할 폴더 생성
os.makedirs(save_weights_dir, exist_ok=True)
data_path = '.\\dataset'
nc = len(os.listdir(os.path.join(data_path, os.listdir(data_path)[0]))) # num_classes
lr = 0.025 # learning rate
criterion = nn.CrossEntropyLoss().to(device)
