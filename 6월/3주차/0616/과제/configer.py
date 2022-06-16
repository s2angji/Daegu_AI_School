import os
import torch
import torch.nn as nn

# 모델 가중치 저장할 폴더 생성
os.makedirs('\\weights', exist_ok=True)

# device
# device = 'cuda' if torch.cuda.is_available() else 'mps'
# device = torch.device('mps')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 하이퍼파라메타 값 세팅
batch_size = 36
num_epochs = 10
val_every = 1 # 평가 간격
save_weights_dir = '.\\weights'
data_path = '.\\dataset'
nc = 5 # num_classes
lr = 0.025 # learning rate
criterion = nn.CrossEntropyLoss().to(device)
