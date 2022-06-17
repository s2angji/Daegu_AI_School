import os
import torch
import torch.nn as nn

"""모델 가중치 저장할 폴더 생성"""
os.makedirs(".\\weights", exist_ok=True)

"""device"""
"""1. device 설정"""
# device = torch.device("mps")
# cpu 사용자
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

"""하이퍼 파레메타 값 세팅"""
batch_size = 36
num_epochs = 10
val_every = 1
save_weights_dir = ".\\weights"
data_path = ".\\dataset"
nc = 5
lr = 0.025
criterion = nn.CrossEntropyLoss().to(device)
