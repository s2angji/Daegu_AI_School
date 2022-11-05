"""
다층퍼셉트론으로 손글씨 분류
사이킷런 패키지에서 제공하는 분류용 예측 데이터를 이용
0 ~ 9 까지의 숫자를 손으로 쓴 이미지 데이터로 load_digits() 명령어로 로드
각 이미지 사이즈는 8 x 8 = 64px 구성 흑백이미지 이미지 갯수 1,797개
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

import torch
import torch.nn as nn
from torch import optim

digits = load_digits()
print("image 행렬 \n", digits.images[0])
print("타겟 >>> ", digits.target[0])
print("전체 데이터 >> ", len(digits.images))

"""
상위 5개만 샘플이미지 확인
zip 
image = [1, 2, 3, 4]
label = ['사과', '자몽', '바나나', '수박']

list(zip(image, label))

1 사과 2 자몽 3 바나나 4 수박
"""
images_and_labels = list(zip(digits.images, digits.target))

for index, (image, label) in enumerate(images_and_labels[:10]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("sample : % i" % label)
plt.show()

# 데이터 생성
x = digits.data
y = digits.target

# 모델 생성
model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터 텐서
x = torch.tensor(x, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.int64).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

loss_list = []
for epoch in range(101):
    optimizer.zero_grad() # optimizer 초기화
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print("epoch {:4d}/{} loss : {:.6f}".format(epoch, 100, loss.item()))

    loss_list.append(loss.item())

plt.title("loss")
plt.plot(loss_list)
plt.show()
