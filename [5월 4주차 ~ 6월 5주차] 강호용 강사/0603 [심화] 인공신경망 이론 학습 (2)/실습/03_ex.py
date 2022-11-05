import torch
import torch.nn as nn
import torch.nn.functional as F

"""데이터 생성"""
x_train = torch.FloatTensor([[73, 80, 75],
                            [93, 88, 93],
                            [89, 91, 90],
                            [96, 98, 100],
                            [73, 66, 70]])

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

"""모델 생성"""
model = nn.Linear(3, 1)

"""optimizer 설정"""
optimizer = torch.optim.SGD(model.parameters(), lr=0.000005)

nb_epochs = 2000

for epoch in range(nb_epochs+1):

    output = model(x_train)

    # loss 계산
    loss = F.mse_loss(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch : {:4d}/{} loss : {:.6f}'.format(epoch,
              nb_epochs, loss.item()))


new_val = torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_val)
print("예측값 >> ", pred_y)
