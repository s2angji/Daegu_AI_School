from matplotlib import dviread
import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as F


def mixupFuction(x, y, device):
    mixupAlpha = 1.0  # Mixup 정도 조절 값
    lam = np.random.beta(mixupAlpha, mixupAlpha)
    batchSize = x.size()[0]

    index = torch.randperm(batchSize).to(device)
    mixed_x = lam * x + (1-lam) * x[index]
    yA, yB = y, y[index]
    return mixed_x, yA, yB, lam


def mixupCriterion(criterion, pred, yA, yB, lam):
    return lam * criterion(pred, yA) + (1-lam) * criterion(pred, yB)


class LabelSmootingCrossEntropy(Module):

    def __init__(self):
        super(LabelSmootingCrossEntropy, self).__init__()

    def forward(self, y, targets, smoothing=0.1):
        confidence = 1. - smoothing
        log_probs = F.log_softmax(y, dim=-1)  # 예측 확률 계산
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss

        return loss.mean()
