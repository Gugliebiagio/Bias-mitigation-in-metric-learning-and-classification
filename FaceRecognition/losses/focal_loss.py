import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss implementation
    (https://arxiv.org/pdf/1708.02002v2.pdf)
    """

    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()