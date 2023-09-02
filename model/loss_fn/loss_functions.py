from typing import Any

from torch import nn
from torch.nn.modules.loss import BCEWithLogitsLoss


class WeightedLosses(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, *input: Any, **kwargs: Any):
        c_loss = 0
        for loss, w in zip(self.losses, self.weights):
            c_loss += w * loss.forward(*input, **kwargs)
        return c_loss


class BinaryCrossentropy(BCEWithLogitsLoss):
    pass
