import torch.nn as nn

from torchmetrics.image import TotalVariation

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
        self.tv = TotalVariation('mean')

    def forward(self, x):
        return self.tv(x)
