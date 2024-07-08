import torch
import torch.nn as nn

from torchmetrics.image import TotalVariation, StructuralSimilarityIndexMeasure

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
        self.tv = TotalVariation('mean')

    def forward(self, x):
        return self.tv(x)

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, x, y):
        return 1 - self.ssim(x, y)
