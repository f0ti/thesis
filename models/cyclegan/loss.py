import torch.nn as nn

from torchmetrics.image import TotalVariation, StructuralSimilarityIndexMeasure, SpectralDistortionIndex

class TVLoss(nn.Module):
    def __init__(self):
        self.tv = TotalVariation('mean')

    def forward(self, x):
        return self.tv(x)


class SSIMLoss(nn.Module):
    def __init__(self):
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, x, y):
        return 1 - self.ssim(x, y)


class SDILoss(nn.Module):
    def __init__(self):
        self.sdi = SpectralDistortionIndex()

    def forward(self, x, y):
        return self.sdi(x, y)
