import torch
import torch.nn as nn

from torchmetrics.image import TotalVariation, StructuralSimilarityIndexMeasure, SpectralDistortionIndex

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


class SDILoss(nn.Module):
    def __init__(self):
        super(SDILoss, self).__init__()
        self.sdi = SpectralDistortionIndex()

    def forward(self, x, y):
        return self.sdi(x, y)


def loss_functions_buffer(loss_fns_config):
    loss_fns = {}
    for loss_fn in loss_fns_config:
        if loss_fn == "gan":
            criterion_GAN = torch.nn.MSELoss()
            loss_fns["gan"] = criterion_GAN
        elif loss_fn == "cycle":
            criterion_cycle = torch.nn.L1Loss()
            loss_fns["cycle"] = criterion_cycle
        elif loss_fn == "identity":
            criterion_identity = torch.nn.L1Loss()
            loss_fns["identity"] = criterion_identity
        elif loss_fn == "tv":
            criterion_tv = TotalVariationLoss()
            loss_fns["tv"] = criterion_tv
        elif loss_fn == "ssim":
            criterion_ssim = SSIMLoss()
            loss_fns["ssim"] = criterion_ssim
        elif loss_fn == "sdi":
            criterion_sdi = SDILoss()
            loss_fns["sdi"] = criterion_sdi

    return loss_fns
