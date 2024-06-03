import torch
import torch.nn as nn

from torch import Tensor
from torchmetrics.image import VisualInformationFidelity
from torchmetrics.image.fid import FrechetInceptionDistance

class VIFLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vifp = VisualInformationFidelity()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.vifp(input, target)

class FrechetID(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fid = FrechetInceptionDistance(feature=64, input_img_size=(3, 256, 256))

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        self.fid.update(input, real=False)
        self.fid.update(target, real=True)
        return self.fid.compute()
