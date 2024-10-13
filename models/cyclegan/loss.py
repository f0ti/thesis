import torch.nn as nn

from torchmetrics.image import TotalVariation, StructuralSimilarityIndexMeasure, SpectralDistortionIndex, LearnedPerceptualImagePatchSimilarity

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

        self.tv = TotalVariation('mean')

    def forward(self, x):
        return self.tv(x)


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

        self.ssim = StructuralSimilarityIndexMeasure()

    def forward(self, x, y):
        return 1-self.ssim(x, y)


class LPIPS(nn.Module):
    """The Learned Perceptual Image Patch Similarity is used to judge the perceptual similarity between
    two images. LPIPS essentially computes the similarity between the activations of two image patches
    for some pre-defined network. This measure has been shown to match human perception well. A low LPIPS
    score means that image patches are perceptual similar."""
    def __init__(self):
        super(LPIPS, self).__init__()

        # closer to "traditional" perceptual loss, when used for optimization
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='mean') 

    def forward(self, x, y):
        return self.lpips(x, y)
