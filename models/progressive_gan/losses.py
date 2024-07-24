""" Module implementing various loss functions """

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
from torchmetrics.image import TotalVariation, StructuralSimilarityIndexMeasure, SpectralDistortionIndex

from networks import Discriminator, Generator


class GANLoss:
    def dis_loss(
        self,
        discriminator: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        calculate the discriminator loss using the following data
        Args:
            discriminator: the Discriminator used by the GAN
            real_samples: real batch of samples
            fake_samples: fake batch of samples
            depth: resolution log 2 of the images under consideration
            alpha: alpha value of the fader
            labels: optional in case of the conditional discriminator

        Returns: computed discriminator loss
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(
        self,
        discriminator: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        """
        calculate the generator loss using the following data
        Args:
            discriminator: the Discriminator used by the GAN
            real_samples: real batch of samples
            fake_samples: fake batch of samples
            depth: resolution log 2 of the images under consideration
            alpha: alpha value of the fader
            labels: optional in case of the conditional discriminator

        Returns: computed discriminator loss
        """
        raise NotImplementedError("gen_loss method has not been implemented")


class StandardGAN(GANLoss):
    def __init__(self):
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(
        self,
        discriminator: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float
    ) -> Tensor:
        
        real_scores = discriminator(real_samples, depth, alpha)
        fake_scores = discriminator(fake_samples, depth, alpha)

        real_loss = self.criterion(
            real_scores, torch.ones(real_scores.shape).to(real_scores.device)
        )
        fake_loss = self.criterion(
            fake_scores, torch.zeros(fake_scores.shape).to(fake_scores.device)
        )
        return (real_loss + fake_loss) / 2

    def gen_loss(
        self,
        discriminator: Discriminator,
        _: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float
    ) -> Tensor:
        
        fake_scores = discriminator(fake_samples, depth, alpha)
        return self.criterion(
            fake_scores, torch.ones(fake_scores.shape).to(fake_scores.device)
        )


class WganGP(GANLoss):
    """
    Wgan-GP loss function. The discriminator is required for computing the gradient
    penalty.
    Args:
        drift: weight for the drift penalty
    """

    def __init__(self, drift: float = 0.001) -> None:
        self.drift = drift

    @staticmethod
    def _gradient_penalty(
        dis: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float,
        reg_lambda: float = 10
    ) -> Tensor:
        """
        private helper for calculating the gradient penalty
        Args:
            dis: the discriminator used for computing the penalty
            real_samples: real samples
            fake_samples: fake samples
            depth: current depth in the optimization
            alpha: current alpha for fade-in
            reg_lambda: regularisation lambda

        Returns: computed gradient penalty
        """
        batch_size = real_samples.shape[0]

        # generate random epsilon
        epsilon = torch.rand((batch_size, 1, 1, 1)).to(real_samples.device)

        # create the merge of both real and fake samples
        merged = epsilon * real_samples + ((1 - epsilon) * fake_samples)
        merged.requires_grad_(True)

        op = dis(merged, depth, alpha)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = torch.autograd.grad(
            outputs=op,
            inputs=merged,
            grad_outputs=torch.ones_like(op),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(
        self,
        discriminator: Discriminator,
        real_samples: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float,
        labels: Optional[Tensor] = None,
    ) -> Tensor:
        real_scores = discriminator(real_samples, depth, alpha)
        fake_scores = discriminator(fake_samples, depth, alpha)
        loss = (
            torch.mean(fake_scores)
            - torch.mean(real_scores)
            + (self.drift * torch.mean(real_scores**2))
        )

        # calculate the WGAN-GP (gradient penalty)
        gp = self._gradient_penalty(
            discriminator, real_samples, fake_samples, depth, alpha
        )
        loss += gp

        return loss

    def gen_loss(
        self,
        discriminator: Discriminator,
        _: Tensor,
        fake_samples: Tensor,
        depth: int,
        alpha: float,
    ) -> Tensor:
        fake_scores = discriminator(fake_samples, depth, alpha)
        return -torch.mean(fake_scores)


class CycleGANLoss:
    def __init__(self, lambda_cycle: float = 8.0, lambda_identity: float = 0.5):
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def dis_loss(
        self,
        generator_AB: Generator,
        generator_BA: Generator,
        discriminator_A: Discriminator,
        discriminator_B: Discriminator,
        real_A: Tensor,
        real_B: Tensor,
    ) -> Tensor:

        # Discriminator A
        loss_real_A = self.criterion_GAN(discriminator_A(real_A), torch.ones(real_A.size(0)).to(real_A.device))
        fake_A = generator_BA(real_B)
        loss_fake_A = self.criterion_GAN(discriminator_A(fake_A), torch.zeros(fake_A.size(0)).to(fake_A.device))

        # Discriminator B
        loss_real_B = self.criterion_GAN(discriminator_B(real_B), torch.ones(real_B.size(0)).to(real_B.device))
        fake_B = generator_AB(real_A)
        loss_fake_B = self.criterion_GAN(discriminator_B(fake_B), torch.zeros(fake_B.size(0)).to(fake_B.device))

        return (loss_real_A + loss_fake_A + loss_real_B + loss_fake_B) / 4

    def gen_loss(
        self,
        generator_AB: Generator,
        generator_BA: Generator,
        discriminator_A: Discriminator,
        discriminator_B: Discriminator,
        real_A: Tensor,
        real_B: Tensor,
    ) -> Tensor:
        
        # Identity loss
        loss_id_A = self.criterion_identity(generator_BA(real_A), real_A)
        loss_id_B = self.criterion_identity(generator_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2 * self.lambda_identity
        
        # GAN loss
        fake_B = generator_AB(real_A)
        fake_A = generator_BA(real_B)
        loss_GAN_AB = self.criterion_GAN(discriminator_B(fake_B), torch.ones(fake_B.size(0)).to(fake_B.device))
        loss_GAN_BA = self.criterion_GAN(discriminator_A(fake_A), torch.ones(fake_A.size(0)).to(fake_A.device))

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        reconstructed_A = generator_BA(fake_B)
        reconstructed_B = generator_AB(fake_A)
        loss_cycle_A = self.criterion_cycle(reconstructed_A, real_A)
        loss_cycle_B = self.criterion_cycle(reconstructed_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2 * self.lambda_cycle

        return loss_GAN + loss_cycle + loss_identity


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
