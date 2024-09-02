import torch
from custom_layers import (
    EqualizedConv2d,
    EqualizedConvTranspose2d,
    MinibatchStdDev,
    PixelwiseNorm,
)
from torch import Tensor
from torch.nn import AvgPool2d, MaxPool2d, Conv2d, ConvTranspose2d, Embedding, LeakyReLU, Module, Sigmoid
from torch.nn.functional import interpolate


class GenInitialBlock(Module):
    """
    Module implementing the initial block of the input
    Args:
        in_channels: number of input channels to the block
        out_channels: number of output channels of the block
        use_eql: whether to use equalized learning rate
    """

    def __init__(self, in_channels: int, out_channels: int, use_eql: bool = True) -> None:
        super(GenInitialBlock, self).__init__()
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2d if use_eql else Conv2d

        # bring the input feature maps based on the paper                            # in_channels x 256 x 256
        self.conv1 = ConvBlock(in_channels, 64, kernel_size=3, stride=2, padding=3)  # 64 x 130 x 130
        self.conv2 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=2)          # 128 x 66 x 66
        self.conv3 = ConvBlock(128, 128, kernel_size=3, stride=3, padding=1)         # 128 x 22 x 22
        self.conv4 = ConvBlock(128, 256, kernel_size=3, stride=3, padding=1)         # 256 x 8 x 8
        self.avgpool = AvgPool2d(2)                                                  # 256 x 4 x 4
        
        self.pixNorm = PixelwiseNorm()
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pixNorm(x)
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.avgpool(x)
        x = self.pixNorm(x)
        return x


class GenGeneralConvBlock(torch.nn.Module):
    """
    Module implementing a general convolutional block
    Args:
        in_channels: number of input channels to the block
        out_channels: number of output channels required
        use_eql: whether to use equalized learning rate
    """

    def __init__(self, in_channels: int, out_channels: int, use_eql: bool) -> None:
        super(GenGeneralConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2d if use_eql else Conv2d

        self.conv_1 = ConvBlock(in_channels, out_channels, (3, 3), padding=1, bias=True)
        self.conv_2 = ConvBlock(out_channels, out_channels, (3, 3), padding=1, bias=True)
        self.pixNorm = PixelwiseNorm()
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x: Tensor) -> Tensor:
        y = interpolate(x, scale_factor=2)  # upsample
        y = self.pixNorm(self.lrelu(self.conv_1(y)))  # convolution 3x3
        y = self.pixNorm(self.lrelu(self.conv_2(y)))  # convolution 3x3

        return y


class DisFinalBlock(torch.nn.Module):
    """
    Final block for the Discriminator
    Args:
        in_channels: number of input channels
        use_eql: whether to use equalized learning rate
    """

    def __init__(self, in_channels: int, out_channels: int, use_eql: bool) -> None:
        super(DisFinalBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2d if use_eql else Conv2d

        self.conv_1 = ConvBlock(
            in_channels + 1, in_channels, (3, 3), padding=1, bias=True
        )
        self.conv_2 = ConvBlock(in_channels, out_channels, (4, 4), bias=True)
        self.conv_3 = ConvBlock(out_channels, 1, (1, 1), bias=True)
        self.batch_discriminator = MinibatchStdDev()
        self.lrelu = LeakyReLU(0.2)
        self.sigmoid = Sigmoid()
        
    def forward(self, x: Tensor) -> Tensor:
        y = self.batch_discriminator(x)  # this adds one more feature map to the input
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))
        y = self.conv_3(y)
        y = y.view(-1)
        return y


class DisGeneralConvBlock(torch.nn.Module):
    """
    General block in the discriminator
    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        use_eql: whether to use equalized learning rate
    """

    def __init__(self, in_channels: int, out_channels: int, use_eql: bool) -> None:
        super(DisGeneralConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2d if use_eql else Conv2d

        self.conv_1 = ConvBlock(in_channels, in_channels, (3, 3), padding=1, bias=True)
        self.conv_2 = ConvBlock(in_channels, out_channels, (3, 3), padding=1, bias=True)
        self.downSampler = AvgPool2d(2)
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x: Tensor) -> Tensor:
        y = self.lrelu(self.conv_1(x))
        y = self.lrelu(self.conv_2(y))
        y = self.downSampler(y)
        return y
