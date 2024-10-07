import torch
import torch.nn as nn

from typing import List

# most of the code belongs to: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def padding_layer(self, padding_type):
        if padding_type == 'reflect':
            return nn.ReflectionPad2d(1)
        elif padding_type == 'replicate':
            return nn.ReplicationPad2d(1)
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        conv_block += [self.padding_layer(padding_type)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [self.padding_layer(padding_type)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)  # add skip connections
        return out


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels, out_channels, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GeneratorResNet, self).__init__()
        
        use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            # this solves the checkerboard artifacts
            # model += [
            #     nn.Upsample(scale_factor = 2, mode='bilinear'),
            #     nn.ReflectionPad2d(1),
            #     nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0),
            #     norm_layer(int(ngf * mult / 2)),
            #     nn.ReLU(True)
            # ]
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                kernel_size=3, stride=2,
                padding=1, output_padding=1,
                bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x
    

class GeneratorUNet(nn.Module):
    def __init__(self, input_nc: int, output_nc: int, num_downs: int, ngf: int = 64, 
                 norm_layer=nn.BatchNorm2d, use_dropout: bool = False):
        super(GeneratorUNet, self).__init__()

        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        
        for _ in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, 
                 innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d
        
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)  # type: ignore

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers: List = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)  # PatchGAN outputs 16x16
        )

    def forward(self, x):
        x = self.model(x)
        return x
    

class PatchDiscriminator(nn.Module):
    def __init__(self, channels):
        super(PatchDiscriminator, self).__init__()

        self.output_shape = (1, 30, 30)

        out_features = 64
        model = [nn.Conv2d(channels, out_features, 4, stride=2, padding=1)]
        model += [nn.LeakyReLU(0.2, inplace=True)]

        for n in range(3):
            in_features = out_features
            out_features *= 2
            if n == 2:
                model += [nn.Conv2d(in_features, out_features, 4, stride=1, padding=1)]
            else:
                model += [nn.Conv2d(in_features, out_features, 4, stride=2, padding=1)]
            
            model += [nn.InstanceNorm2d(out_features),
                      nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(out_features, 1, 4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x
