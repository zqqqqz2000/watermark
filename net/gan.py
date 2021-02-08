import functools

import torch
import torch.nn as nn
from .norms import SpectralNorm


class G(nn.Module):
    """Generator."""

    def __init__(self, img_c=3, layerX2=4):
        super(G, self).__init__()

        conv_layers = []
        conv_transpose_layers = []

        curr_dim = img_c

        for curr_layer in range(layerX2):
            conv_layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 3)))
            conv_layers.append(nn.BatchNorm2d(curr_dim * 2))
            conv_layers.append(nn.ReLU())
            curr_dim *= 2

        for curr_layer in range(layerX2):
            conv_transpose_layers.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 3)))
            conv_transpose_layers.append(nn.BatchNorm2d(curr_dim // 2))
            conv_transpose_layers.append(nn.ReLU())
            curr_dim //= 2
        self.conv_layers = nn.Sequential(*conv_layers)
        self.conv_transpose_layers = nn.Sequential(*conv_transpose_layers)
        self.last = nn.Sigmoid()

    def frozen(self, requires_grad=True):
        parameters = self.named_parameters()
        for name, layers in parameters:
            layers.requires_grad = requires_grad

    def forward(self, img: torch.Tensor):
        out = self.conv_layers(img)
        out = self.conv_transpose_layers(out)
        out = self.last(out) - 0.5
        return out


class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

    def frozen(self, state=True):
        parameters = self.named_parameters()
        for name, layers in parameters:
            layers.requires_grad = state
