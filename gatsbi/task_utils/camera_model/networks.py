"""
Architecture similar to Pix-to-pix GAN.
Code closely follows
https://machinelearningmastery.com/\
    how-to-develop-a-pix2pix-gan-for-image-to-image-translation/
"""
import torch
import torch.nn as nn

from gatsbi.networks import (AddConvNoise, BaseNetwork, Discriminator,
                             ModuleWrapper)


class ConvBlock(nn.Module):
    """Convolution block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        spec_norm=False,
        norm=True,
        nonlin=True,
    ):
        """Set up convolution block."""
        super(ConvBlock, self).__init__()
        block = []
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        if spec_norm:
            conv = nn.utils.spectral_norm(conv)
        block.append(conv)

        if nonlin:
            block.append(nn.LeakyReLU(0.2))

        if norm:
            batch_norm = nn.BatchNorm2d(out_channels)
            # if spec_norm:
            #     batch_norm = nn.utils.spectral_norm(batch_norm)
            block.append(batch_norm)

        self.block = nn.Sequential(*block)

    def forward(self, x):
        """Forward pass."""
        return self.block(x)


class TransConvBlock(nn.Module):
    """Transpose convolutional block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        spec_norm=False,
    ):
        """Set up transpose convolutional block."""

        super(TransConvBlock, self).__init__()

        conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        batch_norm = nn.BatchNorm2d(out_channels)
        if spec_norm:
            conv = nn.utils.spectral_norm(conv)

        self.block = nn.Sequential(conv, nn.ReLU(), batch_norm)

    def forward(self, x):
        """Forward pass."""
        return self.block(x)


class CameraGenerator(BaseNetwork):
    """Generator network for camera model."""

    def __init__(self):
        """Set up generator network."""
        gen_hidden_layers = [
            ConvBlock(1, 8, 2),
            ConvBlock(8, 16, 2),
            ConvBlock(16, 32, 2),
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=0),
            AddConvNoise(1, 200, 64, 3, heteroscedastic=True, conv2d=True, add=False),
            TransConvBlock(128, 32, kernel_size=3),
            TransConvBlock(64, 16, kernel_size=2),
            TransConvBlock(32, 8, 3),
            TransConvBlock(16, 4, 2),
            nn.ConvTranspose2d(4, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        ]
        super(CameraGenerator, self).__init__(hidden_layers=gen_hidden_layers)

    def forward(self, x):
        """Forward pass."""
        enc1 = self._hidden_layers[0](x)
        enc2 = self._hidden_layers[1](enc1)
        enc3 = self._hidden_layers[2](enc2)

        latent = self._hidden_layers[3](enc3)
        noisy_latent = self._hidden_layers[4](latent)
        dec1 = torch.cat([self._hidden_layers[5](noisy_latent), enc3], 1)
        dec2 = torch.cat([self._hidden_layers[6](dec1), enc2], 1)
        dec3 = torch.cat([self._hidden_layers[7](dec2), enc1], 1)
        output = self._hidden_layers[8:](dec3)

        return output


class CameraDiscriminator(Discriminator):
    """Discriminator network for camera model."""

    def __init__(self):
        """Set up discriminator network."""
        dis_hidden_layers = [
            ModuleWrapper(torch.cat, axis=1),
            ConvBlock(2, 4, 2, spec_norm=True),
            ConvBlock(4, 8, 2, spec_norm=True),
            ConvBlock(8, 4, 2, spec_norm=True),
            nn.utils.spectral_norm(nn.Conv2d(4, 1, kernel_size=5, stride=1, padding=0)),
            nn.Sigmoid(),
        ]
        super(CameraDiscriminator, self).__init__(dis_hidden_layers, conditional=False)
