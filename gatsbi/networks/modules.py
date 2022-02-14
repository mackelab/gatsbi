from typing import Callable, Optional

import torch
from torch import nn


class ModuleWrapper(nn.Module):
    """Class to wrap a torch function."""

    def __init__(self, func, **kwargs):
        """
        Wrap a torch function (eg. math operations) as an nn.Module object.

        For example: `ModuleWrapper(torch.exp)` makes torch.exp operation an
        nn.Module object that can then be used as a hidden layer in a torch
        network.

        Args:
            func: torch function
            *args: arguments to `func`
        """
        super(ModuleWrapper, self).__init__()
        self.func = func
        self.func_args = kwargs

    def forward(self, _input: torch.tensor) -> torch.tensor:
        """Forward pass."""
        return self.func(_input, **self.func_args)


class AddNoise(nn.Module):
    """Add noise along a particular input dimension."""

    def __init__(
        self,
        lat_dim: int,
        output_dim: int,
        add_dim: Optional[int] = -1,
        noise_dist: Optional[Callable] = torch.randn,
        heteroscedastic: Optional[bool] = True,
    ):
        """
        Add noise along a particular input dimension.

        Args:
            lat_dim: number of noise dimensions
            output_dim: number of output dimensions
            add_dim: input dimension along which noise is to be added
            noise_dist: method for sampling noise from a particular
                        distribution
            heteroscedastic: If True, input is also multiplied with
                             noise before being added.
        """
        super(AddNoise, self).__init__()
        self.lat_dim = lat_dim
        self.output_dim = output_dim
        self.add_dim = add_dim
        self.noise_dist = noise_dist
        self.heteroscedastic = heteroscedastic

        self.W = nn.Parameter(torch.randn(output_dim, lat_dim), requires_grad=True)

    def forward(self, inp: torch.tensor) -> torch.tensor:
        """Forward pass."""
        device = inp.device

        noise_shape = list(inp.shape)
        permute_dims = [i for i, _ in enumerate(noise_shape)]
        permute_shape = list(inp.shape)
        permute_shape[0] = self.output_dim
        permute_shape[self.add_dim] = noise_shape[0]

        permute_dims[self.add_dim] = 0
        permute_dims[0] = self.add_dim
        noise_shape[self.add_dim] = self.lat_dim

        noise = self.noise_dist(noise_shape)
        noise = noise.permute(*permute_dims)
        noise = noise.reshape(self.lat_dim, -1).to(device)

        mult_noise = torch.mm(self.W, noise).reshape(*permute_shape)
        mult_noise = mult_noise.permute(*permute_dims)
        if self.heteroscedastic:
            mult_noise = inp * mult_noise
        return inp + mult_noise


class AddConvNoise(nn.Module):
    """Add noise convolved with a linear filter."""

    def __init__(
        self,
        lat_dim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        noise_dist: Optional[Callable] = torch.randn,
        heteroscedastic: Optional[bool] = False,
        conv2d: Optional[bool] = False,
        convtrans: Optional[bool] = True,
        add: Optional[bool] = True,
    ):
        """
        Add noise convolved with a linear filter.

        Args:
            lat_dim: number of latent dimensions
            in_channels: number of input channels
            out_channels: number of output channels
            kernel_size: size of convolution kernel
            noise_dist: method for sampling noise from a particular
                        distribution
            heteroscedastic: If True, input is also multiplied with noise
                             before being added.
            conv2d: If True, noise sample and convolution is 2D.
            convtrans: If True, convolution is transposed.
            add: if True, add noise to input. If False, concatenate noise and
                 input.
        """
        super(AddConvNoise, self).__init__()
        self.lat_dim = lat_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.noise_dist = noise_dist
        self.heteroscedastic = heteroscedastic
        self.conv2d = conv2d
        self.convtrans = convtrans
        self.add = add

        conv = None
        if self.convtrans:
            if self.conv2d:
                conv = nn.ConvTranspose2d
            else:
                conv = nn.ConvTranspose1d
        else:
            if self.conv2d:
                conv = nn.Conv2d
            else:
                conv = nn.Conv1d

        self.conv = conv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            bias=False,
        )

    def forward(self, inp: torch.tensor) -> torch.tensor:
        """Forward pass."""
        device = inp.device
        if self.conv2d:
            noise_shape = [len(inp), self.in_channels, self.lat_dim, self.lat_dim]
        else:
            noise_shape = [len(inp), self.in_channels, self.lat_dim]

        noise = self.noise_dist(noise_shape).to(device)
        conv_noise = self.conv(noise)
        if self.heteroscedastic:
            conv_noise = inp * conv_noise
        if self.add:
            return inp + conv_noise
        else:
            return torch.cat([inp, conv_noise], 1)


class ParamLeakyReLU(nn.Module):
    """Parametrized LeakyReLU nonlinearity."""

    def __init__(self):
        """Parametrize slope of LeakyReLU function."""
        super(ParamLeakyReLU, self).__init__()
        self.negative_slope = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, _input: torch.tensor) -> torch.tensor:
        """Forward pass."""
        maximum = torch.max(torch.zeros_like(_input), _input)
        minimum = self.negative_slope * torch.min(torch.zeros_like(_input), _input)
        return maximum + minimum


class Collapse(nn.Module):
    """Collapse tensor to 2D."""

    def __init__(self):
        """Collapse tensor to 2D."""
        super(Collapse, self).__init__()

    def forward(self, input: torch.tensor) -> torch.tensor:
        """Forward pass."""
        shape = input.shape
        return input.reshape(shape[0], -1)


nonlin_dict = dict(
    leaky_relu=nn.LeakyReLU, relu=nn.ReLU, param_leaky_relu=ParamLeakyReLU
)
