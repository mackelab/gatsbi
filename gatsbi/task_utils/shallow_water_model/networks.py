import numpy as np
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn

from gatsbi.networks.base import BaseNetwork
from gatsbi.networks.models import Discriminator
from gatsbi.networks.modules import AddNoise, Collapse


def _remove_noise(list_gen_layers, noise_layer):
    ind = np.where([isinstance(layer, noise_layer) for layer in list_gen_layers])[0]

    # Remove added conv noise layers
    for i in ind:
        list_gen_layers.pop(i)
        ind[ind > i] -= 1
    return list_gen_layers


def _make_embed_net():
    # Get generator layers
    gen = ShallowWaterGenerator(ngf=64)
    list_hl = list(gen._hidden_layers)

    # Find where noise is added and remove this layer
    list_hl = _remove_noise(list_hl, AddNoise)

    # Spec norm for some layers
    # TODO: more sophisticated way of finding correct layers to spec norm
    for i, g in enumerate(list_hl[::3]):
        list_hl[3 * i] = sn(g)

    return list_hl


class ShallowWaterGenerator(BaseNetwork):
    """Generator network for shallow water model."""

    def __init__(self, ngf=128, nc=1):
        """Set up generator network."""
        gen_hidden_layers = [
            nn.Conv2d(2, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(
                ngf * 4,
                # affine=False,
                # track_running_stats=False
            ),
            nn.ReLU(True),
            nn.Conv2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(
                ngf * 2,
                # affine=False,
                # track_running_stats=False
            ),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(
                ngf,
                # affine=False,
                # track_running_stats=False
            ),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            Collapse(),
            AddNoise(25, 36, heteroscedastic=False),
            nn.Linear(36, 100, bias=False),
        ]
        super(ShallowWaterGenerator, self).__init__(gen_hidden_layers)


class ShallowWaterDiscriminator(Discriminator):
    """Discriminator for shallow water model."""

    def __init__(self):
        """Set up discriminator."""
        embed_net_2 = _make_embed_net()

        dis_hidden_layers = [
            sn(nn.Linear(in_features=200, out_features=256, bias=True)),
            nn.LeakyReLU(0.2),
            sn(nn.Linear(in_features=256, out_features=256, bias=True)),
            nn.LeakyReLU(0.2),
            sn(nn.Linear(in_features=256, out_features=256, bias=True)),
            nn.LeakyReLU(0.2),
            sn(nn.Linear(in_features=256, out_features=1, bias=True)),
            nn.Sigmoid(),
        ]
        dictionary = {"net_1": [None], "net_2": embed_net_2}
        super(ShallowWaterDiscriminator, self).__init__(
            dis_hidden_layers, conditional=True, embed=True, embed_networks=dictionary
        )
