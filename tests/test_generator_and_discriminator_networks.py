import unittest

import torch
import torch.nn as nn

from gatsbi.networks.base import BaseNetwork
from gatsbi.networks.models import Discriminator
from gatsbi.networks.modules import AddNoise


# Make network
def _make_generator():
    gen_hidden_layers = [nn.Linear(2, 2, bias=True), AddNoise(2, 2), nn.LeakyReLU()]
    return BaseNetwork(gen_hidden_layers)


def _make_discriminator():
    dis_hidden_layers = [nn.Linear(4, 1, bias=True), nn.Sigmoid()]
    embed_net_1 = [None]
    embed_net_2 = [None]

    return Discriminator(
        dis_hidden_layers,
        conditional=True,
        embed=True,
        embed_networks={"net_1": embed_net_1, "net_2": embed_net_2},
    )


class TestNetworks(unittest.TestCase):
    """Test code for set up and forward pass through networks."""

    def test_generator_forward_pass(self):
        """Test setup and forward pass through generator."""
        gen = _make_generator()
        output = gen(torch.zeros(1, 2))
        self.assertEqual(output.shape, torch.Size([1, 2]))

    def test_discriminator_architecture(self):
        """Test setup and forward pass through discriminator."""
        dis = _make_discriminator()
        output = dis([torch.zeros(1, 2), torch.zeros(1, 2)])
        self.assertEqual(output.shape, torch.Size([1, 1]))
