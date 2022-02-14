from typing import List

import torch
from torch import nn


class BaseNetwork(nn.Module):
    """Base class for all GAN networks /  doubles as GAN generator class."""

    def __init__(self, hidden_layers: List[nn.Module]):
        """
        Set up base class.

        Args:
            hidden_layers (list): list of nn.Module objects that will be fed
                                  into nn.Sequential, to build the network.
        """
        super(BaseNetwork, self).__init__()
        self._hidden_layers = nn.Sequential(*hidden_layers)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            _input: torch Tensor input to the network

        Returns:
            Output of the network
        """
        return self._hidden_layers(_input)
