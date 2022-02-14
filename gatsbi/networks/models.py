from typing import List, Optional

import torch
from torch import nn

from gatsbi.networks.base import BaseNetwork


def _concatenate(inputs):
    """Concatenates input along last dimension."""
    return torch.cat(inputs, -1)


def _add(inputs, heteroscedastic=False):
    """Add list of inputs, as long as they are of same shape."""
    # TODO: currently assumes that input list has only 2 elements, and second
    # input is noise
    assert torch.all(torch.tensor([inp.shape == inputs[0].shape for inp in inputs[1:]]))
    if heteroscedastic:
        return inputs[0] * (1 + inputs[1])
    else:
        return inputs[0] + inputs[1]


class Generator(BaseNetwork):
    """GAN generator network."""

    def __init__(
        self,
        hidden_layers: List[nn.Module],
        embed: Optional[bool] = False,
        embed_networks: Optional[dict] = dict(net_1=[None], net_2=[None]),
        heteroscedastic: Optional[bool] = False,
    ):
        """
        Set up GAN generator network.

        Inputs are embedded with the embedding networks, and then added
        together before being passed through the hidden layers of the network.
        This is useful for constructing generators with finer control over the
        latent variables input to the network e.g. while doing DDLS (Che et
        al., 2020) or DRS (Azadi et al., 2019).

        A generator constructed using this class allows the latent variables
        to be sampled externally, and passed to the network as input, rather
        than letting the sampling happen under the hood with every call to the
        generator (achieved using the AddNoise module in
        gatsbi.networks.modules)

        Args:
            hidden_layers: list of nn.Module objects that will be fed into
                           nn.Sequential, to build the network. If `embed` is
                           True, this list should be configured with
                           torch.nn.Module objects that operate on the inputs
                           *after* embedding.
            embed: If True, embeds inputs to network before passing through
                   `hidden_layers`.
            embed_networks: dictionary of embedding layers (configured the
                            same as `hidden_layers` i.e. as list of nn.Module
                            objects). The dictionary should have the same
                            number of entries as inputs to the network, and
                            should be formatted as follows:
                            `{'net_%d' % <input index> : list of modules}`
                            If no embedding is required for a particular input,
                            then the corresponding list in the dictionary
                            should contain a NoneType object i.e.
                            `'net_i' : list(None)`. If no networks are input,
                            forward pass defaults to no-embedding scenario.
            heteroscedastic(bool): If True, addition of inputs after embedding
                                   is heteroscedastic i.e. inputs[0] +
                                   inputs[0] * inputs[1]
        """
        super(Generator, self).__init__(hidden_layers)
        self.embed = embed
        self.heteroscedastic = heteroscedastic
        if self.embed:
            embed_nets = {}
            for k, net in embed_networks.items():
                embed_nets[k] = nn.Sequential(*net)
            self.embed_nets = nn.ModuleDict(embed_nets)

    def _get_embeddings(self, inputs):
        """
        Return list of embedded inputs.

        If embedding network corresponding to a particular input is None,
        then embedding = input.

        Args:
            inputs (list): list of torch.Tensors; inputs to embedding networks.
        """
        embedded_inputs = []
        # Loop through embedding networks
        for (k, net), input_to_net in zip(sorted(self.embed_nets.items()), inputs):
            # If there is no embedding network corresponding to this input,
            # then take the input as is
            if isinstance(net[0], type(None)):
                embedded_inputs.append(input_to_net)
            # Otherwise, get embedding corresponding to input
            else:
                output = net(input_to_net)
                embedded_inputs.append(output)
        return embedded_inputs

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            inputs: list of torch.Tensors; input to generator / embedding
                    network

        Returns:
            Output of the network
        """
        input_to_hidden_network = inputs
        if type(inputs) == list and len(inputs) < 2:
            input_to_hidden_network = inputs[0]

        if self.embed:
            assert hasattr(self, "embed_nets")
            assert len(self.embed_nets) == len(inputs)
            embedded_inputs = self._get_embeddings(inputs)
            input_to_hidden_network = _add(
                embedded_inputs, heteroscedastic=self.heteroscedastic
            )
        return self._hidden_layers(input_to_hidden_network)


class Discriminator(Generator):
    """GAN discriminator network."""

    def __init__(
        self,
        hidden_layers: List[nn.Module],
        conditional: Optional[bool] = True,
        embed: Optional[bool] = False,
        embed_networks: Optional[dict] = dict(net_1=[None], net_2=[None]),
    ):
        """
        GAN discriminator network.

        Args:
            hidden_layers: Same as Generator input `hidden_layers`.
            conditional: If True, inputs are concatenated along the last
                         dimension before being passed through network.
                         Default is True.
            embed: Same as Generator input `embed`.
            embed_networks: Same as Generator input `embed_networks`.
        """
        super(Discriminator, self).__init__(
            hidden_layers=hidden_layers,
            embed=embed,
            embed_networks=embed_networks,
            heteroscedastic=False,
        )
        self.conditional = conditional

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            inputs: list of torch.Tensors; input to discriminator / embedding
                    network

        Returns:
            Output of the network
        """
        input_to_hidden_network = inputs
        if self.conditional and not self.embed:
            assert isinstance(inputs, list)
            input_to_hidden_network = _concatenate(inputs)
        elif self.embed:
            assert hasattr(self, "embed_nets")
            assert len(self.embed_nets) == len(inputs)

            embedded_inputs = self._get_embeddings(inputs)
            input_to_hidden_network = _concatenate(embedded_inputs)
        return self._hidden_layers(input_to_hidden_network)
