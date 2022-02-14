from argparse import Namespace as NSp
from typing import Callable, Optional

import torch
import torch.nn as nn
from pyro.distributions import MultivariateNormal as MvN

from gatsbi.networks import (AddNoise, BaseNetwork, Discriminator, Generator,
                             nonlin_dict)


class ProposalWrapper:
    """Wrapper class for proposal sampling."""

    def __init__(
        self,
        generator: Callable,
        observation: torch.tensor,
        seq_impwts: Optional[str] = "impwts",
        lat_dist: Optional[Callable] = None,
        lat_dim: Optional[int] = None,
    ) -> None:
        """
        Wrap proposal for sampling.

        Args:
            generator: generator network.
            observation: observation on which to condition.
            seq_impwts: type of importance weights.
            lat_dist: latent distribution sampler for EBM-based
                      sequential estimation.
            lat_dim: number of latent distributions.
        """
        self.generator = generator
        self.observation = observation
        self.seq_impwts = seq_impwts
        self.lat_dist = lat_dist
        if ("ebm" in seq_impwts) and (self.lat_dist is None):
            self.lat_dist = MvN(torch.zeros(lat_dim), torch.eye(lat_dim))

    def prior(self, num_samples: int) -> Callable:
        """
        Sample from generator.

        Args:
            num_samples: number of samples
        """
        obs = self.observation.repeat(num_samples, 1)
        if "ebm" in self.seq_impwts:
            z = self.lat_dist.sample(torch.Size([len(obs)])).to("cpu").squeeze()
            return self.generator.cpu().forward([z, obs]).data
        else:
            return self.generator.cpu().forward(obs).data


def make_generator(
    gen_seed: int = None, seq_impwts: Optional[str] = "impwts", **gen_network_kwargs
) -> BaseNetwork or Generator:
    """
    Make generator.

    Args:
        gen_seed: random seed for generator architecture.
        seq_impwts: type of importance weights.
    """
    if type(gen_seed) == int:
        torch.manual_seed(gen_seed)

    gnk = NSp(**gen_network_kwargs)
    gen_layers = [
        [
            nn.Linear(un1, un2, bias=True),
            nonlin_dict[gnk.gen_nonlin](**gnk.gen_nonlin_kwargs),
        ]
        for un1, un2 in zip(gnk.gen_units[:-1], gnk.gen_units[1:])
    ]
    gen_layers = [ll for g in gen_layers for ll in g]
    if "ebm" in seq_impwts:
        embed_net_1 = [
            nn.Linear(
                gnk.add_noise_kwargs["lat_dim"],
                gnk.add_noise_kwargs["output_dim"],
                bias=False,
            )
        ]
        embed_net_2 = gen_layers[: gnk.add_noise_layer]
        hidden_layers = gen_layers[gnk.add_noise_layer :]
        gen = Generator(
            hidden_layers=hidden_layers,
            embed=True,
            embed_networks={"net_1": embed_net_1, "net_2": embed_net_2},
            heteroscedastic=gnk.add_noise_kwargs["heteroscedastic"],
        )
    else:
        gen_hidden_layers = [
            *gen_layers[: gnk.add_noise_layer],
            AddNoise(**gnk.add_noise_kwargs),
            *gen_layers[gnk.add_noise_layer :],
        ]
        gen = BaseNetwork(gen_hidden_layers)
    return gen


def make_discriminator(
    dis_seed: Optional[int] = None, **dis_network_kwargs
) -> Discriminator:
    """
    Make discriminator.

    Args:
        dis_seed: random seed for discriminator architecture.

    Returns:
        Discriminator: [description]
    """
    dis_units = dis_network_kwargs["dis_units"]
    dis_nonlin = dis_network_kwargs["dis_nonlin"]
    dis_nonlin_kwargs = dis_network_kwargs["dis_nonlin_kwargs"]

    if type(dis_seed) == int:
        torch.manual_seed(dis_seed)
    dis_hidden_layers = [
        [
            nn.utils.spectral_norm(nn.Linear(un1, un2, bias=True)),
            nonlin_dict[dis_nonlin](**dis_nonlin_kwargs),
        ]
        for un1, un2 in zip(dis_units[:-1], dis_units[1:])
    ]
    dis_hidden_layers[-1][-1] = nn.Sigmoid()
    dis_hidden_layers = [ll for g in dis_hidden_layers for ll in g]
    embed_net_1 = [None]
    embed_net_2 = [None]

    dis = Discriminator(
        dis_hidden_layers,
        conditional=True,
        embed=True,
        embed_networks={"net_1": embed_net_1, "net_2": embed_net_2},
    )
    return dis


def load_generator(
    state_dict: dict,
    gen: BaseNetwork or Generator,
    seq_impwts: Optional[str] = "impwts",
) -> BaseNetwork or Generator:
    """
    Load generator from state dict.

    The method is specifically for taking regular generator and loading
    for EBM-based sequential estimation.
    Args:
        state_dict: saved state dictionary for generator.
        gen: randomly initialised generator network.
        seq_impwts: type of importance weights.
    """
    if "ebm" in seq_impwts:
        key_w = [k for k in state_dict.keys() if "W" in k][0]
        gen.embed_nets["net_1"][0].weight.data = state_dict[key_w]

        keys_emb = [kk for kk in state_dict.keys() if int(kk[15]) < int(key_w[15])]
        for k1, k2 in zip(keys_emb[::2], keys_emb[1::2]):
            assert k1[15] == k2[15]
            gen.embed_nets["net_2"][int(k1[15])].weight.data = state_dict[k1]
            gen.embed_nets["net_2"][int(k2[15])].bias.data = state_dict[k2]

        keys_hid = [kk for kk in state_dict.keys() if int(kk[15]) > int(key_w[15])]

        for k1, k2 in zip(keys_hid[::2], keys_hid[1::2]):
            assert k1[15] == k2[15]
            gen._hidden_layers[int(k1[15]) - len(keys_emb)].weight.data = state_dict[k1]
            gen._hidden_layers[int(k2[15]) - len(keys_emb)].bias.data = state_dict[k2]
    else:
        gen.load_state_dict(state_dict)
    return gen
