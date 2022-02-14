import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal as MvN

from gatsbi.networks.base import BaseNetwork
from gatsbi.networks.models import Discriminator
from gatsbi.networks.modules import AddNoise
from gatsbi.optimize import Base as Optimize
from gatsbi.optimize import UnrolledOpt


def _prior(num_samples):
    nr = MvN(torch.zeros(10), covariance_matrix=torch.eye(10))
    return nr.sample(torch.Size([num_samples]))


def _simulator(prior_samples):
    nr = MvN(torch.zeros(10), covariance_matrix=0.1 * torch.eye(10))
    return nr.sample(torch.Size([len(prior_samples)])) + prior_samples


class TestBaseOptimiser(unittest.TestCase):
    """Test setup and training pass through base optimiser."""

    def test_base_optimizer_training_iteration(self):
        """Test setup and 2 epochs of updates through optimiser."""
        gen = BaseNetwork([nn.Linear(10, 10), AddNoise(10, 10), nn.LeakyReLU()])
        dis = Discriminator([nn.Linear(20, 1), nn.Sigmoid()], conditional=True)

        gen_opt_args = [0.0001, (0.5, 0.99)]
        dis_opt_args = [0.0001, (0.5, 0.99)]

        loss = "cross_entropy"
        training_opts = {
            "gen_iter": 1,
            "dis_iter": 1,
            "max_norm_gen": np.inf,
            "max_norm_dis": np.inf,
            "num_simulations": 20,
            "sample_seed": 42,
            "hold_out": 1,
            "batch_size": 10,
        }

        opt = Optimize(
            gen,
            dis,
            _prior,
            _simulator,
            [gen_opt_args, dis_opt_args],
            loss_str=loss,
            training_opts=training_opts,
        )
        opt.train(epochs=2)
        self.assertEqual(opt.epoch_ct, 2)


class TestUnrolledGANOpt(unittest.TestCase):
    """Test unrolled GAN optimiser."""

    def test_unrolled_optimizer_training_iteration(self):
        """Test setup and 2 epochs of updates through optimiser."""
        gen = BaseNetwork([nn.Linear(10, 10), AddNoise(10, 10), nn.LeakyReLU()])
        dis = Discriminator([nn.Linear(20, 1), nn.Sigmoid()], conditional=True)

        gen_opt_args = [0.0001, (0.5, 0.99)]
        dis_opt_args = [0.0001, (0.5, 0.99)]

        loss = "cross_entropy"
        training_opts = {
            "gen_iter": 1,
            "dis_iter": 1,
            "max_norm_gen": np.inf,
            "max_norm_dis": np.inf,
            "num_simulations": 20,
            "sample_seed": 42,
            "hold_out": 1,
            "batch_size": 10,
            "unroll_steps": 1,
        }

        opt = UnrolledOpt(
            generator=gen,
            discriminator=dis,
            prior=_prior,
            simulator=_simulator,
            optim_args=[gen_opt_args, dis_opt_args],
            loss_str=loss,
            training_opts=training_opts,
        )
        opt.train(epochs=2)
        dict_true = [
            torch.any(val1 != val2)
            for val1, val2 in zip(
                opt.dis_state_dict.values(), opt.discriminator.state_dict().values()
            )
        ]
        self.assertEqual(opt.epoch_ct, 2) and self.assertTrue(np.any(dict_true))
