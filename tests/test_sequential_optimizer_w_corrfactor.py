import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal as MvN

from gatsbi.networks.base import BaseNetwork
from gatsbi.networks.models import Discriminator, Generator
from gatsbi.networks.modules import AddNoise
from gatsbi.optimize.sequential import SequentialOpt as Optimize
from gatsbi.optimize.utils import _sample
from gatsbi.utils.load_data import MakeDataset, make_loader


def _prior(num_samples):
    nr = MvN(torch.zeros(10), covariance_matrix=torch.eye(10))
    return nr.sample(torch.Size([num_samples]))


def _proposal(num_samples):
    nr = MvN(torch.zeros(10), covariance_matrix=0.5 * torch.eye(10))
    return nr.sample(torch.Size([num_samples]))


def _simulator(prior_samples):
    nr = MvN(torch.zeros(10), covariance_matrix=0.1 * torch.eye(10))
    return nr.sample(torch.Size([len(prior_samples)])) + prior_samples


class TestSeqOptimiser(unittest.TestCase):
    """Test optimiser with corrections for sequential estimation."""

    def seq_optimizer_impwts_correction_fwdpass(self, round_number):
        """Test 2 epochs of training for importance weight correction."""
        # Make networks
        gen = BaseNetwork([nn.Linear(10, 10), AddNoise(10, 10), nn.LeakyReLU()])
        dis = Discriminator([nn.Linear(20, 1), nn.Sigmoid()], conditional=True)

        # Make prior samples and dataloader
        thetas, obs = _sample(_prior, _simulator, sample_seed=42, num_samples=20)
        dataloader = {
            "0": make_loader(10, {"inputs": [thetas, obs], "hold_out": 10}, MakeDataset)
        }

        # Set hyperparameters
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
            "impwts",
            generator=gen,
            discriminator=dis,
            prior=_prior,
            simulator=_simulator,
            optim_args=[gen_opt_args, dis_opt_args],
            dataloader=dataloader,
            loss_str=loss,
            round_number=round_number,
            training_opts=training_opts,
        )
        opt.train(epochs=2)
        return opt

    def seq_optimizer_ebm_correction_fwdpass(self, round_number):
        """Test 2 epochs of updates with EBM correction with MCMC."""
        # Make networks
        gen = Generator(
            hidden_layers=[nn.Linear(10, 10), nn.LeakyReLU()],
            embed=True,
            embed_networks={"net_1": [nn.Linear(10, 10)], "net_2": [nn.Linear(10, 10)]},
        )
        dis = Discriminator([nn.Linear(20, 1), nn.Sigmoid()], conditional=True)

        # Make prior samples and dataloader
        thetas, obs = _sample(_prior, _simulator, sample_seed=42, num_samples=20)
        dataloader = {
            "0": make_loader(
                10, {"inputs": [thetas, obs], "hold_out": 10}, loader_class=MakeDataset
            )
        }

        # Set hyperparameters
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
            "warmup_steps": 1,
            "batch_size": 10,
        }

        opt = Optimize(
            seq_type="ebm_mcmc",
            lat_dim=10,
            generator=gen,
            discriminator=dis,
            prior=_prior,
            simulator=_simulator,
            optim_args=[gen_opt_args, dis_opt_args],
            dataloader=dataloader,
            loss_str=loss,
            round_number=round_number,
            training_opts=training_opts,
        )
        opt.train(epochs=2)
        return opt

    def seq_optimizer_ebm_rejsample_fwdpass(self, round_number):
        """
        Test 2 epochs of updates with EBM correction and rejection
        sampling.
        """
        # Make networks
        gen = Generator(
            hidden_layers=[nn.Linear(10, 10), nn.LeakyReLU()],
            embed=True,
            embed_networks={"net_1": [nn.Linear(10, 10)], "net_2": [nn.Linear(10, 10)]},
        )
        dis = Discriminator([nn.Linear(20, 1), nn.Sigmoid()], conditional=True)

        # Make prior samples and dataloader
        thetas, obs = _sample(_prior, _simulator, sample_seed=42, num_samples=20)
        dataloader = {
            "0": make_loader(
                10, {"inputs": [thetas, obs], "hold_out": 10}, loader_class=MakeDataset
            )
        }

        # Set hyperparameters
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
            "num_particles": 100,
        }

        opt = Optimize(
            seq_type="ebm_rej",
            lat_dim=10,
            generator=gen,
            discriminator=dis,
            prior=_prior,
            simulator=_simulator,
            optim_args=[gen_opt_args, dis_opt_args],
            dataloader=dataloader,
            loss_str=loss,
            round_number=round_number,
            training_opts=training_opts,
        )
        opt.train(epochs=2)
        return opt

    def test_seq_impwts_optimizer_round0(self):
        """Test importance weights optimiser when round is 0."""
        opt = self.seq_optimizer_impwts_correction_fwdpass(0)
        self.assertEqual(opt.epoch_ct, 2) and self.assertFalse(
            hasattr(opt, "classifier_theta")
        ) and self.assertEqual(len(opt.dataloader.keys()), 1)

    def test_seq_impwts_optimizer_round1(self):
        """Test importance weights optimiser when round is 1."""
        opt = self.seq_optimizer_impwts_correction_fwdpass(1)
        self.assertEqual(opt.epoch_ct, 2) and self.assertTrue(
            hasattr(opt, "classifier_theta")
        ) and self.assertEqual(len(opt.dataloader.keys()), 2)

    def test_seq_ebm_optimizer_round0(self):
        """Test EBM optimiser when round is 0."""
        opt = self.seq_optimizer_ebm_correction_fwdpass(0)
        self.assertEqual(opt.epoch_ct, 2) and self.assertFalse(
            hasattr(opt, "classifier_theta")
        ) and self.assertEqual(len(opt.dataloader.keys()), 1)

    def test_seq_ebm_optimizer_round1(self):
        """Test EBM optimiser when round is 1."""
        opt = self.seq_optimizer_impwts_correction_fwdpass(1)
        self.assertEqual(opt.epoch_ct, 2) and self.assertTrue(
            hasattr(opt, "classifier_theta")
        ) and self.assertEqual(len(opt.dataloader.keys()), 2)

    def test_seq_ebm_optimizer_rejection_sample_round1(self):
        """Test EBM optimiser with rejection sampling when round is 1."""
        opt = self.seq_optimizer_ebm_rejsample_fwdpass(1)
        self.assertEqual(opt.epoch_ct, 2) and self.assertTrue(
            hasattr(opt, "classifier_theta")
        ) and self.assertEqual(len(opt.dataloader.keys()), 2)
