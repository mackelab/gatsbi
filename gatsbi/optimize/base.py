from argparse import Namespace
from time import time
from typing import Callable, Optional, Tuple

import numpy as np
import pandas
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

import gatsbi.utils as utils
from gatsbi.networks import BaseNetwork, Discriminator, Generator

from .utils import (_check_data_bank, _log_metrics, _make_checkpoint, _sample,
                    _stop_training)


class Base:
    """Base class for amortised GATSBI training."""

    def __init__(
        self,
        generator: BaseNetwork or Generator,
        discriminator: Discriminator,
        prior: Callable,
        simulator: Callable,
        optim_args: Tuple[float, Tuple[float, float]],
        dataloader: Optional[dict] = dict(),
        loss: Optional[str] = "cross_entropy",
        round_number: Optional[int] = 0,
        reuse_samples: Optional[bool] = False,
        training_opts: Optional[dict] = dict(
            gen_iter=1,
            dis_iter=1,
            max_norm_gen=np.inf,
            max_norm_dis=np.inf,
            num_simulations=1000,
            sample_seed=42,
            hold_out=100,
            batch_size=1000,
            log_dataloader=True,
        ),
        logger: Optional[Callable] = None,
    ) -> None:
        """
        Set up base class for amortised GATSBI training.

        Args:
            generator: generator network that takes either observations, or
                       observations and latents as inputs.
            discriminator: discriminator network that takes observations and
                           parameters as inputs.
            prior: prior distribution function that takes number of samples as
                   input and returns corresponding number of parameter samples.
            simulator: simulator function that takes parameter samples as
                       input and returns corresponding observations.
            *optim_args: arguments for ADAM optimiser corresponding to
                         generator and discriminator, in that order.
            dataloader: dictionary with round numbers as keys, and entries as
                        MakeDataset objects.
            loss: 'cross_entropy', 'kl_div' or 'wasserstein'.
                   See gatsbi.utils.loss_funcns docs for more information
            round_number: round number for training
            reuse_samples: if True, reuse samples from previous rounds for
                           training.
            training_opts: hyper-parameters for training GAN
                gen_iter (int): number of generator updates per epoch. Default
                                is 1.
                dis_iter (int): number of discriminator updates per epoch.
                                Default is 1.
                max_norm_gen (float): maximum value to clip generator
                                      gradients. If None, gradients are not
                                      clipped.
                max_norm_dis (float): maximum value to clip discriminator
                                      gradients. If None, gradients are not
                                      clipped.
                num_simulations (int): simulation budget i.e. maximum number
                                       of calls to simulator allowed across all
                                       rounds. Default is 1000.
                samples_seed (int): random seed for prior/simulator sampling.
                                    Default is 42.
                hold_out (int): number of prior samples / observations to hold
                                out as test data. Default is 100.
                batch_size (int): batch size for gradient descent.
            logger: wandb.run object for logging data. If None, data is not
                    logged.
        """
        # Set constants
        self.epoch_ct = 0
        self.round_number = round_number
        self.reuse_samples = reuse_samples
        # keep track of which round the samples are from
        self.sample_from_round = self.round_number
        self.df = pandas.DataFrame(
            columns=[
                "dis_loss",
                "gen_loss",
                "dreal_mean",
                "dreal_std",
                "dfake_mean",
                "dfake_std",
                "dis_grad",
                "gen_grad",
                "global_step",
            ]
        )
        self.start = time()

        # Initialise networks / callables
        self.generator = generator
        self.discriminator = discriminator
        self.simulator = simulator
        self.prior = prior

        self.dataloader = dataloader
        self.loss = getattr(utils, loss)
        self.logger = logger

        # Initialise optimisers
        gen_opt_args, dis_opt_args = optim_args[0], optim_args[1]
        self.generator_optim = Adam(self.generator.parameters(), *gen_opt_args)
        self.discriminator_optim = Adam(self.discriminator.parameters(), *dis_opt_args)

        # Set device and check that all networks are on the same device --
        # need this to flexibly switch between training on cpu and gpu
        self.device = list(self.discriminator.parameters())[0].device
        assert list(self.discriminator.parameters())[0].device == self.device

        # Hyper-parameters for training
        self.training_opts = Namespace(**training_opts)

        # Make training data loader
        if not _check_data_bank(self.round_number, self.dataloader):
            data = _sample(
                prior=self.prior,
                simulator=self.simulator,
                sample_seed=self.training_opts.sample_seed,
                num_samples=self.training_opts.num_simulations,
            )
            inputs_to_loader_class = {
                "inputs": data,
                "hold_out": self.training_opts.hold_out,
            }
            loader = utils.make_loader(
                self.training_opts.batch_size,
                inputs_to_loader_class,
                loader_class=utils.MakeDataset,
            )
            self.dataloader[str(self.round_number)] = loader

        # Logging progress
        if self.logger is not None:
            _make_checkpoint(self, init=True)

    def _fwd_pass_generator(self, obs):
        return self.generator(obs)

    def _calc_loss(self, theta_fake, obs, theta=None):
        d_fake, d_real = self.discriminator([theta_fake, obs]), None
        if theta is not None:
            d_real = self.discriminator([theta, obs])
        return self.loss(d_fake, d_real).mean()

    def _update_discriminator(self, theta, obs):
        # Zero gradients
        self.discriminator_optim.zero_grad()

        # Prepare data
        theta, obs = theta.to(self.device), obs.to(self.device)
        theta_fake = self._fwd_pass_generator(obs).detach()

        # Loss and backward
        loss_val = self._calc_loss(theta_fake, obs, theta)
        loss_val.backward(retain_graph=True)

        # Clip gradients
        if self.training_opts.max_norm_dis < np.inf:
            clip_grad_norm_(
                self.discriminator.parameters(),
                max_norm=self.training_opts.max_norm_dis,
            )

        # Update parameters
        self.discriminator_optim.step()

    def _update_generator(self, obs):
        # Zero gradients
        self.generator_optim.zero_grad()

        # Prepare data
        obs = obs.to(self.device)
        theta_fake = self._fwd_pass_generator(obs)

        # Loss and backward
        loss_val = self._calc_loss(theta_fake, obs)
        loss_val.backward(retain_graph=True)

        # Clip gradients
        if self.training_opts.max_norm_gen < np.inf:
            _ = clip_grad_norm_(
                self.generator.parameters(), max_norm=self.training_opts.max_norm_gen
            )

        # Update parameters
        self.generator_optim.step()

    def _data_iterator(self, iter_limit):
        if not self.reuse_samples:
            dataloader = self.dataloader[str(self.round_number)]
            return [
                (i, theta, obs, self.round_number)
                for i, (_, (theta, obs)) in enumerate(dataloader)
                if i < iter_limit
            ]
        elif self.reuse_samples:
            round_choice = np.random.choice(
                range(self.round_number + 1), size=iter_limit, replace=True,
            )
            dataloaders = [
                self.dataloader[str(k)] for k in range(self.round_number + 1)
            ]
            iterator = [
                data for i, data in enumerate(zip(*dataloaders)) if i < iter_limit
            ]
            iterator2 = [(data[rnd], rnd) for data, rnd in zip(iterator, round_choice)]
            iterator3 = [
                (i, dat[1][0], dat[1][1], rnd)
                for (i, (dat, rnd)) in enumerate(iterator2)
            ]
            return iterator3

    def train(self, epochs: int, log_freq: Optional[int] = 1000) -> None:
        """
        Train GATSBI.

        Args:
            epochs: number of training epochs.
            log_freq: frequency at which to checkpoint.
        """
        stop_training = False
        epoch = 0
        while not stop_training and (epoch < epochs):
            epoch += 1
            self.epoch_ct += 1

            # Train discriminator
            for (i, theta, obs, rnd) in self._data_iterator(
                self.training_opts.dis_iter
            ):
                print("Dis iter %d" % i, rnd)
                self.sample_from_round = rnd
                tic = time()
                self._update_discriminator(theta, obs)
                print("Time", time() - tic)
            torch.cuda.empty_cache()

            # Train generator
            for (i, _, obs, rnd) in self._data_iterator(self.training_opts.gen_iter):
                print("Gen iter %d" % i)
                self.sample_from_round = rnd
                tic = time()
                self._update_generator(obs)
                print("Time", time() - tic)
            torch.cuda.empty_cache()

            # Log metrics and stop training
            if (self.epoch_ct % log_freq == 0) or (epoch == epochs):
                print("Logging metrics")
                _log_metrics(self)
                if self.logger is not None:
                    _make_checkpoint(self, init=False)

                stop_training = _stop_training(self)
