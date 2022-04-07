from copy import deepcopy
from time import time
from typing import Optional
from os.path import join

import torch

from gatsbi.optimize.base import Base
from gatsbi.optimize.utils import (_log_metrics, _make_checkpoint,
                                   _stop_training)


class UnrolledOpt(Base):
    """Unrolled GAN implementation."""

    def __init__(self, **kwargs):
        """Set up unrolled GAN optimisation."""
        super(UnrolledOpt, self).__init__(**kwargs)
        assert hasattr(self.training_opts, "unroll_steps")

        with torch.no_grad():
            self.dis_state_dict = deepcopy(self.discriminator).state_dict()

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
            self.discriminator.load_state_dict(self.dis_state_dict)
            for (i, theta, obs, rnd) in self._data_iterator(
                self.training_opts.dis_iter
            ):
                print("Dis iter %d" % i)
                self.sample_from_round = rnd
                tic = time()
                self._update_discriminator(theta, obs)
                print("Time", time() - tic)
            # Need to save and load statedict, since deepcopy doesnt work after discrim. fwd. pass
            torch.save(self.discriminator.state_dict(), join(self.logger.dir, "unrolled_discrim.pt"))
            self.dis_state_dict = torch.load(join(self.logger.dir, "unrolled_discrim.pt"))
#             self.dis_state_dict = deepcopy(self.discriminator).state_dict()
            torch.cuda.empty_cache()

            # Roll discriminator:
            for (i, theta, obs, rnd) in self._data_iterator(
                self.training_opts.unroll_steps
            ):
                print("Dis unroll %d" % i)
                self.sample_from_round = rnd
                tic = time()
                self._update_discriminator(theta, obs)
                print("Time", time() - tic)

            # Train generator
            for (i, _, obs, rnd) in self._data_iterator(self.training_opts.gen_iter):
                print("Gen iter %d" % i)
                self.sample_from_round = rnd
                tic = time()
                self._update_generator(obs)
                print("Time", time() - tic)
            torch.cuda.empty_cache()

            # Log metrics and stop training
            if self.epoch_ct % log_freq == 0:
                print("Logging metrics")
                if self.logger is not None:
                    _log_metrics(self)
                    _make_checkpoint(self, init=False)

                stop_training = _stop_training(self)
