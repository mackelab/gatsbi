from time import time
from typing import Callable, Optional

import torch
from pyro.distributions import MultivariateNormal as MvN
from pyro.distributions.rejector import Rejector
from pyro.infer.mcmc import HMC, MCMC

from gatsbi.optimize.base import Base
from gatsbi.optimize.utils import _check_data_bank
from gatsbi.utils import Classifier


class RejectionSample(Rejector):
    """Rejection sampling in pyro."""

    def __init__(self, num_particles=10, **kwargs):
        """
        Set up rejection sampling.

        Args:
            num_particles: maximum number of times to repeat rejection
                           sampling.
        """
        super(RejectionSample, self).__init__(**kwargs)
        self.num_particles = num_particles

    def rsample(
        self, sample_shape: Optional[torch.Size] = torch.Size()
    ) -> torch.tensor:
        """Rejection sample."""
        # Implements parallel batched accept-reject sampling.
        x = self.propose(sample_shape) if sample_shape else self.propose()
        probs = self.log_prob_accept(x).clamp_(0.0, 1.0)

        # probs = torch.exp(log_prob_accept).clamp_(0.0, 1.0)
        done = torch.bernoulli(probs).bool()
        ct = 0
        while not done.all() and ct < self.num_particles:
            proposed_x = self.propose(sample_shape)
            prob_accept = self.log_prob_accept(proposed_x).clamp_(0.0, 1.0)
            # prob_accept = torch.exp(log_prob_accept).clamp_(0.0, 1.0)
            accept = torch.bernoulli(prob_accept).bool()
            if accept.any():
                x[accept] = proposed_x[accept]
                done |= accept
            ct += 1
        print("successfully sampled)", sum(done), "/", len(done))
        return x


class SequentialOpt(Base):
    """Sequential GAN training with importance weights correction."""

    def __init__(
        self,
        seq_type: str,
        classifier_theta: Optional[Classifier] = None,
        classifier_obs: Optional[Classifier] = None,
        classifier_theta_kwargs: Optional[dict] = {},
        classifier_obs_kwargs: Optional[dict] = {},
        latent_distribution: Optional[Callable] = None,
        lat_dim: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Set up sequential optimiser.

        Args:
            seq_type: 'impwts' for importance weights correction, 'invimpwts'
                       for inverse important weights correction, 'ebm_mcmc'
                       for EBM correction with MCMC sampling, 'ebm_rej' for
                       EBM correction with rejection sampling.
            classifier_theta: trained classifier in theta space. If None, a
                              classifier will be created from
                              `classifier_theta_kwargs` and trained before
                              GAN training commences.
            classifier_obs: trained classifier in observation space. If None,
                            a classifier will be created from
                            `classifier_obs_kwargs` and trained before
                            GAN training commences.
            classifier_theta_kwargs: dictionary of arguments for input to
                                     classifier for theta samples. See
                                     Classifier docs for more information.
            classifier_obs_kwargs: dictionary of arguments for input to
                                   classifier for observations from simulator.
                                   See Classifier docs for more information.
            latent_distribution: pyro.distributions object, distribution for
                                 sampling latent variables. If None, defaults
                                 to standard multivariate Gaussian.
            lat_dim: number of latent dimensions. Must have integer input if
                     input to latent_distribution is None.
            **kwargs: arguments for Base optimiser. See
                      gatsbi.optimize.base.Base docs for more information.
            NOTE: if round_number > 1, input `prior` should be the proposal
                  prior i.e. prior for the corrent round;
                  training_opts.num_simulations should be simulation for the
                  current round
        """
        super(SequentialOpt, self).__init__(**kwargs)
        self.seq_type = seq_type
        self.classifier_theta = classifier_theta
        self.classifier_obs = classifier_obs
        self.classifier_theta_kwargs = classifier_theta_kwargs
        self.classifier_obs_kwargs = classifier_obs_kwargs
        self.lat_dist = latent_distribution
        self.lat_dim = lat_dim

        if "ebm" in self.seq_type:
            self._make_latent_distribution()
            if "mcmc" in self.seq_type:
                self.generator.float()
                assert hasattr(self.training_opts, "warmup_steps")
            elif "rej" in self.seq_type:
                assert (
                    hasattr(self.training_opts, "num_particles")
                    and self.training_opts.num_particles > 1
                )

        # If round number is > 0, samples from first round should already be
        # in self.dataloader
        if self.round_number > 0:
            assert _check_data_bank(self.round_number - 1, self.dataloader)
            if self.classifier_theta is None:
                print("Training theta classifier")
                self.classifier_theta = self._train_classifier("theta")
            if self.classifier_obs is None:
                print("Training obs classifier")
                self.classifier_obs = self._train_classifier("x")

    # -------------------------------------------------------------------------------------------------
    # Functions common to all sequential methods

    def _train_classifier(self, theta_or_obs="theta"):
        dim, kwargs = (
            (0, self.classifier_theta_kwargs)
            if theta_or_obs == "theta"
            else (1, self.classifier_obs_kwargs)
        )

        s_prior = self.dataloader["%d" % 0].dataset.inputs[dim]
        s_prop = self.dataloader["%d" % (self.round_number)].dataset.inputs[dim]

        classifier = Classifier(s_prior, s_prop, **kwargs)
        return classifier

    def _correction_factor(self, theta, obs):
        if self.seq_type == "impwts":
            return self.classifier_theta.odds(
                theta, invert=False
            ) * self.classifier_obs.odds(obs, invert=True)
        elif self.seq_type == "invimpwts" or "ebm" in self.seq_type:
            return self.classifier_theta.odds(
                theta, invert=True
            ) * self.classifier_obs.odds(obs, invert=False)
        else:
            raise NotImplementedError

    # -------------------------------------------------------------------------------------------------
    # Functions modifying loss for explicit correction factor

    def _calc_loss(self, theta_fake, obs, theta=None):
        # Revert to Base optimiser if round number is 0 or for EBM correction
        if (self.sample_from_round == 0) or ("ebm" in self.seq_type):
            return super(SequentialOpt, self)._calc_loss(theta_fake, obs, theta)

        d_fake, d_real = self.discriminator([theta_fake, obs]), None
        correction0, correction1 = 1.0, 1.0

        if theta is not None:
            d_real = self.discriminator([theta, obs])
            if self.seq_type == "impwts":
                correction0 = self._correction_factor(theta, obs)

        if self.seq_type == "invimpwts":
            correction1 = self._correction_factor(theta_fake, obs)

        return self.loss(d_fake, d_real, [-correction0, -correction1]).mean()

    # -------------------------------------------------------------------------------------------------
    # Functions modifying call to generator forward pass for EBM correction

    def _make_latent_distribution(self):
        if self.lat_dist is None:
            assert self.lat_dim is not None
            self.lat_dist = MvN(
                torch.zeros(self.lat_dim).to(self.device),
                torch.eye(self.lat_dim).to(self.device),
            )
        else:
            assert hasattr(self.lat_dist, "log_prob")

    def _potential_funcn(self, single_obs):
        def potential_energy(inp):
            z = inp["z"]

            # need to this so that MCMC does not break when we update z / obs
            z_torch = torch.cuda.FloatTensor(z)
            obs_torch = torch.cuda.FloatTensor(single_obs).unsqueeze(0)

            gz = self.generator([z_torch, obs_torch])
            corr_fac = self._correction_factor(gz, single_obs.unsqueeze(0))
            log_ratio = torch.log(corr_fac)
            return -self.lat_dist.log_prob(z) - log_ratio

        return potential_energy

    def _mcmc_sample(self, obs):
        z_samples = []
        for single_obs in obs:
            hmc = HMC(
                potential_fn=self._potential_funcn(single_obs), adapt_step_size=False
            )
            z_init = self.lat_dist.sample(torch.Size([1]))
            mcmc = MCMC(
                hmc,
                num_samples=1,
                warmup_steps=self.training_opts.warmup_steps,
                initial_params={"z": z_init},
            )
            mcmc.run()
            z_samples.append(mcmc.get_samples()["z"])
        z_samples = torch.cat(z_samples, 0).squeeze()
        return z_samples

    def _scale_factor(self, obs):
        z = self.lat_dist.sample(torch.Size([len(obs)]))
        prob_z = torch.exp(self.lat_dist.log_prob(z))
        gz = self.generator([z, obs])
        prob_accept = self._correction_factor(gz, obs)
        return (prob_z * prob_accept).max()

    def _rej_sample(self, obs):
        tic = time()

        def prob_accept(x):
            gz = self.generator([x, obs]).detach()
            lpa = self._correction_factor(gz, obs) / self._scale_factor(obs)
            return lpa

        zt = RejectionSample(
            num_particles=self.training_opts.num_particles,
            propose=self.lat_dist,
            log_prob_accept=prob_accept,
            log_scale=None,
        ).rsample(torch.Size([len(obs)]))
        print("time for rej samp", time() - tic)
        return zt

    def _fwd_pass_generator(self, obs):
        if "ebm" not in self.seq_type:
            return super(SequentialOpt, self)._fwd_pass_generator(obs)

        if (self.sample_from_round == 0) and ("ebm" in self.seq_type):
            z = self.lat_dist.sample(torch.Size([len(obs)])).squeeze()

            # Need this because lat_dist / MCMC sampling returns Double
            # tensors and networks take FloatTensors
            if self.device.type == "cuda":
                z = torch.cuda.FloatTensor(z)
                obs = torch.cuda.FloatTensor(obs)
            else:
                z = torch.FloatTensor(z)
                obs = torch.FloatTensor(z)
            return super(SequentialOpt, self)._fwd_pass_generator([z, obs])

        # Get z_samples via MCMC
        if self.seq_type == "ebm_mcmc":
            z_samples = self._mcmc_sample(obs)
        # Get z samples by rejection sampling
        elif self.seq_type == "ebm_rej":
            z_samples = self._rej_sample(obs)

        # generator forward pass
        return self.generator([z_samples, obs])
