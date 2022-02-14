from os import makedirs
from os.path import join
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn

from sbibm.metrics.c2st import c2st


class MakeResults:
    def __init__(
        self,
        generator: nn.Module,
        task: Callable,
        seq_impwts: Optional[str] = "impwts",
        lat_dist: Optional[Callable] = None,
        save_dir: Optional[str] = None,
    ) -> None:
        """
        Get posterior samples from the generator, posterior predictive samples
        from the simulator and calculate corresponding metrics.

        Args:
            generator: generator network from which to sample parameters.
            task: sbibm task object.
            seq_impwts: importance weights type.
            lat_dist: latent distribution sampler for EBM-based sequential estimation.
            save_dir: path to directory to save generated samples.
        """
        self.generator = generator
        self.task = task
        self.seq_impwts = seq_impwts
        self.lat_dist = lat_dist
        self.save_dir = save_dir

        if "ebm" in self.seq_impwts:
            assert self.lat_dist is not None

        if self.save_dir is not None:
            makedirs(join(self.save_dir, "gatsbi_results"), exist_ok=True)

    @torch.no_grad()
    def make_posterior_samples(self, obs_num: int) -> torch.Tensor:
        """
        Get posterior samples from generator network with rejection sampling.

        Args:
            obs_num: observation number from `sbibm` database conditioned on
                     which to sample from the GATSBI posterior.
        """
        # get gen samples
        num_samples = len(self.task.get_reference_posterior_samples(obs_num))

        # need only the upper bound, because prior bounds are symmetric for both apps
        rej_thresh = self.task.prior_params["high"]

        self.generator.cuda()
        obs = self.task.get_observation(obs_num).cuda()

        # Rejection sample
        tot_samples = 0
        i = 0
        sample_size = num_samples
        gatsbi_samples = []
        while sample_size > 0 and tot_samples < (10 * num_samples):
            sample_size -= i
            tot_samples += sample_size
            obs_repeat = obs.repeat(sample_size, 1).cuda()
            if "ebm" not in self.seq_impwts:
                samp = self.generator.forward(obs_repeat).data.cpu()
            else:
                z = self.lat_dist.sample(torch.Size([sample_size]))
                samp = self.generator.forward([z, obs_repeat]).data.cpu()

            inds = torch.all(torch.abs(samp) < rej_thresh, -1)
            i = sum(inds)
            gatsbi_samples.append(samp[inds])

        gatsbi_samples = torch.cat(gatsbi_samples, 0)

        # Save generated samples
        if self.save_dir is not None:
            np.save(
                join(
                    self.save_dir,
                    "gatsbi_results",
                    "gatsbi_samples_obs_num_%d.npy" % obs_num,
                ),
                gatsbi_samples.numpy(),
            )
        return gatsbi_samples

    def make_post_pred_samples(
        self, gatsbi_samples: torch.Tensor, obs_num: int
    ) -> torch.Tensor:
        """
        Get posterior predictive samples from GATSBI posterior samples.

        Args:
            gatsbi_samples: samples from GATSBI posterior.
            obs_num: observation number corresponding to GATSBI samples.
        """
        post_pred_samples = self.task.get_simulator()(gatsbi_samples)
        if self.save_dir is not None:
            np.save(
                join(
                    self.save_dir,
                    "gatsbi_results",
                    "gatsbi_ppc_samples_obs_num_%d.npy" % obs_num,
                ),
                post_pred_samples.numpy(),
            )
        return post_pred_samples

    def calc_c2st(self, obs_num: int) -> dict:
        """
        Calculate C2ST for a particular observation (saves
        corresponding posterior and posterior predictive samples).

        Args:
            obs_num: observation number for which to calculate metrics.
        """
        reference_posterior = self.task.get_reference_posterior_samples(obs_num)

        gatsbi_samples = self.make_posterior_samples(obs_num)
        _ = self.make_post_pred_samples(gatsbi_samples, obs_num)

        # calculate metrics
        c2st_val = c2st(reference_posterior, gatsbi_samples)
        return {"c2st_%d" % obs_num: c2st_val.data.numpy()}

    def calc_c2st_all_obs(self) -> dict:
        """
        Calculate C2ST for all observations in `sbibm` test set.
        """
        all_obs_seeds = self.task.observation_seeds

        c2st_list = []
        for num, _ in enumerate(all_obs_seeds, start=1):
            print(num)
            c2st_val = self.calc_c2st(num)["c2st_%d" % num]
            c2st_list.append(c2st_val)

        return {
            "c2st_mean": np.mean(c2st_list),
            "c2st_std": np.std(c2st_list),
            "c2st": c2st_list,
        }
