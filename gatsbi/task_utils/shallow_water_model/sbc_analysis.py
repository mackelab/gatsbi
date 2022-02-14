from os.path import join

import numpy as np
import torch
import torch.nn as nn


def get_rank_statistic(
    generator: nn.Module,
    path_to_samples: str,
    num_dims: int = 100,
    num_samples: int = None,
    save: bool = False,
    save_dir: str = None,
):
    """
    Calculate rank statistics.

    generator: trained GATSBI generator network.
    path_to_samples: file from which to load groundtruth samples.
    num_dims: number of samples for which to calculate rank.
    num_samples: total number of test samples.
    save: if True, save ranks as npz file.
    save_dir: location at which to save ranks.
    """
    generator.cuda()
    sbc = np.load(path_to_samples)
    thos = torch.FloatTensor(sbc["depth_profiles"])
    xos = torch.FloatTensor(sbc["z_vals"])

    # Calculate ranks
    if num_samples is None:
        num_samples = len(thos)
    else:
        thos = thos[:num_samples]
        xos = xos[:num_samples]
    ndim = thos.shape[1]
    ranks = [[] for _ in range(ndim)]

    f = torch.distributions.Normal(loc=torch.zeros(1), scale=10)
    all_samples = []
    for k, (tho, xo) in enumerate(zip(thos.squeeze(), xos.squeeze())):
        samples = []
        for j in range(num_dims):
            samples.append(
                generator(torch.FloatTensor(xo).cuda().unsqueeze(0)).data.cpu() + 10
            )

        samples = torch.cat(samples, 0)
        all_samples.append(samples.unsqueeze(0))
        # Calculate rank under Gaussian.
        for i in range(ndim):
            slp = f.log_prob(samples[:, i])
            gtlp = f.log_prob(tho[i])
            rr = (slp < gtlp).sum().item()
            ranks[i].append(rr)
    all_samples = torch.cat(all_samples, 0)
    if save:
        np.savez(join(save_dir, "gatsbi_sbc.npz"), ranks=ranks, samples=all_samples)
    return np.array(ranks), all_samples
