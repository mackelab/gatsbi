from typing import Optional

import numpy as np
from torch import Tensor


def gaussian_kernel(size: int, sigma: float, tau: float) -> np.ndarray:
    """
    Gaussian smoothing kernel for prior.

    Args:
        size: size (length of sqaure) of the gaussian kernel matrix.
        sigma: amplitude of the Gaussian kernel
        tau:  correlation length-scale of kernel.

    Returns:
    Gaussian kernel matrix
    """
    x = np.arange(size)
    xg, yg = np.meshgrid(x, x)
    sq = (xg - yg) ** 2
    return sigma * np.exp(-sq / (2 * tau))


class DepthProfilePrior:
    """Class for setting up gaussian kernel and sampling depth profiles."""

    def __init__(
        self,
        size: Optional[int] = 100,
        sigma: Optional[float] = 15.0,
        tau: Optional[float] = 100.0,
        mean: Optional[float] = 10.0,
        return_seed: Optional[bool] = False,
    ):
        """
        Set up prior.

        Args:
            size: size (length of sqaure) of the gaussian kernel matrix.
            sigma: amplitude of the Gaussian kernel
            tau:  correlation length-scale of kernel.
            mean: average depth
            return_seed: if True, returns random-sampling seed.
        """
        self.size = size
        self.sigma = sigma
        self.tau = tau
        self.mean = mean
        self.return_seed = return_seed

    def __call__(self, num_samples: int = 1, seed: int = 42) -> Tensor:
        """Return random batch of depth profiles from multi-variate Gaussian."""
        return self.sample(num_samples, seed)

    def sample(self, num_samples: int, seed: Optional[int] = 42) -> np.ndarray:
        """
        Forward pass.

        num_samples: number of depth profile samples from prior
        seed: random-sampling seed. Default is 42.
        """
        cov = gaussian_kernel(self.size, self.sigma, self.tau)
        np.random.seed(seed)
        # self.mean = np.random.uniform(low=self.low, high=self.high)
        loc = np.ones(self.size) * self.mean
        depth_profile = np.random.multivariate_normal(loc, cov, size=num_samples)
        if self.return_seed:
            return depth_profile, seed
        else:
            return depth_profile
