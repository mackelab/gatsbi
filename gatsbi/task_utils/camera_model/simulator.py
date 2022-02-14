import scipy
import scipy.ndimage
import torch
from skimage.util import random_noise
from torch import Tensor


def grayscale_camera(
    theta: Tensor,
    noise: str = "poisson",
    psf_width: float = 3.0,
    noise_scale: float = 1.0,
    noise_gain: float = 1.0,
) -> Tensor:
    """
    Return noisy blurred grayscale image.

    Args:
        theta: input image to be blurred.
        noise: noise type.
        psf_width: width of point-spread function.
        noise_scale: scale for noise distribution.
        noise_gain: gain for noise distribution.
    """
    assert (
        theta.ndim == 2
    ), f"""Camera needs input shape (image, image),
                                 but shape is {theta.shape}."""
    # M, N = theta.shape
    # assert M == N, "Image must be quadratic"

    # Add noise
    image1 = noise_gain * random_noise(noise_scale * theta, mode=noise)
    # Add point spread via Gaussian filter
    image2 = scipy.ndimage.gaussian_filter(image1, sigma=psf_width)
    return torch.FloatTensor(image2)


class GrayScaleSim:
    """Noisy blurred grayscale simulator."""

    def __init__(self, **kwargs):
        """Return noisy blurred grayscale image."""
        self.kwargs = kwargs

    def __call__(self, theta: Tensor) -> Tensor:
        """Call to simulator."""
        return self.grayscale_simulator(theta).unsqueeze(1)

    def grayscale_simulator(self, theta: Tensor,) -> Tensor:
        """Forward pass."""
        return torch.stack(
            [grayscale_camera(th.squeeze(), **self.kwargs) for th in theta]
        )
