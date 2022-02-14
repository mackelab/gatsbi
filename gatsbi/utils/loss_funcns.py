from typing import Optional

import torch
from torch import Tensor


def cross_entropy(
    d_fake: Tensor, d_real: Optional[Tensor] = None, mult: Optional[list] = [-1.0, -1.0]
) -> Tensor:
    """
    Vanilla cross-entropy loss as in Goodfellow et al. 2014.

    Args:
        d_fake: discriminator output for fake data.
        d_real: discriminator output for real data. If None, the corresponding
                term is dropped from the cross entropy calculation.
        mult: multiplication factors for first term (expectation wrt real data)
              and second term (expectation wrt fake data)
    Returns:
        Negative cross entropy of inputs
    """
    if isinstance(d_real, type(None)):
        d_real = torch.ones_like(d_fake)
        mult = [-1 * mult[0], -1 * mult[1]]
    return (mult[0] * torch.log(d_real + 1e-8)) + (
        mult[1] * torch.log(1 - d_fake + 1e-8)
    )


def kldiv(
    d_fake: Tensor, d_real: Optional[Tensor] = None, mult: Optional[list] = [-1.0, -1.0]
) -> Tensor:
    """
    Cross-entropy loss for discriminator, KL divergence for generator.

    Args:
        d_fake: discriminator output for fake data.
        d_real: discriminator output for real data. If None, the corresponding
                term is dropped from the cross entropy calculation.
        mult: multiplication factors for first term (expectation wrt real data)
              and second term (expectation wrt fake data)
    Returns:
        Negative cross entropy of inputs
    """
    if isinstance(d_real, type(None)):
        return torch.log(1 - d_fake + 1e-8) - torch.log(d_fake + 1e-8)
    else:
        return cross_entropy(d_fake, d_real, mult=mult)


def wasserstein(
    d_fake: Tensor, d_real: Optional[Tensor] = None, mult: Optional[list] = [1.0, -1.0]
) -> Tensor:
    """
    Wasserstein loss.
    Args:
        d_fake: discriminator output for fake data.
        d_real: discriminator output for real data. If None, the corresponding
                term is dropped from the cross entropy calculation.
        mult: multiplication factors for first term (expectation wrt real data)
              and second term (expectation wrt fake data)
    Returns:
        Wasserstein distance of inputs
    """
    if isinstance(d_real, type(None)):
        return -mult[1] * d_fake
    else:
        return (mult[0] * d_real) + (mult[1] * d_fake)


loss_dict = {"cross_entropy": cross_entropy, "kldiv": kldiv, "wasserstein": wasserstein}
