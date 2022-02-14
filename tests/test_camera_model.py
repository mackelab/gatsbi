import pytest
import torch

from gatsbi.utils.camera_model import (CIFAR10Prior, EMNISTPrior, MNISTPrior,
                                       grayscale_camera, rgb_camera)


@pytest.mark.slow
@pytest.mark.parametrize("prior_type", ("mnist", "emnist"))
def test_mnist_prior(prior_type):
    """Test forward pass through camera model prior."""
    # Assuming the tests are called from root of the repository.
    crop_size = 10
    path_to_data = "./gatsbi/utils/camera_model/data/%s.data" % prior_type
    if prior_type == "mnist":
        prior = MNISTPrior(path_to_data, download=False, crop_size=crop_size)
    else:
        prior = EMNISTPrior(
            path_to_data=path_to_data, download=False, crop_size=crop_size
        )

    num_samples = 5
    torch.manual_seed(num_samples)
    theta1 = prior.sample(num_samples)
    torch.manual_seed(num_samples)
    theta2 = prior(num_samples)

    assert (theta1 == theta2).all()

    assert theta1.shape == torch.Size([num_samples, crop_size, crop_size])


@pytest.mark.slow
@pytest.mark.parametrize("prior_type", ("mnist", "emnist"))
def test_camera(prior_type):
    """Test forward pass through gray scale simulator for camera model."""
    # Assuming the tests are called from root of the repository.
    path_to_data = "./gatsbi/utils/camera_model/data/%s.data" % prior_type
    crop_size = 10
    if prior_type == "mnist":
        prior = MNISTPrior(path_to_data, download=False, crop_size=crop_size)
    if prior_type == "emnist":
        prior = EMNISTPrior(
            path_to_data=path_to_data, download=False, crop_size=crop_size
        )

    def simulator(theta):
        return grayscale_camera(theta, noise="poisson", psf_width=1.0)

    num_samples = 5
    theta = prior(num_samples)
    x = torch.stack([simulator(th.squeeze(0)) for th in theta])

    assert x.shape == torch.Size([num_samples, crop_size, crop_size])


@pytest.mark.slow
def test_cifar10_prior():
    """Test forward pass through camera model prior."""
    # Assuming the tests are called from root of the repository.
    path_to_data = "./gatsbi/utils/camera_model/data/cifar10.data"
    crop_size = 31
    prior = CIFAR10Prior(
        path_to_data, download=False, crop_size=crop_size, use_grayscale=False
    )

    num_samples = 5
    torch.manual_seed(num_samples)
    theta1 = prior.sample(num_samples)
    torch.manual_seed(num_samples)
    theta2 = prior(num_samples)

    assert (theta1 == theta2).all()

    assert theta1.shape == torch.Size([num_samples, 3, crop_size, crop_size])


@pytest.mark.slow
def test_rgb_camera():
    """Test forward pass through rgb simulator for camera model."""
    # Assuming the tests are called from root of the repository.
    path_to_data = "./gatsbi/utils/camera_model/data/cifar10.data"
    crop_size = 31
    prior = CIFAR10Prior(
        path_to_data, download=False, crop_size=crop_size, use_grayscale=False
    )

    def simulator(theta):
        return rgb_camera(theta, noise="poisson", psf_width=1.0)

    num_samples = 5
    theta = prior(num_samples)
    x = torch.stack([simulator(th.squeeze(0)) for th in theta])

    assert x.shape == torch.Size([num_samples, 3, crop_size, crop_size])
