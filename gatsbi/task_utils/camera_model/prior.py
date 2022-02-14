from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
from torchvision.transforms import CenterCrop, Compose, ToTensor

# Default path to data
default_path_to_data = Path(__file__).parent / "data"


def _merge_train_test(data_funcn, path_to_data, unsqueeze=False, **funcn_kwargs):

    train = torch.cat(
        [
            dat[0].unsqueeze(0)
            for dat in data_funcn(path_to_data, train=True, **funcn_kwargs)
        ],
        0,
    )
    test = torch.cat(
        [
            dat[0].unsqueeze(0)
            for dat in data_funcn(path_to_data, train=False, **funcn_kwargs)
        ],
        0,
    )
    return torch.cat([train, test], 0)


class EMNISTPrior:
    """EMNIST image prior."""

    def __init__(
        self, path_to_data: str = None, crop_size: int = 28, download: bool = False
    ) -> None:
        """
        Instantiate an implicit prior over the EMNIST training dataset.

        Args:
            path_to_data: path to the PyTorch EMNIST dataset.
            crop_size: side length of the image after cropping.
                       Default=28 giving the original image size 28x28.
            download: Whether to download and save the data to the specified
                      path.
        """
        # Set variables
        self.path_to_data = path_to_data
        if self.path_to_data is None:
            self.path_to_data = default_path_to_data

        self.crop_size = crop_size
        self.download = download
        # Load data
        self.transform = Compose([CenterCrop(crop_size), ToTensor()])
        self.data = _merge_train_test(
            EMNIST,
            self.path_to_data,
            split="bymerge",
            download=self.download,
            transform=self.transform,
        )
        self.num_data = len(self.data)

    def __call__(self, num_samples: int = 1) -> Tensor:
        """Return a random batch of images from the EMNIST dataset."""
        return self.sample(num_samples)

    def sample(self, num_samples: int = 1) -> Tensor:
        """Return a random batch of images from the EMNIST dataset."""
        assert self.data is not None and self.num_data is not None
        loader = DataLoader(self.data, shuffle=True, batch_size=num_samples)

        for image in loader:
            samples = torch.FloatTensor(image)
            break

        return samples
