from typing import Callable, Dict, Iterable

import numpy as np
from torch.utils import data
from torch.utils.data import dataloader


class MakeDataset(data.Dataset):
    """Data-loading class for GANs."""

    def __init__(self, inputs, hold_out=0):
        """
        Set up data-loading class for GANs.

        Args:
            inputs (list): list of inputs to GAN networks.
            hold_out (int): number of samples to hold out as test dataset from
                            `inputs`.
        """
        self.inputs = inputs
        self.hold_out = hold_out

        # If hold_out length is nonzero, make a test dataset
        if self.hold_out > 0.0:
            self.inputs_test = [inp[-self.hold_out :] for inp in self.inputs]

    def __len__(self):
        """Return size of dataset."""
        # Check length of al inputs is same (need this to batch properly)
        assert np.all([len(inp) == len(self.inputs[0]) for inp in self.inputs[1:]])
        return len(self.inputs[0]) - self.hold_out

    def __getitem__(self, idx):
        """Get data."""
        batch_inputs = list()
        for inp in self.inputs:
            batch_inputs.append(inp[idx])
        return idx, batch_inputs


def make_loader(
    batch_size: int, inputs_to_loader_class: Dict, loader_class: Callable = MakeDataset
) -> Iterable[data.DataLoader]:
    """
    Make iterable data-loader.

    Args:
        batch_size: batch size for data-loader.
        inputs_to_loader_class: inputs to loader class.
        loader_class: data.Dataset class for loading data. Default is
                      MakeDataset.

    Returns:
        Iterable data-loader

    """
    dataset = loader_class(**inputs_to_loader_class)
    return dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=True)
