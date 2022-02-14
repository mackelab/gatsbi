import unittest

import torch

from gatsbi.utils.load_data import MakeDataset, make_loader


class TestDataLoader(unittest.TestCase):
    """Test code for optimiser dataloader."""

    x = torch.ones(10, 1)
    y = torch.zeros(10, 1)
    hold_out = 1

    def test_dataset_maker_class(self):
        """Test code for making dataset class passed to torch.DataLoader."""
        dataset = MakeDataset(inputs=[self.x, self.y], hold_out=self.hold_out)

        self.assertEqual(dataset.__len__(), 9) and hasattr(dataset, "inputs_test")

    def test_dataset_loader_wrapper(self):
        """Test code for making torch.DataLoader object."""
        batch_size = 1
        dataloader = make_loader(
            batch_size=batch_size,
            inputs_to_loader_class={"inputs": [self.x, self.y], "hold_out": 0},
        )
        out = [s for s in dataloader]
        self.assertEqual(len(out), 10) and self.assertEqual(
            len(out[0]), 2
        ) and self.assertIsInstance(out[0][0], int) and self.assertIsInstance(
            out[0][1][0], torch.tensor
        ) and self.assertIsInstance(
            out[0][1][1], torch.tensor
        )
