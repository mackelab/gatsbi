import unittest

import torch

from gatsbi.utils.classifier import Classifier


class TestClassifier(unittest.TestCase):
    """Test classifier code."""

    classifier = Classifier(
        torch.randn(10, 5) + 10,
        torch.randn(10, 5) + 4,
        hidden_layer_sizes=[5, 5],
        max_iter=1000,
    )

    def test_classifier_setup(self):
        """Test training code for classifier."""
        predict = self.classifier.MLP.predict_proba(torch.randn(1, 5) + 4).sum()
        self.assertEqual(predict, 1)

    def test_classifier_odds_calculation(self):
        """Test if calculation of odds for input to classifier is sensible."""
        samp = torch.randn(1, 5) + 4
        proba = self.classifier.MLP.predict_proba(samp).squeeze()
        probs = proba / (1 - proba)
        odds = self.classifier.odds(samp, invert=False).data.numpy()
        inv_odds = self.classifier.odds(samp, invert=True).data.numpy()

        self.assertAlmostEqual(odds, probs[0], 2) and self.assertAlmostEqual(
            inv_odds, probs[1], 2
        )
