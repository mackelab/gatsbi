from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.neural_network import MLPClassifier as MLPClassifier


class Classifier(nn.Module):
    """Train a classifier between prior and proposal distribution."""

    def __init__(
        self,
        training_prior_samples: torch.tensor,
        training_proposal_samples: torch.tensor,
        **kwargs
    ) -> None:
        """
        Train a classifier between prior and proposal samples.

        Args:
            training_prior_samples: training data for classifier sampled from
                                    prior distribution.
            training_proposal_samples: training data for classifier sampled
                                       from proposal distribution.
            **kwargs: keyword arguments for MLPClassifier object from
                      sklearn.neural_network. See sklearn docs for more
                      information.
        """
        self.MLP = MLPClassifier(**kwargs)
        self.prior_samples = training_prior_samples.cpu().data.numpy()
        self.proposal_samples = training_proposal_samples.cpu().data.numpy()
        self._train_mlp()

    def _train_mlp(self):
        regressor = np.concatenate([self.prior_samples, self.proposal_samples], 0)
        targets = np.concatenate(
            [np.zeros(len(self.prior_samples)), np.ones(len(self.proposal_samples))]
        )
        self.MLP.fit(regressor, targets)

    def odds(
        self, samples: torch.tensor, invert: Optional[bool] = False
    ) -> torch.tensor:
        """
        Return probability density ratio of input samples.

        Args:
            samples: samples for which to compute ratio of densities.
            invert: If False. return proposal / prior. If True, return prior /
                    proposal.
        """
        samples_numpy = samples.cpu().data.numpy()
        if samples_numpy.ndim == 1:
            samples_numpy = np.expand_dims(samples_numpy, 0)
        prob_prop = self.MLP.predict_proba(samples_numpy).squeeze()
        odds = torch.tensor((prob_prop + 1e-3) / (1 - prob_prop + 1e-3))
        odds = odds.to(samples.device.type)
        if not invert:
            return odds[..., 0]
        else:
            return odds[..., 1]
