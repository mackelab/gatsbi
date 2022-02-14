import os
import os.path
import pickle

import numpy as np
import torch
import torch.nn as nn
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
from torch.distributions import MultivariateNormal

from gatsbi.task_utils.shallow_water_model.networks import _make_embed_net
from gatsbi.task_utils.shallow_water_model.prior import gaussian_kernel

# Hyperparams for training different from default.
num_hidden_features = 100
training_batch_size = 1000

device = "cuda:0"

# Load data.
# TODO: change path to local data dir.
path_to_sims = "./shallow_water_data/"

files = sorted(f for f in os.listdir(path_to_sims) if f.endswith(".npz"))

# Load data sequentially from files
depth_profiles, z_vals = [], []
for file in files:
    dd = np.load(os.path.join(path_to_sims, file))
    depth_profiles.append(dd["depth_profile"].squeeze())
    z_vals.append(dd["z_vals"].squeeze())

# Reshape and turn into torch tensors
depth_profiles = torch.FloatTensor(np.concatenate(depth_profiles, 0))
z_vals = np.concatenate(z_vals, 0)[:, :, 1:]
z_vals = torch.FloatTensor(z_vals)

theta = depth_profiles
x = z_vals

# Set up prior.
cov = torch.tensor(gaussian_kernel(size=100, sigma=15, tau=100.0), dtype=torch.float32)
# cholesky cannot factor this cov because of slightly negative eigenvals
# (not positive-definite)
# (check np.random.default_rng().multivariate_normal(np.array(loc),
# np.array(cov), method='svd' or 'eigh') - that works
# fix:
covpos = cov + 1e-4 * torch.eye(*cov.shape)
loc = torch.ones(100) * 10
prior = MultivariateNormal(loc=loc, covariance_matrix=covpos)

# Set up density estimator
# Initialize the DCGAN discriminator with appropriate input shape.
# Take the 2nd embedding net as embedding net for sbi.
embedding_net = nn.Sequential(*_make_embed_net())
print("Build posterior")
density_estimator_build_fun = posterior_nn(
    model="nsf",
    hidden_features=num_hidden_features,
    embedding_net=embedding_net,
    z_score_x=True,
)

# Run inference.
print("run inference")
inference = SNPE(
    prior=prior,
    density_estimator=density_estimator_build_fun,
    device=device,
    show_progress_bars=True,
)
density_estimator = inference.append_simulations(theta, x).train(
    training_batch_size=training_batch_size
)
posterior = inference.build_posterior(density_estimator)

# Save inference.
with open("../runs/shallow_water_model/npe_results.p", "wb") as fh:
    pickle.dump(dict(posterior=posterior, de=density_estimator), fh)
