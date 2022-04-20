# GATSBI: Generative Adversarial Training for Simulation-Based Inference
---
Code package `gatsbi` implementing method and experiments described in the associated manuscript  ["GATSBI: Generative Adversarial Training for Simulation-Based Inference"](https://openreview.net/forum?id=kR1hC6j48Tp&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2022%2FConference%2FAuthors%23your-submissions))

The code depends both on the simulation-based inference package [`sbi`](https://github.com/mackelab/sbi) and the benchmark framework [`sbibm`](https://github.com/mackelab/sbibm).
___
### Installation

With a working Python environment, install `gatsbi` using `pip`:
```
pip install "git+https://github.com/mackelab/gatsbi"
```
___
### Mininmal example

For a minimal demonstration of how to use `gatsbi` see `quickstart.ipynb`.
___
### Experiments

The paper describes results for the following experiments: 2 benchmark tasks, the shallow water model, and a noisy camera model.

Code for setting up priors, simulator, GAN networks and any other pre-/post-processing code is available inside `gatsbi.task_utils`.

Hyperparameter settings for each of the experiments are available in `tasks/`

To reproduce the exact experiments described in the paper, use the following run_scripts from the repository's root directory (note that this relies on `wandb` to log experiments)
- `run_benchmarks.py` for benchmark tasks
    ```
    python run_benchmarks.py --project_name="Benchmarks" --task_name="two_moons"
    ```
    `task_name` = `slcp` or `two_moons` for amortized GATSBI, `slcp_seq` or `two_moons_seq` for sequential GATSBI.
- `run_highdim_applications.py` for high dimensional tasks
    ```
    python run_highdim_applications.py --project_name="High Dimensional Applications" --task_name="shallow_water_model"
    ```
    `task_name` = `shallow_water_model` or `camera_model`.
- `run_inference_nle/npe/nre.py` for running NPE / NLE / NRE on shallow water model.
    ```
    python run_inference_nle/npe/nre.py
    ```
Note that we **do not** provide training data for the shallow water model in this repository. Please use `sample_shallow_water.py` to generate training samples locally.
___
### Figures

Code to reproduce the figures in the paper is available in `plotting_code`, along with the required data `plotting_code/plotting_data`, and the final plots `plotting_code/plots`. Note that accessing the data requires Git LFS installation.
