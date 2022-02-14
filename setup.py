#!/usr/bin/env python

from setuptools import find_packages, setup

package_name = "gatsbi"
version = "1.0"
exclusions = ["notebooks", "src"]

_packages = find_packages(exclude=exclusions)

_base = [
    "numpy",
    "matplotlib",
    "pandas",
    "pyyaml",
    "scikit-learn",
    "torch",
    "plotly",
    "wandb",
    "scipy",
    "torchvision",
    "scikit-image",
    "jupyter",
]
_sbi_extras = [
    "sbibm@git+https://github.com/mackelab/sbibm#egg=sbibm",
    "sbi@git+https://github.com/mackelab/sbi#egg=sbi",
]

setup(
    name=package_name,
    version=version,
    description="Generative Adversarial Networks for Simulation-Based\
                   Inference",
    author="Poornima Ramesh",
    author_email="poornima.ramesh@tum.de",
    url="https://github.com/mackelab/gatsbi",
    packages=["gatsbi", "tests"],
    install_requires=(_base + _packages + _sbi_extras),
)
