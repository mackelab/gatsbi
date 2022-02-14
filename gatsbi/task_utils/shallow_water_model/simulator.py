import shutil
import subprocess
import time
from importlib import import_module
from os import makedirs
from os.path import join
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.fft import fft2

path_to_fcode = str(Path(__file__).parent / "shallow_water01_modified.f90")


def observation_noise(
    observation: np.ndarray,
    seed: Optional[int] = 42,
    gain: Optional[float] = 1,
    scale: Optional[float] = 0.25,
) -> np.ndarray:
    """
    Add white noise to observations.

    Args:
        observation: simulation to which to add noise.
        seed: random-sampling seed.
        gain: gain value to scale up observation values.
        scale: std of white noise.
    """
    np.random.seed(seed)
    return gain * (observation) + (scale * np.random.randn(*observation.shape))


class ShallowWaterSimulator:
    """Simulator class to generate surface waves given a depth profile."""

    def __init__(
        self,
        path_to_fcode: Optional[str] = path_to_fcode,
        outdir: Optional[int] = 0,
        fourier: Optional[bool] = True,
        return_seed: Optional[bool] = False,
        **observation_noise_kwargs
    ):
        """
        Set up simulator.

        Args:
            path_to_fcode: path to fortran code
            outdir: name of directory in which to temporarily store
                    generated data. THE NAME SHOULD BE AN INTEGER EXACTLY 7
                    CHARACTERS LONG #TODO: Fix this!
            fourier: if True, returns 2D fourirer transform of simulator
                     outputs.
            return_seed: if True, returns random-sampling seeds for
                        observation noise.
            **observation_noise_kwargs: all keyword argurments for
                                        observation_noise() function,
                                        except "seed".
        """
        self.path_to_fcode = path_to_fcode
        self.outdir = outdir
        self.fourier = fourier
        self.return_seed = return_seed
        self.observation_noise_kwargs = observation_noise_kwargs

        # import shallow_water module from fortran code
        try:
            self.sw = import_module("shallow_water")
        except ModuleNotFoundError:
            bashcommand = "python -m numpy.f2py -c %s -m shallow_water" % path_to_fcode
            subprocess.call(bashcommand.split(" "))
            self.sw = import_module("shallow_water")

    def __call__(
        self,
        depth_profiles: np.ndarray,
        seeds_u: Optional[list] = None,
        seeds_z: Optional[list] = None,
    ):
        """Call to simulator."""
        return self.sample(depth_profiles, seeds_u, seeds_z)

    def sample(
        self,
        depth_profiles: np.ndarray,
        seeds_u: Optional[list] = None,
        seeds_z: Optional[list] = None,
    ) -> Tuple[np.ndarray]:
        """
        Forward pass through simulator.

        Args:
            depth_profiles: array of depth profiles for which to
                            make surface wave simulations.
            seeds_u: seeds for surface wave height observation noise.
            seeds_z: seeds for surface wave speed observation noise.

        Returns:
            tuple containing surface wave velocity profile and height profile
            as a function of time (i.e timebins x length).
        """
        # Make directory for output of fortran code
        if self.outdir == 0:
            self.outdir = int((time.time() % 1) * 1e7)
        makedirs("%07d" % self.outdir, exist_ok=True)
        file_z = join("%07d" % self.outdir, "z%s.dat")
        file_u = join("%07d" % self.outdir, "u%s.dat")

        u_vals = []
        z_vals = []

        # Make random-sampling seeds
        if seeds_u is None:
            seeds_u = 1000 + np.arange(len(depth_profiles), dtype=np.int)
        if seeds_z is None:
            seeds_z = 2000 + np.arange(len(depth_profiles), dtype=np.int)

        for depth_profile, seed_u, seed_z in zip(depth_profiles, seeds_u, seeds_z):
            # Fwd pass through simulator
            self.sw.shallow_water(depth_profile, int(self.outdir))

            # read z output into single array
            z = np.zeros((101, 100))
            for i in range(0, 101):
                str_i = ("{0:03d}").format(i)
                with open(file_z % (str_i), "r") as f:
                    z[i] = np.loadtxt(f)

            # read u output into a single array
            u = np.zeros((101, 100))
            for i in range(0, 101):
                str_i = ("{0:03d}").format(i)
                with open(file_u % (str_i), "r") as f:
                    u[i] = np.loadtxt(f)

            # fourier transform
            if self.fourier:
                fft_u_real = np.expand_dims(fft2(u).real, 0)
                fft_z_real = np.expand_dims(fft2(z).real, 0)
                fft_u_imag = np.expand_dims(fft2(u).imag, 0)
                fft_z_imag = np.expand_dims(fft2(z).imag, 0)
                u = np.concatenate([fft_u_real, fft_u_imag], 0)
                z = np.concatenate([fft_z_real, fft_z_imag], 0)

            # add observation noise
            u = observation_noise(u, seed_u, **self.observation_noise_kwargs)
            z = observation_noise(z, seed_z, **self.observation_noise_kwargs)

            u_vals.append(u)
            z_vals.append(z)

        # Remove save directory to free memory
        shutil.rmtree("%07d" % self.outdir)

        if self.return_seed:
            return np.array(u_vals), np.array(z_vals), seeds_u, seeds_z
        else:
            return np.array(u_vals), np.array(z_vals)
