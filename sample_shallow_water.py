"""Code to generate samples from shallow water model using multiprocessing."""

import argparse
import multiprocessing.pool as pool
import time
from os import makedirs
from os.path import join

import numpy as np

from gatsbi.task_utils.shallow_water_model import Prior, Simulator


def _seed_by_time_stamp(num_seeds):
    seeds = []
    for i in range(num_seeds):
        tic = time.time()
        seeds.append(int((tic % 1) * 1e7))
    return seeds


def _fwd_pass_prior_and_simulator(sim_num):
    seed_depth, seed_u, seed_z = _seed_by_time_stamp(3)
    depth_profile = Prior(return_seed=False)(seed=seed_depth)
    depth_profile_copy = depth_profile.copy()
    u, z = Simulator(outdir=seed_depth)(
        depth_profile, seeds_u=[seed_u], seeds_z=[seed_z]
    )

    return depth_profile_copy - 10.0, u, z, seed_depth, seed_u, seed_z, sim_num


def main(args):
    """Sample function."""
    data = []
    # Keep track of simulations
    simulation_number = np.arange(args.num_simulations, dtype=np.int)

    # simulate in parallel with mpi
    with pool.Pool(args.num_processes) as pool_procs:
        data.append(pool_procs.map(_fwd_pass_prior_and_simulator, simulation_number))
    pool_procs.close()
    pool_procs.join()

    # save data
    makedirs(args.path_to_save, exist_ok=True)
    outfile = join(args.path_to_save, "data_%d.npz" % args.job_num)
    np.savez_compressed(
        outfile,
        depth_profile=[dat[0] for dat in data[0]],
        u_vals=[dat[1] for dat in data[0]],
        z_vals=[dat[2] for dat in data[0]],
        seeds_depth=[dat[3] for dat in data[0]],
        seeds_u=[dat[4] for dat in data[0]],
        seeds_z=[dat[5] for dat in data[0]],
        sim_ind=[dat[6] for dat in data[0]],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_simulations", type=int, default=10000)
    parser.add_argument("--num_processes", type=int, default=20)
    parser.add_argument("--path_to_save", type=str, default="./shallow_water_data/")
    parser.add_argument("--job_num", type=int, default=1)
    args = parser.parse_args()
    main(args)
