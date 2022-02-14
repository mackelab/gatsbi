import argparse
import importlib
from argparse import Namespace as NSp
from os import makedirs
from os.path import join

import torch
import wandb
import yaml
from torch import nn

from gatsbi.optimize import Base as Opt
from gatsbi.optimize import UnrolledOpt as UOpt
from gatsbi.task_utils.run_utils import _update_defaults


def main(args):
    args, unknown_args = args
    print(args, unknown_args)

    # Get defaults
    with open(join("tasks", args.task_name, "defaults.yaml"), "r") as f:
        defaults = yaml.load(f, Loader=yaml.Loader)

    # Update defaults
    if len(unknown_args) > 0:
        defaults = _update_defaults(defaults, unknown_args)

    # Get application module
    application = importlib.import_module("gatsbi.task_utils.%s" % args.task_name)

    # Make a logger
    print("Making logger")
    makedirs(join("..", "runs", args.task_name), exist_ok=True)
    wandb.init(
        project=args.project_name,
        group=args.group_name,
        id=args.run_id,
        resume=args.resume,
        config=defaults,
        notes="",
        dir=join("..", "runs", args.task_name),
    )
    config = NSp(**wandb.config)

    run = wandb.run
    with run:
        print("Making networks")
        # Make generator and discriminator

        gen = application.Generator()
        dis = application.Discriminator()

        # Make networks work across multiple GPUs
        if args.multi_gpu:
            gen = nn.DataParallel(gen)
            dis = nn.DataParallel(dis)

        if args.resume:
            assert args.resume_dir is not None
            chpt = torch.load(join(args.resume_dir, "checkpoint_models0.pt"))
            gen.load_state_dict(chpt["generator_state_dict"])
            dis.load_state_dict(chpt["dis_state_dict"])

        gen.cuda()
        dis.cuda()

        # Make optimiser
        print("Making optimiser")
        batch_size = min(1000, int(config.batch_size_perc * config.num_simulations))
        prior = application.Prior()
        simulator = application.Simulator()
        dataloader = {}
        if hasattr(application, "get_dataloader"):
            dataloader = application.get_dataloader(
                batch_size, config.hold_out, config.path_to_data
            )

        # Make optimizer
        if args.task_name == "shallow_water_model":
            opt = UOpt(
                generator=gen,
                discriminator=dis,
                prior=prior,
                simulator=simulator,
                optim_args=[config.gen_opt_args, config.dis_opt_args],
                dataloader=dataloader,
                loss=config.loss,
                round_number=0,
                training_opts={
                    "gen_iter": config.gen_iter,
                    "dis_iter": config.dis_iter,
                    "max_norm_gen": config.max_norm_gen,
                    "max_norm_dis": config.max_norm_dis,
                    "num_simulations": config.num_simulations,
                    "sample_seed": 42,
                    "hold_out": config.hold_out,
                    "batch_size": batch_size,
                    "unroll_steps": config.unroll_steps,
                    "log_dataloader": config.log_dataloader,
                },
                logger=run,
            )
        else:
            opt = Opt(
                generator=gen,
                discriminator=dis,
                prior=prior,
                simulator=simulator,
                optim_args=[config.gen_opt_args, config.dis_opt_args],
                dataloader=dataloader,
                loss=config.loss,
                round_number=0,
                training_opts={
                    "gen_iter": config.gen_iter,
                    "dis_iter": config.dis_iter,
                    "max_norm_gen": config.max_norm_gen,
                    "max_norm_dis": config.max_norm_dis,
                    "num_simulations": config.num_simulations,
                    "sample_seed": 42,
                    "hold_out": config.hold_out,
                    "batch_size": batch_size,
                    "log_dataloader": config.log_dataloader,
                },
                logger=run,
            )

        if args.resume:
            opt.epoch_ct = chpt["epoch"]

        # Train model
        print("Training")
        opt.train(args.epochs, 100)

    wandb.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--group_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--multi_gpu", type=bool, default=False)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--resume_dir", type=str, default=None)
    main(parser.parse_known_args())
