from os.path import join
from time import time

import numpy as np
import torch


def _sample(prior, simulator, sample_seed, num_samples):
    """Return samples from prior and simulator."""
    if type(sample_seed) == int:
        torch.manual_seed(sample_seed)
    prior_samples = prior(num_samples)
    observations = simulator(prior_samples)
    return prior_samples, observations


def _check_data_bank(round_number, dataloader_dict):
    if "%d" % round_number in dataloader_dict.keys():
        return True
    return False


def _make_checkpoint(opt, init):
    """
    Save network and optimiser state dictionaries as checkpoint.

    Args:
        opt: Optimize object
        init: Set to True if checkpoint is to be made before training
    """
    checkpoint = {
        "epoch": opt.epoch_ct,
        "generator_state_dict": opt.generator.state_dict(),
        "gen_optimizer_state_dict": opt.generator_optim.state_dict(),
        "dis_state_dict": opt.discriminator.state_dict(),
        "dis_opt_state_dict": opt.discriminator_optim.state_dict(),
    }
    init_str = str(opt.round_number)
    if init:
        init_str = "_init"
    torch.save(checkpoint, join(opt.logger.dir, "checkpoint_models%s.pt" % init_str))

    if hasattr(opt, "classifier_theta"):
        torch.save(
            opt.classifier_theta,
            join(opt.logger.dir, "classifier_theta%s.pt" % opt.round_number),
        )
    if hasattr(opt, "classifier_obs"):
        torch.save(
            opt.classifier_obs,
            join(opt.logger.dir, "classifier_obs%s.pt" % opt.round_number),
        )
    if opt.training_opts.log_dataloader:
        torch.save(
            opt.dataloader,
            join(opt.logger.dir, "checkpoint_dataloader%s.pt" % init_str),
        )


def _log_metrics(opt):
    """
    Log metrics and figures.

    Args:
        opt: Optimize object.
    """
    # Make data
    dataloader = opt.dataloader[str(opt.round_number)]
    theta_test, obs_test = dataloader.dataset.inputs_test
    theta_test = theta_test.to(opt.device)
    obs_test = obs_test.to(opt.device)

    opt.generator.eval()
    opt.discriminator.eval()
    theta_fake_cv = opt._fwd_pass_generator(obs_test)
    theta_fake_cv_detach = theta_fake_cv.clone().detach()

    dis_fake_cv = opt.discriminator([theta_fake_cv, obs_test])
    dis_fake_cv_detach = opt.discriminator([theta_fake_cv_detach, obs_test])
    dis_real_cv = opt.discriminator([theta_test, obs_test])

    loss_dis = opt.loss(dis_fake_cv_detach, dis_real_cv).mean()
    loss_gen = opt.loss(dis_fake_cv).mean()

    loss_dis.backward(retain_graph=True)
    loss_gen.backward(retain_graph=True)

    gen_grads = torch.sqrt(
        sum([torch.norm(p.grad) ** 2 for p in opt.generator.parameters()])
    )
    dis_grads = torch.sqrt(
        sum([torch.norm(p.grad) ** 2 for p in opt.discriminator.parameters()])
    )
    if opt.logger is not None:
        step = opt.logger.step
    else:
        step = len(opt.df)
    opt.df.loc[step] = {
        "dis_loss": loss_dis.mean().item(),
        "gen_loss": loss_gen.mean().item(),
        "dreal_mean": dis_real_cv.mean().item(),
        "dfake_mean": dis_fake_cv.mean().item(),
        "dreal_std": dis_real_cv.std().item(),
        "dfake_std": dis_fake_cv.std().item(),
        "dis_grad": dis_grads.item(),
        "gen_grad": gen_grads.item(),
        "global_step": opt.epoch_ct,
    }
    torch.cuda.empty_cache()
    opt.generator.train()
    opt.discriminator.train()

    # Update logger
    if opt.logger is not None:
        opt.logger.history.add(dict(opt.df.loc[opt.logger.step]))


def _stop_training(opt):
    """
    Check criteria to stop training.

    Args:
        opt: Optimize object

    Returns: boolean value. If True, stop training networks.

    """
    if not hasattr(opt.training_opts, "stop_thresh"):
        stop_thresh = 0.001
    else:
        stop_thresh = opt.training_opts.stop_thresh
    lr = opt.generator_optim.defaults["lr"]
    grad = np.array(opt.df.loc[:, ["gen_grad"]][-20:])
    gen_loss = np.array(opt.df.loc[:, ["gen_loss"]][-10:])

    dreal_mean = np.array(opt.df.loc[:, ["dreal_mean"]][-10:])
    dfake_mean = np.array(opt.df.loc[:, ["dfake_mean"]][-10:])

    dfake_std = np.array(opt.df.loc[:, ["dfake_std"]][-10:])
    dreal_std = np.array(opt.df.loc[:, ["dreal_std"]][-10:])

    # If nans or infs in generator params
    for p in opt.generator.parameters():
        if not torch.all(torch.isfinite(p)):
            return True

    # If nans or infs in discriminator params
    for p in opt.discriminator.parameters():
        if not torch.all(torch.isfinite(p)):
            return True

    # If discriminator is overconfident  -- i.e variance of discrim.
    # outputs does not change
    if (
        (len(dfake_std) >= 10)
        and (dfake_std.mean() < 1e-6)
        and (dreal_std.mean() < 1e-6)
    ):
        print("Discriminator is over-confident")
        return True

    # If generator loss collapses to 0.
    elif abs(gen_loss).mean() < 1e-6:
        print("Generator loss is 0.")
        return True

    # If generator gradients are zero
    elif abs(grad.mean()) < 1e-4 * lr:
        print("Generator gradient is 0.")
        return True

    # If discriminator output is same for real and fake data
    elif (len(dreal_mean) >= 10) and (
        abs((dreal_mean - dfake_mean).mean()) < stop_thresh
    ):
        print("Discriminator output .5")
        return True

    # If training runs for more than 3 days
    elif (time() - opt.start) > (72 * 3600):
        if opt.round_number < 1:
            return True
    return False
