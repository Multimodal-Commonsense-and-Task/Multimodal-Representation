# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Modified for AC-Sampler (ICLR 2026): Metropolis-Hastings refinement of diffusion sampling
# with a time-dependent discriminator. Built on the EDM sampler
# (Karras et al., 2022) and the DG code base (Kim et al., 2023).

"""Generate CIFAR-10 samples with AC-Sampler.

The sampler follows the EDM Heun trajectory. At user-chosen intermediate
timesteps ("branch times") it runs Metropolis-adjusted Langevin proposals
on the noisy image x_t. The acceptance ratio combines the score network
(proposal kernel) and the time-dependent discriminator (density ratio
between real and generated marginals). Accepted states are pushed into a
queue and later denoised to t=0 as additional trajectories, so one root
trajectory can produce many samples while sharing the early steps.

Running the script without --branch_time reproduces the plain EDM sampler.
"""

import os
import io
import json
import time
import random
import pickle

import click
import numpy as np
import torch
import PIL.Image
from torchvision.utils import make_grid, save_image
from sortedcontainers import SortedList

import classifier_lib

#----------------------------------------------------------------------------
# AC-Sampler.

def ac_sampler(
    net, discriminator, vpsde, latents, class_labels=None, class_idx=None,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    time_min=0.01, time_max=1.0, dg_weight=0., dg_proposal=0.,
    snr=0.16, burn_in=10, num_skip=0, do_mh=True, pair_dict=None,
    batch_size=100, num_samples=50000, outdir=None, save_type='npz', do_seed=1,
    randn_like=torch.randn_like,
):
    pair_dict = dict(pair_dict) if pair_dict else {}
    device = latents.device
    nfe = {'total_nfe': 0, 'total_samples': 0}

    S_churn_vec = torch.full((latents.shape[0],), float(S_churn), device=device)
    S_churn_max = torch.full((latents.shape[0],), np.sqrt(2) - 1, device=device)
    S_noise_vec = torch.full((latents.shape[0],), float(S_noise), device=device)

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    def log_ratio_fn(x, t, labels):
        return classifier_lib.get_grad_log_ratio(
            discriminator, vpsde, x, t, net.img_resolution, time_min, time_max, labels, log_only=True).detach().cpu()

    def dg_fn(x, t, labels):
        dg, _ = classifier_lib.get_grad_log_ratio(
            discriminator, vpsde, x, t, net.img_resolution, time_min, time_max, labels, log=True)
        return dg

    def denoise(x, t, labels):
        nfe['total_nfe'] += x.shape[0]
        return net(x, t, labels).to(torch.float64)

    def mh_branch(x_branch, lst_idx_branch, labels_branch, num_samples_branch):
        """Run MH-Langevin chains at a fixed noise level and collect accepted states."""
        t_branch = t_steps[lst_idx_branch]
        num_branch = x_branch.shape[0]
        accepted = [[] for _ in range(num_branch)]
        accepted_d = [[] for _ in range(num_branch)]
        cnt = torch.zeros(num_branch, dtype=torch.long, device=device)
        burn_in_left = torch.full((num_branch,), burn_in, dtype=torch.long, device=device)
        tot_accept = torch.zeros(num_branch, dtype=torch.long, device=device)
        mask_continue = cnt < num_samples_branch

        # Initial state of each chain.
        log_ratio = log_ratio_fn(x_branch, t_branch, labels_branch)
        denoised = denoise(x_branch, t_branch, labels_branch)
        d = (x_branch - denoised) / t_branch[:, None, None, None]
        score = -d / t_branch[:, None, None, None]
        if dg_proposal != 0:
            score -= dg_proposal * dg_fn(x_branch, t_branch, labels_branch) / (t_branch[:, None, None, None] ** 2)

        while mask_continue.any():
            active_idx = torch.where(mask_continue)[0]
            x_active = x_branch[active_idx]
            score_active = score[active_idx]
            t_active = t_branch[active_idx]
            labels_active = labels_branch[active_idx] if labels_branch is not None else None
            log_ratio_active = log_ratio[active_idx]

            # Langevin proposal with SNR-based step size.
            noise = torch.randn_like(x_active)
            score_norm = (score_active ** 2).sum(dim=(1, 2, 3)).sqrt()
            noise_norm = (noise ** 2).sum(dim=(1, 2, 3)).sqrt()
            epsilon = (snr * (2 * noise_norm / score_norm))[:, None, None, None]
            x_prop = x_active + (epsilon ** 2 / 2) * score_active + epsilon * noise

            # Score, discriminator log-ratio at the proposal.
            log_ratio_prop = log_ratio_fn(x_prop, t_active, labels_active)
            denoised_prop = denoise(x_prop, t_active, labels_active)
            d_prop = (x_prop - denoised_prop) / t_active[:, None, None, None]
            score_prop = -d_prop / t_active[:, None, None, None]
            if dg_proposal != 0:
                score_prop -= dg_proposal * dg_fn(x_prop, t_active, labels_active) / (t_active[:, None, None, None] ** 2)
            score_prop_norm = (score_prop ** 2).sum(dim=(1, 2, 3)).sqrt()

            # MH acceptance: proposal-kernel ratio + discriminator ratio.
            accept_logprob = (epsilon.squeeze() ** 2 / 8 * (score_norm ** 2 - score_prop_norm ** 2)).detach().cpu()
            accept_logprob += (log_ratio_prop - log_ratio_active)
            accept = accept_logprob > torch.log(torch.rand_like(accept_logprob) + 1e-7)
            tot_accept[active_idx[accept]] += 1
            if not do_mh:  # plain Langevin: accept every proposal
                accept = torch.ones_like(accept)

            for j, i in enumerate(active_idx):
                if not accept[j]:
                    continue
                if burn_in_left[i] > 0:
                    burn_in_left[i] -= 1
                elif tot_accept[i] % (num_skip + 1) == 0:
                    accepted[i].append(x_prop[j:j + 1].clone().detach())
                    accepted_d[i].append(d_prop[j:j + 1].clone().detach())
                    cnt[i] += 1
                    if cnt[i] >= num_samples_branch:
                        mask_continue[i] = False
                x_branch[i:i + 1] = x_prop[j:j + 1]
                score[i:i + 1] = score_prop[j:j + 1]
                log_ratio[i:i + 1] = log_ratio_prop[j:j + 1]

        # Flatten chains -> (num_branch * num_samples_branch, ...), chain-major order.
        xs, ds, labels_out, idx_out = [], [], [], []
        for i in range(num_branch):
            for k in range(len(accepted[i])):
                xs.append(accepted[i][k])
                ds.append(accepted_d[i][k])
                idx_out.append(lst_idx_branch[i:i + 1])
                if labels_branch is not None:
                    labels_out.append(labels_branch[i:i + 1])
        xs = torch.cat(xs, dim=0)
        ds = torch.cat(ds, dim=0)
        idx_out = torch.cat(idx_out, dim=0)
        labels_out = torch.cat(labels_out, dim=0) if labels_branch is not None else None
        return xs, ds, idx_out, labels_out

    def sampling_loop(x_next, lst_idx, labels, d_cur_next, queue, remain_samples):
        t_cur = t_steps[lst_idx]
        t_next = t_steps[lst_idx + 1]
        x_cur = x_next

        # Optional stochastic churn (EDM). Disabled for CIFAR-10 deterministic sampling.
        mask = (t_cur >= S_min) & (t_cur <= S_max)
        gamma_vec = torch.where(mask, torch.minimum(S_churn_vec / num_steps, S_churn_max), torch.zeros_like(S_churn_vec))
        t_hat = net.round_sigma(t_cur + gamma_vec * t_cur)
        delta_sigma = (t_hat ** 2 - t_cur ** 2).clamp(min=0)
        x_hat = x_cur + delta_sigma.sqrt()[:, None, None, None] * S_noise_vec[:, None, None, None] * randn_like(x_cur)

        # Euler step (re-use the derivative carried over from the MH branch when available).
        if d_cur_next is not None:
            d_cur = d_cur_next
        else:
            denoised = denoise(x_hat, t_hat, labels)
            d_cur = (x_hat - denoised) / t_hat[:, None, None, None]
            if dg_weight != 0.:
                d_cur += dg_weight * dg_fn(x_hat, t_hat, labels) / t_hat[:, None, None, None]
        x_next = x_hat + (t_next - t_hat)[:, None, None, None] * d_cur

        # 2nd order (Heun) correction.
        bool_2nd = lst_idx < num_steps - 1
        if bool_2nd.sum() != 0:
            labels_ = labels[bool_2nd] if labels is not None else None
            denoised = denoise(x_next[bool_2nd], t_next[bool_2nd], labels_)
            d_prime = (x_next[bool_2nd] - denoised) / t_next[bool_2nd][:, None, None, None]
            x_next[bool_2nd] = x_hat[bool_2nd] + (t_next - t_hat)[bool_2nd][:, None, None, None] * (0.5 * d_cur[bool_2nd] + 0.5 * d_prime)

        lst_idx = lst_idx + 1
        d_cur_next = None

        # MH branching at the current timestep.
        cur_time = lst_idx[0].item()
        if cur_time in pair_dict:
            mask_branch = lst_idx == cur_time
            if mask_branch.any():
                num_samples_branch = pair_dict[cur_time]
                # Do not produce more branches than needed to reach num_samples.
                if (len(queue) + num_samples_branch) > remain_samples / batch_size:
                    num_samples_branch = remain_samples / batch_size - len(queue)

                if num_samples_branch > 0:
                    x_branch = x_next[mask_branch].clone().detach()
                    labels_branch = labels[mask_branch].clone().detach() if labels is not None else None
                    lst_idx_branch = lst_idx[mask_branch].clone().detach()
                    xs, ds, idxs, lbls = mh_branch(x_branch, lst_idx_branch, labels_branch, num_samples_branch)

                    # The first accepted state of each chain replaces the current batch...
                    num_branch = int(mask_branch.sum())
                    x_next[mask_branch] = xs[:num_branch].clone().detach()
                    d_cur_next = ds[:num_branch].clone().detach()
                    labels = lbls[:num_branch].clone().detach() if lbls is not None else None
                    lst_idx[mask_branch] = idxs[:num_branch].clone().detach()

                    # ...and the rest are queued as future batches.
                    rest_x, rest_d, rest_idx = xs[num_branch:], ds[num_branch:], idxs[num_branch:]
                    rest_labels = lbls[num_branch:] if lbls is not None else None
                    assert len(rest_x) % batch_size == 0
                    for i in range(0, len(rest_x), batch_size):
                        queue.add((rest_x[i:i + batch_size], rest_d[i:i + batch_size], rest_idx[i:i + batch_size],
                                   rest_labels[i:i + batch_size] if rest_labels is not None else None))

        return x_next, lst_idx, labels, d_cur_next, queue

    def save_img(images, index, labels=None):
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        if save_type == 'png':
            for count, image_np in enumerate(images_np):
                PIL.Image.fromarray(image_np, 'RGB').save(os.path.join(outdir, f'{index * batch_size + count:06d}.png'))
            return
        file_name_npz = os.path.join(outdir, f'samples_{index}.npz')
        file_name_png = os.path.join(outdir, f'sample_{index}.png')
        if labels is None:
            np.savez_compressed(file_name_npz, samples=images_np)
        else:
            np.savez_compressed(file_name_npz, samples=images_np, label=labels.cpu().numpy())
        nrow = int(np.sqrt(images_np.shape[0]))
        image_grid = make_grid(torch.tensor(images_np).permute(0, 3, 1, 2) / 255., nrow, padding=2)
        save_image(image_grid, file_name_png)

    def new_labels():
        if not net.label_dim:
            return None
        labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            labels[:, :] = 0
            labels[:, class_idx] = 1
        return labels

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    lst_idx = torch.zeros((latents.shape[0],), device=device).long()
    x_fin = torch.zeros_like(x_next)
    d_cur_x_next = None
    total_samples = 0
    remain_samples = num_samples
    index = 0
    queue = SortedList(key=lambda item: item[2][0].item())  # process branches with the largest noise first
    start_time = time.time()

    while total_samples <= num_samples:
        x_next, lst_idx, class_labels, d_cur_x_next, queue = sampling_loop(
            x_next, lst_idx, class_labels, d_cur_x_next, queue, remain_samples)
        bool_fin = lst_idx == num_steps
        if bool_fin.sum() == 0:
            continue

        # Collect finished images into x_fin and flush every batch_size images.
        n_fin = int(bool_fin.sum())
        offset = total_samples % batch_size
        if (batch_size - offset) <= n_fin:
            x_fin[offset:] = x_next[bool_fin][:batch_size - offset]
            save_img(x_fin, index=index if do_seed else np.random.randint(10000000), labels=class_labels)
            index += 1
            x_fin = torch.zeros_like(x_next)
            x_fin[:n_fin - batch_size + offset] = x_next[bool_fin][batch_size - offset:]
        else:
            x_fin[offset:offset + n_fin] = x_next[bool_fin]
        total_samples += n_fin
        remain_samples -= n_fin

        # Continue with a queued branch if any, otherwise start a fresh trajectory.
        if len(queue) > 0:
            pop = queue.pop(0)
            x_next = pop[0].clone().detach()
            d_cur_x_next = pop[1].clone().detach()
            lst_idx = pop[2].clone().detach()
            class_labels = pop[3].clone().detach() if pop[3] is not None else None
        else:
            x_next[bool_fin] = torch.randn_like(x_next[bool_fin]).to(torch.float64) * t_steps[0]
            lst_idx[bool_fin] = 0
            d_cur_x_next = None
            class_labels = new_labels()

        # Enough branches queued: stop branching.
        if batch_size * len(queue) >= remain_samples:
            for k in pair_dict:
                pair_dict[k] = 0

        nfe['total_samples'] = int(total_samples)
        nfe['nfe_per_sample'] = nfe['total_nfe'] / max(total_samples, 1)
        with open(os.path.join(outdir, 'nfe_analysis.json'), 'w') as f:
            json.dump(nfe, f, indent=2)
        if total_samples >= num_samples:
            break

    print(f'Done. {total_samples} samples, {nfe["total_nfe"]} NFE '
          f'({nfe["nfe_per_sample"]:.2f} NFE/sample), {time.time() - start_time:.1f} s')

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH',                        type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                 type=str, required=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                  type=click.IntRange(min=0), default=None)
@click.option('--batch', 'batch_size',     help='Batch size', metavar='INT',                                      type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--num_samples',             help='Number of samples to generate', metavar='INT',                   type=click.IntRange(min=1), default=50000, show_default=True)
@click.option('--save_type',               help='Output format', metavar='png|npz',                               type=click.Choice(['png', 'npz']), default='npz', show_default=True)
@click.option('--do_seed',                 help='Fix the random seed', metavar='INT',                             type=click.IntRange(min=0, max=1), default=1, show_default=True)
@click.option('--seed',                    help='Random seed (used when --do_seed=1)', metavar='INT',             type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--device',                  help='Device', metavar='STR',                                          type=str, default='cuda:0', show_default=True)

## EDM sampler
@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                        type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',         type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',        type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                        type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                        type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                        type=float, default=1, show_default=True)

## Discriminator
@click.option('--pretrained_classifier_ckpt', help='Path of ADM classifier (feature extractor)', metavar='PATH',  type=str, default='checkpoints/ADM_classifier/32x32_classifier.pt', show_default=True)
@click.option('--discriminator_ckpt',      help='Path of discriminator', metavar='PATH',                          type=str, default='checkpoints/discriminator/cifar_uncond/discriminator_60.pt', show_default=True)
@click.option('--cond',                    help='Conditional discriminator', metavar='INT',                       type=click.IntRange(min=0, max=1), default=0, show_default=True)
@click.option('--time_min',                help='Minimum time [0,1] to apply the discriminator', metavar='FLOAT', type=click.FloatRange(min=0., max=1.), default=0.01, show_default=True)
@click.option('--time_max',                help='Maximum time [0,1] to apply the discriminator', metavar='FLOAT', type=click.FloatRange(min=0., max=1.), default=1.0, show_default=True)
@click.option('--dg_weight', 'dg_weight',  help='Discriminator guidance weight on the ODE drift (0 = off)', metavar='FLOAT', type=float, default=0., show_default=True)

## AC-Sampler
@click.option('--branch_time',             help='Comma-separated step indices at which MH chains are run', metavar='STR', type=str, default='', show_default=True)
@click.option('--num_samples_branch',      help='Comma-separated number of samples collected per chain at each branch time', metavar='STR', type=str, default='', show_default=True)
@click.option('--snr',                     help='Signal-to-noise ratio for the Langevin step size', metavar='FLOAT', type=click.FloatRange(min=0.), default=0.23, show_default=True)
@click.option('--burn_in',                 help='Accepted proposals discarded before collecting', metavar='INT',  type=click.IntRange(min=0), default=10, show_default=True)
@click.option('--num_skip',                help='Thinning: keep every (num_skip+1)-th accepted state', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--do_mh',                   help='Apply the MH accept/reject step (0 = plain Langevin)', metavar='INT', type=click.IntRange(min=0, max=1), default=1, show_default=True)
@click.option('--dg_proposal',             help='Discriminator guidance weight inside the Langevin proposal', metavar='FLOAT', type=float, default=0., show_default=True)

def main(network_pkl, outdir, class_idx, batch_size, num_samples, save_type, do_seed, seed, device,
         pretrained_classifier_ckpt, discriminator_ckpt, cond, time_min, time_max, dg_weight,
         branch_time, num_samples_branch, snr, burn_in, num_skip, do_mh, dg_proposal, **sampler_kwargs):
    ## Load pretrained score network.
    print(f'Loading network from "{network_pkl}"...')
    with open(network_pkl, 'rb') as f:
        net = pickle.load(f)['ema'].to(device)

    ## Load discriminator (ADM feature extractor + small time-dependent classifier).
    discriminator = classifier_lib.get_discriminator(
        pretrained_classifier_ckpt, discriminator_ckpt, net.label_dim and cond, net.img_resolution, device, enable_grad=True)
    vpsde = classifier_lib.vpsde()

    ## Branch schedule.
    pair_dict = {}
    if branch_time:
        times = [int(i) for i in branch_time.split(',')]
        nums = [int(i) for i in num_samples_branch.split(',')]
        assert len(times) == len(nums), '--branch_time and --num_samples_branch must have the same length'
        pair_dict = dict(zip(times, nums))
    print(f'Branch schedule (step -> samples per chain): {pair_dict if pair_dict else "none (plain EDM)"}')

    ## Set seed.
    if do_seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    ## Pick latents and labels.
    latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[torch.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

    ## Generate images.
    print(f'Generating {num_samples} images to "{outdir}"...')
    os.makedirs(outdir, exist_ok=True)
    sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
    ac_sampler(net, discriminator, vpsde, latents, class_labels, class_idx=class_idx,
                   time_min=time_min, time_max=time_max, dg_weight=dg_weight, dg_proposal=dg_proposal,
                   snr=snr, burn_in=burn_in, num_skip=num_skip, do_mh=bool(do_mh), pair_dict=pair_dict,
                   batch_size=batch_size, num_samples=num_samples, outdir=outdir, save_type=save_type, do_seed=do_seed,
                   **sampler_kwargs)

#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
