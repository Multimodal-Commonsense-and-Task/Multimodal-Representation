# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Compute FID from a directory of samples_*.npz files (EDM protocol)."""

import os
import pickle
import random
from glob import glob

import click
import numpy as np
import scipy.linalg
import torch

import dnnlib

#----------------------------------------------------------------------------

def calculate_inception_stats_npz(image_path, num_samples=50000, device=torch.device('cuda')):
    print('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=True) as f:
        detector_net = pickle.load(f).to(device)

    print(f'Loading images from "{image_path}"...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    files = glob(os.path.join(image_path, 'samples*.npz'))
    random.shuffle(files)
    count = 0
    for file in files:
        images = np.load(file)['samples']
        images = torch.tensor(images).permute(0, 3, 1, 2).to(device)
        features = detector_net(images, **detector_kwargs).to(torch.float64)
        remaining = min(images.shape[0], num_samples - count)
        mu += features[:remaining].sum(0)
        sigma += features[:remaining].T @ features[:remaining]
        count += remaining
        if count % 10000 == 0:
            print(count)
        if count >= num_samples:
            break

    assert count == num_samples, f'Found only {count} images, expected {num_samples}'
    mu /= num_samples
    sigma -= mu.ger(mu) * num_samples
    sigma /= num_samples - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()

#----------------------------------------------------------------------------

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

#----------------------------------------------------------------------------

@click.command()
@click.option('--images', 'image_path', help='Directory containing samples_*.npz', metavar='PATH', type=str, required=True)
@click.option('--ref', 'ref_path',      help='Dataset reference statistics (npz or URL)', metavar='NPZ|URL', type=str, required=True)
@click.option('--num_samples',          help='Number of samples to use', metavar='INT', type=int, default=50000, show_default=True)
@click.option('--device',               help='Device', metavar='STR', type=str, default='cuda:0', show_default=True)
def main(image_path, ref_path, num_samples, device):
    """Calculate FID for a given set of images."""
    print(f'Loading dataset reference statistics from "{ref_path}"...')
    with dnnlib.util.open_url(ref_path) as f:
        ref = dict(np.load(f))
    mu, sigma = calculate_inception_stats_npz(image_path=image_path, num_samples=num_samples, device=device)
    print('Calculating FID...')
    fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
    print(f'{os.path.basename(os.path.normpath(image_path))}, FID = {fid:g}')

#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
