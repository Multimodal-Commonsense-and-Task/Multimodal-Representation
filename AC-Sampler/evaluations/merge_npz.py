"""Merge samples_*.npz batches into a single npz for evaluator.py.

evaluator.py (ADM evaluation suite) expects one npz with key "arr_0" of shape
(N, H, W, 3) uint8. Precision/Recall are computed on 10,000 images by default.
"""

import argparse
import os
from glob import glob

import numpy as np

parser = argparse.ArgumentParser(description='Merge samples_*.npz files into one npz')
parser.add_argument('--samples_dir', type=str, required=True, help='Directory containing samples_*.npz')
parser.add_argument('--num_samples', type=int, default=10000, help='Number of images to keep (random subset)')
parser.add_argument('--out', type=str, default=None, help='Output path (default: <samples_dir>/samples_merged.npz)')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

files = sorted(glob(os.path.join(args.samples_dir, 'samples_*.npz')))
merged = np.concatenate([np.load(f)['samples'] for f in files], axis=0)
print(f'Loaded {merged.shape[0]} images from {len(files)} files')

rng = np.random.default_rng(args.seed)
idx = rng.choice(merged.shape[0], size=min(args.num_samples, merged.shape[0]), replace=False)
out = args.out or os.path.join(args.samples_dir, 'samples_merged.npz')
np.savez_compressed(out, arr_0=merged[idx])
print(f'Saved {len(idx)} images to {out}')
