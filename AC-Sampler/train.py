"""Train the time-dependent discriminator on CIFAR-10.

The discriminator classifies noisy real images vs. noisy generated images at a
random diffusion time. Its logit is the estimated log density ratio
log p_real(x_t) / p_gen(x_t) that AC-Sampler uses in the MH acceptance ratio.
Follows the DG code base (Kim et al., 2023).
"""

import os
import glob

import click
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import classifier_lib
import dnnlib


def npz_concat(filenames, cond=False):
    samples, labels = [], []
    for file in filenames:
        d = np.load(file)
        samples.append(d['samples'])
        if cond:
            labels.append(d['label'])
    samples = np.concatenate(samples)
    if cond:
        return samples, np.concatenate(labels)
    return samples


class BasicDataset(data.Dataset):
    def __init__(self, x_np, y_np, cond_np=None, transform=transforms.ToTensor()):
        super().__init__()
        self.x = x_np
        self.y = y_np
        self.cond = cond_np
        self.transform = transform

    def __getitem__(self, index):
        if self.cond is None:
            return self.transform(self.x[index]), self.y[index]
        return self.transform(self.x[index]), self.y[index], self.cond[index]

    def __len__(self):
        return len(self.x)


@click.command()
@click.option('--savedir',                    help='Directory to save discriminator checkpoints', metavar='PATH', type=str, required=True)
@click.option('--gendir',                     help='Directory with generated samples (samples_*.npz)', metavar='PATH', type=str, required=True)
@click.option('--datadir',                    help='Real data npz (key "arr_0" or "samples")', metavar='PATH',     type=str, required=True)
@click.option('--img_resolution',             help='Image resolution', metavar='INT',                              type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--cond',                       help='Conditional discriminator', metavar='INT',                     type=click.IntRange(min=0, max=1), default=0, show_default=True)
@click.option('--pretrained_classifier_ckpt', help='Path of ADM classifier (feature extractor)', metavar='PATH',   type=str, default='checkpoints/ADM_classifier/32x32_classifier.pt', show_default=True)
@click.option('--num_data',                   help='Number of real/fake images used', metavar='INT',               type=click.IntRange(min=1), default=50000, show_default=True)
@click.option('--batch_size',                 help='Batch size', metavar='INT',                                    type=click.IntRange(min=1), default=128, show_default=True)
@click.option('--epoch',                      help='Number of epochs', metavar='INT',                              type=click.IntRange(min=1), default=60, show_default=True)
@click.option('--lr',                         help='Learning rate', metavar='FLOAT',                               type=click.FloatRange(min=0), default=3e-4, show_default=True)
@click.option('--device',                     help='Device', metavar='STR',                                        type=str, default='cuda:0', show_default=True)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    os.makedirs(opts.savedir, exist_ok=True)

    ## Real data.
    real = np.load(opts.datadir)
    real_data = real['arr_0'] if 'arr_0' in real else real['samples']
    real_label = np.eye(10)[real['label']] if opts.cond else None

    ## Generated data (cached as a single npz after the first run).
    cache = os.path.join(opts.gendir, 'gen_data_for_discriminator_training.npz')
    if not os.path.exists(cache):
        filenames = np.sort(glob.glob(os.path.join(opts.gendir, 'samples*.npz')))
        if opts.cond:
            gen_data, gen_label = npz_concat(filenames, cond=True)
            np.savez_compressed(cache, samples=gen_data, label=gen_label)
        else:
            gen_data = npz_concat(filenames)
            np.savez_compressed(cache, samples=gen_data)
    gen = np.load(cache)
    gen_data = gen['samples']
    gen_label = gen['label'] if opts.cond else None

    ## Combine real (label 1) and fake (label 0).
    real_data = real_data[:opts.num_data]
    gen_data = gen_data[:opts.num_data]
    train_data = np.concatenate((real_data, gen_data))
    train_label = torch.zeros(train_data.shape[0])
    train_label[:real_data.shape[0]] = 1.
    condition = None
    if opts.cond:
        condition = np.concatenate((real_label[:opts.num_data], gen_label[:opts.num_data]))
    train_dataset = BasicDataset(train_data, train_label, condition, transforms.ToTensor())
    train_loader = data.DataLoader(train_dataset, batch_size=opts.batch_size, num_workers=0, shuffle=True, drop_last=True)

    ## Feature extractor (frozen) and discriminator.
    pretrained_classifier = classifier_lib.load_classifier(opts.pretrained_classifier_ckpt, opts.img_resolution, opts.device, eval=False)
    discriminator = classifier_lib.load_discriminator(None, opts.device, opts.cond, eval=False)

    vpsde = classifier_lib.vpsde()
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=opts.lr, weight_decay=1e-7)
    loss_fn = torch.nn.BCELoss()
    scaler = lambda x: 2. * x - 1.

    ## Training.
    for i in range(opts.epoch):
        losses, accs = [], []
        for batch in train_loader:
            optimizer.zero_grad()
            if opts.cond:
                inputs, labels, cond = batch
                cond = cond.to(opts.device)
            else:
                inputs, labels = batch
                cond = None
            inputs = scaler(inputs.to(opts.device))
            labels = labels.to(opts.device)

            # Perturb with the VP-SDE at a random time.
            t, _ = vpsde.get_diffusion_time(inputs.shape[0], inputs.device)
            mean, std = vpsde.marginal_prob(t)
            perturbed_inputs = mean[:, None, None, None] * inputs + std[:, None, None, None] * torch.randn_like(inputs)

            with torch.no_grad():
                feature = pretrained_classifier(perturbed_inputs, timesteps=t, feature=True)
            prediction = discriminator(feature, t, sigmoid=True, condition=cond).view(-1)

            loss = loss_fn(prediction, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            accs.append(((prediction > 0.5).float() == labels).float().mean().item())
        print(f'epoch {i + 1}/{opts.epoch}  BCE loss: {np.mean(losses):.4f}  accuracy: {np.mean(accs):.4f}')
        torch.save(discriminator.state_dict(), os.path.join(opts.savedir, f'discriminator_{i + 1}.pt'))

#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
