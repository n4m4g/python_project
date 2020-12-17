#!/usr/bin/env python3

import time
import argparse
from collections import defaultdict

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from model import VAE


def epoch_time(start_t, end_t):
    t = end_t - start_t
    m = int(t / 60)
    s = int(t - m*60)
    return m, s


def loss_fn(recon_x, x, m, s):
    BSE = nn.functional.binary_cross_entropy(recon_x.view(-1, 28*28),
                                             x.view(-1, 28*28),
                                             reduction='sum')
    KLD = -0.5 * torch.sum(1 + s - m.pow(2) - s.exp())

    return (BSE + KLD) / x.size(0)


def main(args):
    torch.manual_seed(args.seed)
    device = None
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataset = MNIST(root='data',
                    train=True,
                    transform=transforms.ToTensor(),
                    download=True)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=4)

    # x, y = next(iter(data_loader))
    # print(x.shape, y.shape)
    # torch.Size([batch_size, 1, 28, 28]) torch.Size([batch_size])

    vae = VAE(args.en_layer_size,
              args.latent_size,
              args.de_layer_size,
              args.conditional,
              num_labels=10 if args.conditional else 0)
    vae = vae.to(device)

    optimizer = optim.Adam(vae.parameters(), lr=args.lr)

    logs = defaultdict(list)

    total_t = time.time()
    for epoch in range(args.epochs):
        # tracker_epoch = defaultdict(lambda: defaultdict(dict))
        start_t = time.time()
        for iteration, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            # x.shape = (batch_size, 1, 28, 28)
            # y.shape = (batch_size)

            if args.conditional:
                recon_x, m, s, z = vae(x, y)
            else:
                recon_x, m, s, z = vae(x)
            # recon_x.shape = (batch_size, layer_size[-1])
            # m.shape = (batch_size, latent_size)
            # s.shape = (batch_size, latent_size)
            # z.shape = (batch_size, latent_size)

            # for i, yi in enumerate(y):
            #     idx = len(tracker_epoch)
            #     tracker_epoch[idx]['x'] = z[i, 0].item()
            #     tracker_epoch[idx]['y'] = z[i, 1].item()
            #     tracker_epoch[idx]['label'] = yi.item()

            loss = loss_fn(recon_x, x, m, s)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

        end_t = time.time()
        m, s = epoch_time(start_t, end_t)
        total_m, total_s = epoch_time(total_t, end_t)
        print("Epoch {}/{} Batch {}/{}, Loss {:.4f}, {}m{}s/{}m{}s".format(
              epoch+1, args.epochs,
              iteration+1, len(data_loader),
              loss.item(),
              m, s,
              total_m, total_s))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--en_layer_size", type=list, default=[784, 256])
    parser.add_argument("--de_layer_size", type=list, default=[256, 784])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fit_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()
    main(args)
