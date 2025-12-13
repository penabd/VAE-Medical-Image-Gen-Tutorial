import torch
import torch.nn as nn

def cvae_loss(recon_x, x, mu, logvar):
    batch_size = x.size(0)

    recon_loss = nn.functional.mse_loss(
        recon_x, x, reduction='sum'
    ) / batch_size

    kl = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    ) / batch_size

    return recon_loss + kl

