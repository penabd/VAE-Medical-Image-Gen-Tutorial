import torch
import torch.nn as nn

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            recon, mu, logvar = model(imgs, labels)
            loss = cvae_loss(recon, imgs, mu, logvar)
            total_loss += loss.item()

    return total_loss / len(loader)
    