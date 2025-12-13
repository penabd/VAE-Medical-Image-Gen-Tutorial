import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim=32, label_dim=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
        )
        self.flatten = nn.Flatten()

        self.fc_mu = nn.Linear(128*16*16 + label_dim, latent_dim)
        self.fc_logvar = nn.Linear(128*16*16 + label_dim, latent_dim)

    def forward(self, x, y):
        h = self.conv(x)
        h = self.flatten(h)
        h = torch.cat([h, y], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)



class Decoder(nn.Module):
    def __init__(self, latent_dim=32, label_dim=1):
        super().__init__()
        self.fc = nn.Linear(latent_dim + label_dim, 128*16*16)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z, y):
        h = torch.cat([z, y], dim=1)
        h = self.fc(h).view(-1, 128, 16, 16)
        return self.deconv(h)



class CVAE(nn.Module):
    def __init__(self, latent_dim=32, label_dim=1):
        super().__init__()
        self.encoder = Encoder(latent_dim, label_dim)
        self.decoder = Decoder(latent_dim, label_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, y)
        return recon, mu, logvar

