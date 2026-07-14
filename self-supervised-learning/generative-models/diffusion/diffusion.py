import torch
import torch.nn as nn
from torchinfo import summary

from vae import VAE
from unet import UNet

class Diffusion(nn.Module):
    """
    Combining VAE + UNet
    """
    def __init__(self, max_timestamp, pe_dim):
        super().__init__()

        self.vae = VAE()
        self.unet = UNet(max_timestamp=max_timestamp, pe_dim=pe_dim)
    
    def forward(self, img, time):
        mu, sigma = self.vae.encoder(img)    # (32, 32, 1) -> (4, 4, 64), (4, 4, 64)
        z = self.vae.reparameterization(mu, sigma)

        out_latent = self.unet(z, time)
        out = self.vae.decoder(out_latent)

        return out

if __name__ == "__main__":
    max_timestamp = 1000
    pe_dim = 512

    diffusion = Diffusion(max_timestamp=max_timestamp, pe_dim=pe_dim)
    summary(diffusion)