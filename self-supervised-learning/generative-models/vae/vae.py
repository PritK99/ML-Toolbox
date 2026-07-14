import torch
import torch.nn as nn
from tqdm import tqdm
from torchinfo import summary

class Encoder(nn.Module):
    """
    VAE Encoder
    We use groupnorm over batchnorm because with differences in noise level and lower batch size, batchnorm will become unstable.
    """
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),    # (1, 256, 256) -> (32, 256, 256)
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    # (32, 256, 256) -> (32, 128, 128)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),    # (32, 128, 128) -> (64, 128, 128)
            nn.GroupNorm(num_groups=16, num_channels=64),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    # (64, 128, 128) -> (64, 64, 64)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),    # (64, 64, 64) -> (128, 64, 64)
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    # (128, 64, 64) -> (128, 32, 32)
        
            nn.Conv2d(in_channels=128, out_channels=8, kernel_size=3, padding=1),    # (128, 32, 32) -> (8, 32, 32)
            nn.SiLU(),
        )

    def forward(self, img):
        z = self.conv_layers(img)    # (1, 256, 256)-> (8, 32, 32)
        mu, logvar = z.chunk(2, dim=1)    # (4, 32, 32)

        return mu, logvar

class Decoder(nn.Module):
    """
    VAE Decoder
    """
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=3, padding=1),    # (4, 32, 32) -> (128, 32, 32)
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SiLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),     # (128, 32, 32) -> (64, 64, 64)
            nn.GroupNorm(num_groups=16, num_channels=64),
            nn.SiLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),     # (64, 64, 64) -> (32, 128, 128)
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.SiLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=2, stride=2),     # (32, 128, 128) -> (1, 256, 256)
            nn.Sigmoid(),
        )
    
    def forward(self, z):
        img = self.conv_layers(z)    # (4, 32, 32) -> (1, 256, 256)

        return img

class VAE(nn.Module):
    """
    Combining VAE Encoder + reparameterization trick + VAE Decoder
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterization(self, mu, logvar):
        eps = torch.randn_like(mu)
        std = torch.exp(0.5*logvar)

        z = mu + eps*std    # (4, 4, 64)
        return z

    def forward(self, img):
        mu, logvar = self.encoder(img)
        z = self.reparameterization(mu, logvar)
        out = self.decoder(z)

        return out, mu, logvar

# This is to check if all the shapes are proper
if __name__ == "__main__":
    vae = VAE()
    summary(vae, (16, 1, 256, 256))