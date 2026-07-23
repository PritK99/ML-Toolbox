import torch
import torch.nn as nn
from tqdm import tqdm
from torchinfo import summary

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.encoder_conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),    # (3, 256, 256) -> (32, 256, 256)
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    # (32, 256, 256) -> (32, 128, 128)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),    # (32, 128, 128) -> (64, 128, 128)
            nn.GroupNorm(num_groups=2, num_channels=64),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    # (64, 128, 128) -> (64, 64, 64)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),    # (64, 64, 64) -> (128, 64, 64)
            nn.GroupNorm(num_groups=4, num_channels=128),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    # (128, 64, 64) -> (128, 32, 32)

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),    # (128, 32, 32) -> (256, 32, 32)
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    # (256, 32, 32) -> (256, 16, 16)

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),    # (256, 16, 16) -> (512, 16, 16)
            nn.GroupNorm(num_groups=16, num_channels=512),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    # (512, 16, 16) -> (512, 8, 8)

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),    # (512, 8, 8) -> (512, 8, 8)
            nn.GroupNorm(num_groups=16, num_channels=512),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    # (512, 8, 8) -> (512, 4, 4)
        )

        self.encoder_mlp = nn.Linear(512*4*4, latent_dim)

        self.decoder_mlp = nn.Linear(latent_dim, 512*4*4)

        self.decoder_conv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2),     # (512, 4, 4) -> (512, 8, 8)
            nn.GroupNorm(num_groups=16, num_channels=512),
            nn.SiLU(),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2),     # (512, 8, 8)-> (256, 16, 16)
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.SiLU(),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),     # (256, 16, 16) -> (128, 32, 32)
            nn.GroupNorm(num_groups=4, num_channels=128),
            nn.SiLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),     # (128, 32, 32) -> (64, 64, 64)
            nn.GroupNorm(num_groups=2, num_channels=64),
            nn.SiLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),     # (64, 64, 64)-> (32, 64, 64)
            nn.SiLU(),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2),     # (32, 64, 64) -> (32, 128, 128)
            nn.SiLU(),

            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1),    # (32, 256, 256) -> (3, 256, 256)
        )
    
    def forward(self, x):
        """
        The forward function can work with both image and latent as an input
        If input is latent, we will skip the encoder conv layers
        To identify the input, we just check its shape
        """
        if (len(x.shape) > 2):    # This would mean it is an image
            z = self.encoder_conv_layers(x)    # (1, 256, 256) -> (256, 4, 4)
            z = z.view(z.shape[0], -1)    # (256, 4, 4) -> (1, 256*4*4)
            latent = self.encoder_mlp(z)    # (1, 512*4*4) -> (latent_dim,)
        else:
            latent = x
        
        z = self.decoder_mlp(latent)    # (latent_dim,) -> (1, 256*4*4)

        z = z.view(z.shape[0], 512, 4, 4)    # (1,5126*4*4) -> (512, 4, 4)
        out = self.decoder_conv_layers(z)    # (512, 4, 4) -> (1, 256, 256)

        return out, latent

if __name__ == "__main__":
    latent_dim = 512
    model = AutoEncoder(latent_dim)

    summary(model, (16, 3, 256, 256))