import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary

class TimeEmbeddings(nn.Module):
    """
    This is to convert timestep into embeddings
    The scalar value time is converted into an embedding using sinusoidal formula
    This embedding is passed to an MLP to obtain time embedding
    Time embedding is used in every residual block, and it's size is adjusted using a projection layer at each block
    """
    def __init__(self, max_timestep, dim):
        super().__init__()

        pe = torch.zeros(max_timestep, dim)
        position = torch.arange(0, max_timestep, dtype=torch.float).unsqueeze(1) 
        denominator = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))    # Taking log and exponential is more stable

        pe[:, 0::2] = torch.sin(position * denominator)
        pe[:, 1::2] = torch.cos(position * denominator)

        pe = pe.unsqueeze(0)    # This makes it (1, seq_len, d_model) for broadcasting later
        self.register_buffer('pe', pe)    # We need to register pe as buffer so that it can be moved to GPU during training

        self.mlp = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512)
        )

    def forward(self, timestamps):
        sinusoidal = self.pe[0, timestamps]

        if sinusoidal.dim() == 1:    # This will help during inference time
            sinusoidal = sinusoidal.unsqueeze(0)

        positional_encoding = self.mlp(sinusoidal)
        return positional_encoding

class ResidualBlock(nn.Module):
    """
    This is the residual block in UNet
    We also need to include a projection layer for managing time embeddings
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.silu = nn.SiLU()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=(out_channels // 4), num_channels=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=(out_channels // 4), num_channels=out_channels)

        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.projection_mlp = nn.Linear(512, out_channels)
    
    def forward(self, z, time_embedding):
        x = self.norm1(self.conv1(z))

        projected_time_embedding = self.projection_mlp(time_embedding)
        projected_time_embedding = projected_time_embedding[:, :, None, None]
        x += projected_time_embedding

        x = self.silu(x)
        x = self.norm2(self.conv2(x))

        residual = self.conv1x1(z)

        return residual + x

class UNet(nn.Module):
    """
    UNet is used as the main architecture
    """
    def __init__(self, max_timestamp, pe_dim):
        super().__init__()
        self.timer = TimeEmbeddings(max_timestamp, pe_dim)

        self.down1 = ResidualBlock(in_channels=4, out_channels=32)    # (4, 32, 32) -> (32, 32, 32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)    # (32, 32, 32) -> (32, 16, 16)

        self.down2 = ResidualBlock(in_channels=32, out_channels=64)    # (32, 16, 16) -> (64, 16, 16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)    # (64, 16, 16) -> (64, 8, 8)

        self.down3 = ResidualBlock(in_channels=64, out_channels=128)    # (64, 8, 8) -> (128, 8, 8)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)    # (128, 8, 8) -> (128, 4, 4)

        self.bottleneck = ResidualBlock(in_channels=128, out_channels=256)    # (128, 4, 4) -> (256, 4, 4)

        self.conv_transpose3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)     # (256, 4, 4) -> (128, 8, 8)
        self.up3 = ResidualBlock(in_channels=256, out_channels=128)    # (256, 8, 8) -> (128, 8, 8)

        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)     # (128, 8, 8) -> (64, 16, 16)
        self.up2 = ResidualBlock(in_channels=128, out_channels=64)    # (128, 16, 16) -> (64, 16, 16)

        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)     # (64, 16, 16) -> (32, 32, 32)
        self.up1 = ResidualBlock(in_channels=64, out_channels=32)    # (64, 32, 32) -> (32, 32, 32)

        self.conv1x1 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)    # (32, 32, 32) -> (4, 32, 32)
    
    def forward(self, x, time):
        time_embedding = self.timer(time)

        z1 = self.down1(x, time_embedding)
        z = self.maxpool1(z1)

        z2 = self.down2(z, time_embedding)
        z = self.maxpool2(z2)

        z3 = self.down3(z, time_embedding)
        z = self.maxpool3(z3)

        z = self.bottleneck(z, time_embedding)

        z = self.conv_transpose3(z)
        z = self.up3(torch.concat([z, z3], dim=1), time_embedding)

        z = self.conv_transpose2(z)
        z = self.up2(torch.concat([z, z2], dim=1), time_embedding)

        z = self.conv_transpose1(z)
        z = self.up1(torch.concat([z, z1], dim=1), time_embedding)

        z = self.conv1x1(z)

        return z

if __name__ == "__main__":
    max_timestamp = 1000
    pe_dim = 512

    unet = UNet(max_timestamp=max_timestamp, pe_dim=pe_dim)
    summary(unet)