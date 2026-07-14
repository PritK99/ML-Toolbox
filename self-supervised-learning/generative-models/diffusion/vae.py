import torch
import torch.nn as nn
from tqdm import tqdm

from data import get_dataloader

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

class VAELoss(nn.Module):
    def __init__(self, latent_dim, beta_kl):
        super().__init__()
        self.beta_kl = beta_kl

        prior_mu = torch.zeros(latent_dim).to(device)
        prior_logvar = torch.zeros(latent_dim).to(device)

        self.register_buffer("prior_mu", prior_mu)
        self.register_buffer("prior_logvar", prior_logvar)

        self.mse_loss = nn.MSELoss()
    
    def kl_divergence(self, mu, logvar, prior_mu, prior_logvar):
        var = torch.exp(logvar)
        prior_var = torch.exp(prior_logvar)

        kl = 0.5 * ((prior_logvar - logvar) + (var + (mu - prior_mu) ** 2) / prior_var - 1)
        return kl.sum(dim=1).mean()
    
    def forward(self, img, out, mu, logvar):
        kl_loss = self.kl_divergence(mu, logvar, self.prior_mu, self.prior_logvar)
        reconstruction_loss = self.mse_loss(out, img)
        reconstruction_loss /= img.size(0)

        total_loss = reconstruction_loss + self.beta_kl * kl_loss

        return total_loss, kl_loss, reconstruction_loss

def train_one_epoch(vae, dataloader, optimizer, loss_func, device):
    """
    Trains for one epoch
    """
    vae.train()
    
    train_loss = 0
    reconstruction_loss = 0
    kl_loss = 0
    for img in tqdm(dataloader):
        optimizer.zero_grad()
        img = img.to(device)

        out, mu, logvar = vae(img)
        loss, kl_loss, reconstruction_loss = loss_func(img, out, mu, logvar)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_reconstruction_loss += reconstruction_loss.item()
        train_kl_loss += kl_loss.item()
    
    train_loss /= len(dataloader)
    train_reconstruction_loss /= len(dataloader)
    train_kl_loss /= len(dataloader)

    return train_loss, train_reconstruction_loss, train_kl_loss

def train(num_epochs, model, dataloader, noisifier, optimizer, loss_func, device, patience):
    """
    Trains for N epochs
    """
    train_losses = []
    train_reconstruction_losses = []
    train_kl_losses = []
    tolerance = patience
    best_train_loss = float("inf")

    for i in range(num_epochs):
        train_loss, train_reconstruction_loss, train_kl_loss = train_one_epoch(model, dataloader, noisifier, optimizer, loss_func, device)
        train_losses.append(train_loss)
        train_reconstruction_losses.append(train_reconstruction_loss)
        train_kl_losses.append(train_kl_loss)

        if (train_loss < best_train_loss):
            tolerance = patience
            best_train_loss = train_loss
        else:
            tolerance -= 1

            if (tolerance < 1):
                print("Can't be patient anymore.")
    
        print(f"Epoch {i}| Train loss: {train_loss}")

    return train_losses

if __name__ == "__main__":
    batch_size = 32
    max_timestamp = 1000
    pe_dim = 512
    lr = 1e-4
    num_epochs = 10
    patience = 5
    json_path = "../data/diffusion/full_simplified_cow.ndjson"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataloader = get_dataloader(batch_size, json_path)
