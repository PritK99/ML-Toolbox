import torch
import torch.nn as nn

class VAELoss(nn.Module):
    """
    The VAE loss comprises of two terms, reconstruction loss and kl divergence
    Increasing beta_kl psuhes model to learn a stricter probability distribution at cost of reconstruction
    For beta_kl = 0, the vae becomes an autoencoder
    """
    def __init__(self, latent_dim, beta_kl, device):
        super().__init__()
        self.beta_kl = beta_kl

        prior_mu = torch.zeros(latent_dim).to(device)
        prior_logvar = torch.zeros(latent_dim).to(device)    # logvar is 0 means std is 1

        self.register_buffer("prior_mu", prior_mu)
        self.register_buffer("prior_logvar", prior_logvar)

        self.mse_loss = nn.MSELoss(reduction="none")
    
    def kl_divergence(self, mu, logvar, prior_mu, prior_logvar):
        var = torch.exp(logvar)
        prior_var = torch.exp(prior_logvar)

        kl = 0.5 * ((prior_logvar - logvar) + (var + (mu - prior_mu) ** 2) / prior_var - 1)
        return kl.sum(dim=(1, 2, 3)).mean()
    
    def forward(self, img, out, mu, logvar):
        kl_loss = self.kl_divergence(mu, logvar, self.prior_mu, self.prior_logvar)
        reconstruction_loss = self.mse_loss(out, img).flatten(1).sum(dim=1).mean()

        total_loss = reconstruction_loss + self.beta_kl * kl_loss

        return total_loss, kl_loss, reconstruction_loss