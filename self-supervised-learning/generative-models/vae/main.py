
import torch
import torch.nn as nn
from tqdm import tqdm

from vae import VAE
from data import get_dataloader
from loss import VAELoss

def train_one_epoch(vae, dataloader, optimizer, loss_func, device):
    """
    Trains the VAE for one epoch
    """
    vae.train()
    
    train_loss = 0
    train_reconstruction_loss = 0
    train_kl_loss = 0
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

def train(num_epochs, model, dataloader, optimizer, loss_func, device, patience):
    """
    Trains the VAE for N epochs
    """
    train_losses = []
    train_reconstruction_losses = []
    train_kl_losses = []
    tolerance = patience
    best_train_loss = float("inf")

    for i in range(num_epochs):
        train_loss, train_reconstruction_loss, train_kl_loss = train_one_epoch(model, dataloader, optimizer, loss_func, device)
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
    
        print(f"Epoch {i} | Train loss: {train_loss} | KL: {train_kl_loss} | Reconstruction: {train_reconstruction_loss}")

    return train_losses

if __name__ == "__main__":
    batch_size = 32
    max_timestamp = 1000
    pe_dim = 512
    lr = 1e-4
    num_epochs = 10
    patience = 5
    beta_kl = 1
    json_path = "../../../data/full_simplified_cow.ndjson"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataloader = get_dataloader(batch_size, json_path)

    vae = VAE()
    vae = vae.to(device)
    loss_func = VAELoss(latent_dim=(4, 32, 32), beta_kl=beta_kl, device=device)
    
    optimizer = torch.optim.Adam(vae.parameters()) 

    train(num_epochs, vae, dataloader, optimizer, loss_func, device, patience)
