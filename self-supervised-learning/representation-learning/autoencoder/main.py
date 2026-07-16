import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from config import Config
from model import AutoEncoder
from data import get_dataloader
from utils import visualize_reconstructions, plot_latent_space, visualize_interpolations

def train_one_epoch(model, train_dataloader, val_dataloader, optimizer, loss_func, device):
    """
    Trains the model for one epoch
    """
    model.train()
    train_loss = 0
    latents = []    # We store them because we would later require them for plotting
    labels = []
    for iter, (img, label) in enumerate(tqdm(train_dataloader, desc="Training")):
        optimizer.zero_grad()
        img = img.to(device)

        out, latent = model(img)
        loss = loss_func(out, img)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if (iter % 10 == 0):    # We only use 10% of data for visualization
            latents.append(latent.detach().cpu())
            labels.append(label.detach().cpu())

    latents = torch.cat(latents, dim=0)
    labels = torch.cat(labels, dim=0)
    latents = latents.numpy()
    labels = labels.numpy()

    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for img, label in tqdm(val_dataloader, desc="Validating"):
            img = img.to(device)

            out, latent = model(img)
            loss = loss_func(out, img)

            val_loss += loss.item()
    
    train_loss /= len(train_dataloader)
    val_loss /= len(val_dataloader)

    return train_loss, val_loss, latents_2d, labels

def train(num_epochs, model, train_dataloader, val_dataloader, optimizer, loss_func, device, patience):
    """
    Trains the autoencoder for N epochs
    """
    train_losses = []
    val_losses = []
    tolerance = patience
    best_val_loss = float("inf")

    for i in range(num_epochs):
        train_loss, val_loss, latents_2d, labels = train_one_epoch(model, train_dataloader, val_dataloader, optimizer, loss_func, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (train_loss < best_val_loss):
            tolerance = patience
            best_val_loss = train_loss
        else:
            tolerance -= 1

            if (tolerance < 1):
                print("Can't be patient anymore.")

        # This is to visualize the autoencoder representations after each epoch
        visualize_reconstructions(model, val_dataloader, device, i)
        plot_latent_space(latents_2d, labels, i)
        visualize_interpolations(model, val_dataloader, device, i)

        print(f"Epoch {i} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

    return train_losses, val_losses

if __name__ == "__main__":
    config = Config()
    os.makedirs("plots", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = AutoEncoder(config.latent_dim)
    model = model.to(device)

    train_dataloader, val_dataloader = get_dataloader(config.batch_size, config.cow_json_path, config.bulldozer_json_path, config.val_ratio)

    optimizer = torch.optim.Adam(model.parameters(), config.lr) 
    loss_func = nn.MSELoss()

    train_losses, val_losses = train(config.num_epochs, model, train_dataloader, val_dataloader, optimizer, loss_func, device, config.patience)
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.title("Train & Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/train_val_loss.png")
    plt.close()