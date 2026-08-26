import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from config import Config
from model import AutoEncoder
from data import get_dataloaders
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

    return train_loss, val_loss

def train(num_epochs, model, train_dataloader, val_dataloader, optimizer, loss_func, interpolation_dir, device, patience):
    """
    Trains the autoencoder for N epochs
    """
    train_losses = []
    val_losses = []
    tolerance = patience
    best_val_loss = float("inf")

    for i in range(num_epochs):
        train_loss, val_loss = train_one_epoch(model, train_dataloader, val_dataloader, optimizer, loss_func, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (train_loss < best_val_loss):
            tolerance = patience
            best_val_loss = train_loss
            torch.save(model.state_dict(), "results/autoencoder.pth")

        else:
            tolerance -= 1
            if (tolerance < 1):
                print("Can't be patient anymore.")

        # This is to visualize the autoencoder representations after each epoch
        visualize_reconstructions(model, val_dataloader, device, i)
        plot_latent_space(model, train_dataloader, device, i)

        # We visualize the interpolations only for the last epoch
        if (i == num_epochs - 1):
            visualize_interpolations(interpolation_dir, model, device, i)

        print(f"Epoch {i} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

    return train_losses, val_losses

def test(model, test_dataloader, loss_func, device):
    """
    Testing the model reconstruction through RMSE
    """
    model.eval()
    mse = 0
    with torch.no_grad():
        for img, label in tqdm(test_dataloader, desc="Testing"):
            img = img.to(device)

            out, latent = model(img)
            loss = loss_func(out, img)
            mse += loss.item() * img.size(0)
    
    mse /= len(test_dataloader.dataset)
    rmse = np.sqrt(mse)

    return rmse

if __name__ == "__main__":
    config = Config()

    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = AutoEncoder(config.latent_dim)
    model = model.to(device)

    train_dataloader, val_dataloader, test_dataloader, id_to_class = get_dataloaders(config.batch_size, config.train_path, config.test_path, config.synset_mapping_file)

    optimizer = torch.optim.Adam(model.parameters(), config.lr) 
    loss_func = nn.MSELoss()

    train_losses, val_losses = train(config.num_epochs, model, train_dataloader, val_dataloader, optimizer, loss_func, config.interpolation_dir, device, config.patience)
    rmse = test(model, test_dataloader, loss_func, device)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.title("Train & Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/train_val_loss.png")
    plt.close()

    print(f"Reconstruction RMSE over test set: {rmse}")