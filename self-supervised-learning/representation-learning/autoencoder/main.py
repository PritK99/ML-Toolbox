import os
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import get_dataloader
from autoencoder import AutoEncoder

def visualize_reconstructions(model, val_dataloader, device, epoch):
    original = []
    reconstructed = []

    model.eval()
    with torch.no_grad():
        for img, label in val_dataloader:
            img = img.to(device)
            out = model(img)

            out = torch.sigmoid(out)    # We need to apply sigmoid for visualization if we use BCE loss

            for i in range(5):
                original.append(img[i].cpu().numpy().squeeze(0))
                reconstructed.append(out[i].cpu().numpy().squeeze(0))
            
            break 
    
    fig, axes = plt.subplots(5, 2, figsize=(8, 16))
    for i in range(5):
        axes[i, 0].imshow(original[i], cmap="gray")
        axes[i, 1].imshow(reconstructed[i], cmap="gray")
    
    plt.suptitle(f"Reconstruction for epoch {epoch}")
    plt.tight_layout()
    plt.savefig(f"plots/reconstruction_{epoch}.png")
    plt.close()

def train_one_epoch(model, train_dataloader, val_dataloader, optimizer, loss_func, device):
    """
    Trains the model for one epoch
    """
    model.train()
    train_loss = 0
    for img, label in tqdm(train_dataloader, desc="Training"):
        optimizer.zero_grad()
        img = img.to(device)

        out = model(img)
        loss = loss_func(out, img)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for img, label in tqdm(val_dataloader, desc="Validating"):
            img = img.to(device)

            out = model(img)
            loss = loss_func(out, img)

            val_loss += loss.item()
    
    train_loss /= len(train_dataloader)
    val_loss /= len(val_dataloader)

    return train_loss, val_loss

def train(num_epochs, model, train_dataloader, val_dataloader, optimizer, loss_func, device, patience):
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
        else:
            tolerance -= 1

            if (tolerance < 1):
                print("Can't be patient anymore.")

        visualize_reconstructions(model, val_dataloader, device, i)
        print(f"Epoch {i} | Train loss: {train_loss:.2f} | Val loss: {val_loss:.2f}")

    return train_losses, val_losses

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    latent_dim = 512
    model = AutoEncoder(latent_dim)
    model = model.to(device)

    batch_size = 32
    cow_json_path = "../../../data/full_simplified_cow.ndjson"
    bulldozer_json_path = "../../../data/full_simplified_bulldozer.ndjson"
    val_ratio = 0.1
    train_dataloader, val_dataloader = get_dataloader(batch_size, cow_json_path, bulldozer_json_path, val_ratio)

    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr) 

    # loss_func = nn.MSELoss()

    pos_weight = torch.tensor([4.0]).to(device)
    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    num_epochs = 20
    patience = 10
    train_losses, val_losses = train(num_epochs, model, train_dataloader, val_dataloader, optimizer, loss_func, device, patience)
    
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plots/train_loss.png")
    plt.close()

    plt.figure()
    plt.plot(train_losses, label="val")
    plt.title("Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plots/val_loss.png")
    plt.close()