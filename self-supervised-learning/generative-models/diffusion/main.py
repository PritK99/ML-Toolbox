import torch
import torch.nn as nn
from tqdm import tqdm

from data import get_dataloader
from diffusion import Diffusion
from utils import NoiseScheduler

def train_one_epoch(model, dataloader, noisifier, optimizer, loss_func, device):
    """
    Trains for one epoch
    """
    model.train()
    
    train_loss = 0
    for img in tqdm(dataloader):
        optimizer.zero_grad()
        img = img.to(device)
        time = torch.randint(low=0, high=max_timestamp, size=(img.shape[0],))

        noisy_img, noise = noisifier.add_noise(img, time)
        pred_noise = model(noisy_img, time)

        loss = loss_func(pred_noise, noise)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    train_loss = train_loss / len(dataloader)

    return train_loss

def train(num_epochs, model, dataloader, noisifier, optimizer, loss_func, device, patience):
    """
    Trains for N epochs
    """
    train_losses = []
    tolerance = patience
    best_train_loss = float("inf")

    for i in range(num_epochs):
        train_loss = train_one_epoch(model, dataloader, noisifier, optimizer, loss_func, device)
        train_losses.append(train_loss)

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

    model = Diffusion(max_timestamp, pe_dim)
    noisifier = NoiseScheduler(max_timestamp)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    train_losses = train(num_epochs, model, dataloader, noisifier, optimizer, loss_func, device, patience)