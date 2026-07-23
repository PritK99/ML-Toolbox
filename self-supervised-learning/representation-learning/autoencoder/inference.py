"""
This does the same job as visualize interpolations in train.py
But this is an extra utility to play once we have trained model
"""
import os
import torch
import torch.nn as nn

from model import AutoEncoder
from config import Config
from utils import visualize_interpolations

if __name__ == "__main__":
    config = Config()
    model_path = "results/autoencoder.pth"
    interpolation_dir = "interpolations"

    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = AutoEncoder(config.latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    visualize_interpolations(interpolation_dir, model, device, "inference")