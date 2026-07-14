import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from data import get_dataloader

class NoiseScheduler(nn.Module):
    """
    This class is responsible for adding noise as per given timestamp to the image
    """
    def __init__(self, max_timestamp):
        """"
        We precompute the value of alpha bars
        """
        super().__init__()
        betas = torch.linspace(1e-4, 0.2, max_timestamp)
        alphas = 1 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)

    def add_noise(self, img, timestamp):
        """
        This function adds noise to image based on the weighting in forward equations
        """
        alpha_bars_t = self.alpha_bars.to(img.device).view(-1, 1, 1, 1)    # This is because we will have to broadcast later
        noise = torch.randn_like(img)
        noisy_img = torch.sqrt(alpha_bars_t[timestamp])*img + torch.sqrt(1 - alpha_bars_t[timestamp])*noise

        return noisy_img, noise

if __name__ == "__main__":
    batch_size = 32
    max_timestamp = 10
    json_path = "../data/diffusion/full_simplified_cow.ndjson"

    dataloader = get_dataloader(batch_size, json_path)
    noisifier = NoiseScheduler(max_timestamp)

    os.makedirs("plots", exist_ok=True)
    for img in dataloader:
        time = torch.randint(low=0, high=max_timestamp, size=(batch_size,))
        noisy_img, noise = noisifier.add_noise(img, time)
        
        fig, axes = plt.subplots(5, 2, figsize=(12, 12))
        for i in range(5):
            axes[i, 0].imshow(img[i].detach().squeeze(0), cmap="gray")
            axes[i, 0].set_title(f"Original")
            axes[i, 1].imshow(noisy_img[i].detach().squeeze(0), cmap="gray")
            axes[i, 1].set_title(f"Timestamp = {time[i]}")

        plt.tight_layout()
        plt.savefig("plots/noising.png")
        plt.close()

        break
