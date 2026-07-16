import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_reconstructions(model, val_dataloader, device, epoch):
    """
    This is to visualize the reconstruction quality after each epoch
    """
    original = []
    reconstructed = []

    model.eval()
    with torch.no_grad():
        for img, label in val_dataloader:
            img = img.to(device)
            out, latent = model(img)

            for i in range(5):
                original.append(img[i].cpu().numpy().squeeze(0))
                reconstructed.append(out[i].cpu().numpy().squeeze(0))
            
            break 
    
    fig, axes = plt.subplots(5, 2, figsize=(8, 16))
    for i in range(5):
        axes[i, 0].imshow(original[i], cmap="gray")
        axes[i, 1].imshow(reconstructed[i], cmap="gray")
    
    plt.suptitle(f"Reconstructions for epoch {epoch}")
    plt.tight_layout()
    plt.savefig(f"plots/reconstructions_{epoch}.png")
    plt.close()

def plot_latent_space(latents_2d, labels, epoch):
    """
    This is to visualize the latents using PCA after each epoch
    """
    plt.figure(figsize=(8, 8))

    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap="coolwarm", s=8, alpha=0.7)
    plt.colorbar(scatter, ticks=[0, 1], label="Class")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Latent Space")

    plt.suptitle(f"PCA plot for epoch {epoch}")
    plt.tight_layout()
    plt.savefig(f"plots/pca_{epoch}.png")
    plt.close()

def visualize_interpolations(model, val_loader, device, epoch):
    """
    We will visualize the interpolations between a cow and bulldozer
    This is the most exciting part of training an autoencoder
    Because sketches don't blend very well, the results are not meaningful
    """
    cow_samples = []
    bulldozer_samples = []

    val_dataset = val_loader.dataset

    indices = list(range(len(val_dataset)))
    random.shuffle(indices)

    for i in indices:
        img, label = val_dataset[i]

        if (label == 0 and len(cow_samples) < 5):
            cow_samples.append(img)
        
        elif (label == 1 and len(bulldozer_samples) < 5):
            bulldozer_samples.append(img)
        
        if len(cow_samples) == 5 and len(bulldozer_samples) == 5:
            break
    
    cow_samples = torch.stack(cow_samples).to(device)
    bulldozer_samples = torch.stack(bulldozer_samples).to(device)

    model.eval()
    with torch.no_grad():
        cow_reconstructions, cow_latents = model(cow_samples)
        bulldozer_reconstructions, bulldozer_latents = model(bulldozer_samples)
    
    fig, axes = plt.subplots(5, 5, figsize=(16, 16))
    for j in range(5):
        cow_latent = cow_latents[j]
        cow_reconstruction = cow_reconstructions[j].cpu()

        bulldozer_latent = bulldozer_latents[j]
        bulldozer_reconstruction = bulldozer_reconstructions[j].cpu()

        interpolations = []
        steps = [0.25, 0.5, 0.75]
        with torch.no_grad():
            for x in steps:
                new_latent = (1-x)*cow_latent + x*bulldozer_latent
                interpolations.append(new_latent)
            interpolations = torch.stack(interpolations).to(device)
            reconstructions, latents = model(interpolations)

        axes[j, 0].imshow(cow_reconstruction.squeeze(0), cmap="gray")
        axes[j, 0].set_title("0")

        for k in range(len(steps)):
            axes[j, k+1].imshow(reconstructions[k].cpu().squeeze(0), cmap="gray")
            axes[j, k+1].set_title(steps[k])

        axes[j, 4].imshow(bulldozer_reconstruction.squeeze(0), cmap="gray")
        axes[j, 4].set_title("1")
    
    plt.suptitle(f"Interpolations for epoch {epoch}")
    plt.tight_layout()
    plt.savefig(f"plots/interpolation_{epoch}")
    plt.close()