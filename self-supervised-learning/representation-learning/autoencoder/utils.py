import os 
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def denormalize(img):
    """
    We need to denormalize tensors before plotting, else everything gets washed out
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device)

    if (len(img.shape) > 3):
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)

    img = (img * std) + mean
    img = torch.clip(img, 0, 1)

    return img

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
                original.append(denormalize(img[i]).cpu().permute(1, 2, 0).numpy())
                reconstructed.append(denormalize(out[i]).cpu().permute(1, 2, 0).numpy())
            
            break 
    
    fig, axes = plt.subplots(5, 2, figsize=(8, 16))
    for i in range(5):
        axes[i, 0].imshow(original[i])
        axes[i, 1].imshow(reconstructed[i])
    
    plt.suptitle(f"Reconstructions for epoch {epoch}")
    plt.tight_layout()
    plt.savefig(f"results/reconstructions_{epoch}.png")
    plt.close()

def plot_latent_space(model, train_dataloader, device, epoch):
    """
    This is to visualize the latents using PCA after each epoch
    """
    count = 0
    latents = []
    labels = []

    model.eval()
    with torch.no_grad():
        for img, label in train_dataloader:
            mask = label < 10    # We only consider first 10 classes for visualization

            if not mask.any():
                continue
                
            chosen_imgs = img[mask].to(device)
            chosen_labels = label[mask]
            
            out, latent = model(chosen_imgs)

            latents.append(latent.detach().cpu())
            labels.append(chosen_labels.detach().cpu())

            count += chosen_imgs.size(0)
            
            if (count > 10000):    # This is approx 1000 samples from each class
                break
        
    latents = torch.cat(latents, dim=0)
    labels = torch.cat(labels, dim=0)
    latents = latents.numpy()
    labels = labels.numpy()

    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap="tab10", s=8, alpha=0.7)
    plt.colorbar(scatter, ticks=np.arange(10), label="Class")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Latent Space")
    plt.suptitle(f"PCA plot for epoch {epoch}")
    
    plt.tight_layout()
    plt.savefig(f"results/pca_{epoch}.png")
    plt.close()

def visualize_interpolations(interpolation_dir, model, device, epoch):
    """
    This is the most exciting part of training an autoencoder
    Here, we visualize the latent space interpolations between pairs of images
    The `interpolation_dir` directory must follow this exact structure:

    interpolation_dir/
    ├── pair1/
    │   ├── 1.png
    │   └── 2.png
    ├── pair2/
    │   ├── 1.png
    │   └── 2.png
    └── ...
    """
    mean = [0.485, 0.456, 0.406]    # These are the test transforms which we need to apply to all images
    std = [0.229, 0.224, 0.225]
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])
    
    all_images = []
    pairs = os.listdir(interpolation_dir)
    for pair in pairs:
        pair_dir = os.path.join(interpolation_dir, pair)
        pair_images = os.listdir(pair_dir)

        if (len(pair_images) != 2):
            print(f"We need exact 2 images in {pair_dir}")
            return 

        for img_name in pair_images:
            img_path = os.path.join(interpolation_dir, pair, img_name)
            img = Image.open(img_path)

            transformed_img = transforms(img)
            all_images.append(transformed_img)
    
    all_images = torch.stack(all_images)

    model.eval()
    with torch.no_grad():
        all_images = all_images.to(device)
        out, latents = model(all_images)

    for i in range(len(pairs)):
        fig, axes = plt.subplots(1, 5, figsize=(16, 4))
        latent1 = latents[2*i]
        latent2 = latents[2*i + 1]
        reconstruction1 = out[2*i]
        reconstruction2 = out[2*i + 1]

        # We interpolate between latent1 and latent2
        interpolations = []
        steps = [0.25, 0.5, 0.75]
        with torch.no_grad():
            for x in steps:
                new_latent = (1-x)*latent1 + (x)*latent2
                interpolations.append(new_latent)
            interpolations = torch.stack(interpolations).to(device)
            reconstructions, latents_back = model(interpolations)

        axes[0].imshow(denormalize(reconstruction1).cpu().permute(1, 2, 0).squeeze(0), cmap="gray")
        axes[0].set_title("0")

        for k in range(len(steps)):
            axes[k+1].imshow(denormalize(reconstructions[k]).cpu().permute(1, 2, 0).squeeze(0), cmap="gray")
            axes[k+1].set_title(steps[k])

        axes[4].imshow(denormalize(reconstruction2).cpu().permute(1, 2, 0).squeeze(0), cmap="gray")
        axes[4].set_title("1")
    
        plt.suptitle(f"Interpolations for epoch {epoch}")
        plt.tight_layout()
        plt.savefig(f"results/interpolation_pair_{i}")
        plt.close()