import os
import re
import torch
import numpy as np 
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader

from utils import denormalize
from config import Config

def get_dataloaders(batch_size, train_path, test_path, synset_mapping_file):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ])

    train_data = torchvision.datasets.ImageFolder(root=train_path, transform=train_val_transforms)
    test_data = torchvision.datasets.ImageFolder(root=test_path, transform=test_transforms)

    train_data, val_data = random_split(train_data, [0.9, 0.1])    # We will use the original val as test set, and use 10% of train set as val instead

    train_dataloader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # We just create a dictionary using synset mapping for ease
    id_to_class = {}
    synset_to_id = test_data.class_to_idx
    with open(synset_mapping_file, "r", encoding="utf-8") as f:

        lines = f.read().splitlines()
        for line in lines:
            synset, class_name = line.split(' ', 1)
            id_to_class[synset_to_id[synset]] = class_name

    return train_dataloader, val_dataloader, test_dataloader, id_to_class

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    config = Config()

    train_dataloader, val_dataloader, test_dataloader, id_to_class = get_dataloaders(config.batch_size, config.train_path, config.test_path, config.synset_mapping_file)

    for img, label in train_dataloader:
        print(img.shape)
        print(label.shape)

        fig, axes = plt.subplots(5, 1, figsize=(8, 16))
        for i in range(5):
            axes[i].imshow(denormalize(img[i]).permute(1, 2, 0).numpy())
            axes[i].set_title(id_to_class[label[i].item()])
        
        plt.suptitle("Data Samples")
        plt.tight_layout()
        plt.savefig("results/data_samples.png")
        plt.close()

        break
