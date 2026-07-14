import os
import cv2
import json
import numpy as np
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class QuickDrawDataset(Dataset):
    """
    QuickDraw dataset provides an ndjson file for each class
    Each line in the ndjson file contains a drawing
    For now, we only use the cow class which has ~1L images
    https://github.com/googlecreativelab/quickdraw-dataset
    """
    def __init__(self, json_path, transforms):
        super().__init__()

        # First we will load and filter out all the good drawings
        self.drawings = []
        with open(json_path, "r") as file:
            for line in file:
                data = json.loads(line)

                if (data["recognized"] == False):
                    continue

                strokes = data["drawing"]
                self.drawings.append(strokes)
        
        self.transforms = transforms
        
    def __len__(self):
        return len(self.drawings)

    def __getitem__(self, index):
        """
        The data contains stroke points
        We need to use them and connect the points to obtain an image
        """
        strokes = self.drawings[index]

        drawing = np.zeros((256, 256), dtype=np.uint8)
        for stroke in strokes:
            x_coords = stroke[0]
            y_coords = stroke[1]

            for i in range(len(x_coords) - 1):
                cv2.line(drawing, (x_coords[i], y_coords[i]), (x_coords[i+1], y_coords[i+1]), color=255)
        
        drawing = self.transforms(drawing)
        
        return drawing

def get_dataloader(batch_size, json_path):
    """
    This function applies transforms and returns a dataloader
    """
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(degrees=15),
    ])

    dataset = QuickDrawDataset(json_path, transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

    return dataloader

# This is just to test the dataloader function
if __name__ == "__main__":
    batch_size = 16
    json_path = "../../../data/full_simplified_cow.ndjson"

    dataloader = get_dataloader(batch_size, json_path)

    os.makedirs("plots", exist_ok=True)
    for img in dataloader:
        print(img.shape)
        
        fig, axes = plt.subplots(2, 3)
        for i in range(2):
            for j in range(3):
                axes[i, j].imshow(img[2*i + j].detach().squeeze(0), cmap="gray")
        
        plt.tight_layout()
        plt.savefig("plots/data_samples.png")
        plt.close()

        break