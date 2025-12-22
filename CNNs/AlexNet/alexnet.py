import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchview import draw_graph

class AlexNet(nn.Module):
    def __init__(self, in_channels = 3):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, stride=4, padding=0)    # conv1 does not use padding
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.ffn1 = nn.Linear(in_features=9216, out_features=4096)
        self.ffn2 = nn.Linear(in_features=4096, out_features=4096)
        self.ffn3 = nn.Linear(in_features=4096, out_features=1000)
    
    def forward(self, x):
        x = self.conv1(x)    # (227, 227, 3) -> (55, 55, 96)
        x = self.relu(x)
        x = self.lrn1(x)
        x = self.maxpool1(x)    # (55, 55, 96) -> (27, 27, 96)

        x = self.conv2(x)    # (27, 27, 96) -> (27, 27, 256)
        x = self.relu(x)
        x = self.lrn2(x)
        x = self.maxpool2(x)    # (27, 27, 256) -> (13, 13, 256)

        x = self.conv3(x)    # (13, 13, 256) -> (13, 13, 384)
        x = self.relu(x)

        x = self.conv4(x)    # (13, 13, 384) -> (13, 13, 384)
        x = self.relu(x)

        x = self.conv5(x)    # (13, 13, 384) -> (13, 13, 256)
        x = self.relu(x)
        x = self.maxpool3(x)    # (13, 13, 256) -> (6, 6, 256)

        x = x.view((x.size(0), -1))    # (6, 6, 256) -> (1, 9216)

        print(x.size())
        x = self.ffn1(x)    # (1, 9216) -> (1, 4096)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.ffn2(x)    # (1, 4096) -> (1, 4096)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.ffn3(x)    # (1, 9216) -> (1, 1000)

        return x

def train_one_epoch(model, train_dataloader, val_dataloader, loss_func, optimizer, device):
    train_loss = 0
    val_loss = 0

    model.train()
    for img, label in tqdm(train_dataloader):
        img = img.to(device)
        label = label.to(device)

        pred = model(img)

        loss = loss_func(pred, label)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for img, label in tqdm(val_dataloader):
            img = img.to(device)
            label = label.to(device)

            pred = model(img)

            _, pred_class = pred.max(1)
            accuracy += pred_class.eq(label).sum().item()

            loss = loss_func(pred, label)
            val_loss += loss.item()
    
    train_loss /= len(train_dataloader)
    val_loss /= len(val_dataloader)
    accuracy /= (256*len(val_dataloader))

    return train_loss, val_loss, accuracy

def train(model, train_dataloader, val_dataloader, num_epochs, loss_func, optimizer, device):
    train_losses = []
    val_losses = []
    accuracies = []

    for epoch in range(num_epochs):
        train_loss, val_loss, accuracy = train_one_epoch(model, train_dataloader, val_dataloader, loss_func, optimizer, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(accuracy)

        checkpoint_path = os.path.join("alexnet_checkpoints", f"epoch_{epoch+1}.pt")
        torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },checkpoint_path)
        print(f"Saved model to {checkpoint_path}")

        print(f"Completed epoch {epoch}: Train loss = {train_loss}, Val loss = {val_loss}, Accuracy: {accuracy}")
    
    return train_losses, val_losses, accuracies

# Defining dataloader
train_path = "/scratch/pritk/ILSVRC/Data/CLS-LOC/train"
val_path = "/scratch/pritk/ILSVRC/Data/CLS-LOC/val"

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_path, transform=val_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

# Defining device, model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

alexnet = AlexNet()
alexnet = alexnet.to(device)
trainable_params = sum(p.numel() for p in alexnet.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

os.makedirs("alexnet_checkpoints", exist_ok=True)
model_graph = draw_graph(alexnet, input_size=(256, 3, 227, 227), device='meta')
model_graph.visual_graph.render("alexnet_checkpoints/alexnet_architecture", format="png", cleanup=True)

optimizer = torch.optim.SGD(alexnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
loss_func = nn.CrossEntropyLoss()

num_epochs = 5
train_losses, val_losses, accuracies = train(alexnet, train_dataloader, val_dataloader, num_epochs, loss_func, optimizer, device)

# Plotting loss curves and accuracy
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, marker='o', color='blue', label='Training Loss')
plt.plot(epochs, val_losses, marker='s', color='red', label='Validation Loss')
for x, y in zip(epochs, train_losses):
    plt.text(x, y, f"{y:.2f}", color='blue', fontsize=9, ha='center', va='bottom')
for x, y in zip(epochs, val_losses):
    plt.text(x, y, f"{y:.2f}", color='red', fontsize=9, ha='center', va='top')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("alexnet_checkpoints/loss_curve.png")
print("Saved loss curves to ./alexnet_checkpoints/loss_curve.png")

plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracies, marker='o', color='green', label='Validation Accuracy')
for x, y in zip(epochs, accuracies):
    plt.text(x, y, f"{y:.2f}", color='green', fontsize=9, ha='center', va='bottom')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("alexnet_checkpoints/accuracy_curve.png")
print("Saved accuracy curve to ./alexnet_checkpoints/accuracy_curve.png")