import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset
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

        x = self.ffn1(x)    # (1, 9216) -> (1, 4096)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.ffn2(x)    # (1, 4096) -> (1, 4096)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.ffn3(x)    # (1, 9216) -> (1, 1000)

        return x

def train_one_epoch(model, train_dataloader, val_dataloader, loss_func, optimizer, device, validate = True):
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

    # Validation is only required when searching for best model over val dataset
    if (validate):
        model.eval()
        accuracy = 0
        val_counts = 0
        with torch.no_grad():
            for img, label in tqdm(val_dataloader):
                img = img.to(device)
                label = label.to(device)
                val_counts += len(img)

                pred = model(img)

                _, pred_class = pred.max(1)
                accuracy += pred_class.eq(label).sum().item()

                loss = loss_func(pred, label)
                val_loss += loss.item()

        train_loss /= len(train_dataloader)
        val_loss /= len(val_dataloader)
        accuracy /= val_counts
    else:
        train_loss = None
        val_loss = None
        accuracy = None
    
    return train_loss, val_loss, accuracy

def train(model, train_dataloader, val_dataloader, num_epochs, loss_func, optimizer, device):
    best_val_accuracy = 0
    optimal_num_epochs = 0
    scheduler = []
    train_losses = []
    val_losses = []
    accuracies = []
    threshold = 0.001
    plateau_count = 0
    patience = 5

    for epoch in range(num_epochs):
        train_loss, val_loss, accuracy = train_one_epoch(model, train_dataloader, val_dataloader, loss_func, optimizer, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(accuracy)

        print(f"Completed epoch {epoch}: Train loss = {train_loss}, Val loss = {val_loss}, Accuracy: {accuracy}", flush=True)
        
        # If validation accuracy starts to plateau, we reduce lr 
        if (accuracy > best_val_accuracy + threshold):    
            plateau_count = 0
            best_val_accuracy = accuracy
            optimal_num_epochs = epoch + 1

            checkpoint_path = os.path.join("alexnet_checkpoints", f"best_model.pt")
            torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },checkpoint_path)
            print(f"Saved model to {checkpoint_path}", flush=True)
        else:
            plateau_count += 1
        
        if plateau_count >= patience:
            scheduler.append(epoch + 1)    # We need to store the specific epoch where we reduced lr as we will require this for combined evaluation
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

            plateau_count = 0 
            print(f"LR reduced to {optimizer.param_groups[0]['lr']}")
    
    return train_losses, val_losses, accuracies, optimal_num_epochs, scheduler

def train_combined(model, combined_dataloader, optimal_num_epochs, scheduler, loss_func, optimizer, device):

    for epoch in range(optimal_num_epochs):
        train_loss, val_loss, accuracy = train_one_epoch(model, combined_dataloader, None, loss_func, optimizer, device, validate = False)    # We do not require validation for combined evaluation
        print(f"Completed epoch {epoch}")

        if (len(scheduler) > 0):
            if (epoch + 1) in scheduler:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1

    # Saving the final model
    checkpoint_path = os.path.join("alexnet_checkpoints", f"best_model.pt")
    torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },checkpoint_path)
    print(f"Saved model to {checkpoint_path}", flush=True)

def test(model, test_dataloader, device):
    model.eval()
    accuracy = 0
    counts = 0
    with torch.no_grad():
        for img, label in tqdm(test_dataloader):
            img = img.to(device)
            label = label.to(device)
            counts += len(img)

            pred = model(img)
            _, pred_class = pred.max(1)
            accuracy += pred_class.eq(label).sum().item()

    accuracy /= counts

    return accuracy

# Defining dataloader
train_path = "/ssd_scratch/pritk/ILSVRC/Data/CLS-LOC/train"
val_path = "/ssd_scratch/pritk/ILSVRC/Data/CLS-LOC/val"
# test_path = "/ssd_scratch/pritk/ILSVRC/Data/CLS-LOC/test"

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
# test_dataset = datasets.ImageFolder(test_path, transform=val_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
# test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)

# Defining device, model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

alexnet = AlexNet()
alexnet = alexnet.to(device)
trainable_params = sum(p.numel() for p in alexnet.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}", flush=True)

os.makedirs("alexnet_checkpoints", exist_ok=True)
# model_graph = draw_graph(alexnet, input_size=(256, 3, 227, 227), device='meta')
# model_graph.visual_graph.render("alexnet_checkpoints/alexnet_architecture", format="png", cleanup=True)

optimizer = torch.optim.SGD(alexnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
loss_func = nn.CrossEntropyLoss()

num_epochs = 90
train_losses, val_losses, accuracies, optimal_num_epochs, scheduler = train(alexnet, train_dataloader, val_dataloader, num_epochs, loss_func, optimizer, device)

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
print("Saved loss curves to ./alexnet_checkpoints/loss_curve.png", flush=True)

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
print("Saved accuracy curve to ./alexnet_checkpoints/accuracy_curve.png", flush=True)

# We dont have access to the test labels
# # We need to redefine val to align with train transforms
# val_dataset = datasets.ImageFolder(val_path, transform=train_transforms)

# # Trainng best model configuration on train and validation combined and evaluate on test set
# combined_dataset = ConcatDataset([train_dataset, val_dataset])
# combined_dataloader = DataLoader(combined_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)

# alexnet = AlexNet()
# alexnet = alexnet.to(device)
# optimizer = torch.optim.SGD(alexnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
# loss_func = nn.CrossEntropyLoss()

# train_combined(alexnet, combined_dataloader, optimal_num_epochs, scheduler, loss_func, optimizer, device)

# checkpoint_path = "alexnet_checkpoints/best_model.pt"

# checkpoint = torch.load(checkpoint_path, map_location=device)
# alexnet.load_state_dict(checkpoint["model_state_dict"])
# alexnet.to(device)
# test_accuracy = test(alexnet, test_dataloader, device)
# print(f"Final Accuracy: {test_accuracy}", flush=True)