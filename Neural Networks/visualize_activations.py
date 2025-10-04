# This script is used to visualize how activation functions try to approximate the target function
# ReLU can provide a good approximation with 4 hidden units
# Sin can approximate the function with 6 hidden units
# Sigmoid however requires 8 hidden units to show meaningful results

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

# Generating data
x = np.linspace(-2, 2, 100).reshape(-1, 1)
y_true = x**2 + 4    # np.sin(x) + np.cos(x), x**3, 

X_tensor = torch.FloatTensor(x)
y_tensor = torch.FloatTensor(y_true)

class ActivationNet(nn.Module):
    def __init__(self, hidden_units=10, activation='relu'):
        super().__init__()
        self.hidden = nn.Linear(1, hidden_units)
        self.output = nn.Linear(hidden_units, 1)
        self.activation = activation
        
    def forward(self, x):
        x = self.hidden(x)    # W1.T @ X + B1

        # activation_function(W1.T @ X + B1)
        if self.activation == 'relu':
            x = torch.relu(x)
        elif self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.activation == 'sin':
            x = torch.sin(x)

        return self.output(x)    # W2.T @ (activation_function(W1.T @ X + B1)) + B2

activations = ['relu', 'sigmoid', 'sin']
models = {}
predictions = {}
neuron_activations = {}
losses = {}
r2_scores = {}

hidden_units = 8

for activation in activations:
    print(f"Using {activation}")
    model = ActivationNet(hidden_units, activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train the network
    epoch_losses = []
    for epoch in range(5000):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = nn.MSELoss()(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    
    # Store results
    with torch.no_grad():
        models[activation] = model
        y_pred_np = model(X_tensor).numpy()
        predictions[activation] = y_pred_np

        losses[activation] = epoch_losses[-1]
        r2_scores[activation] = r2_score(y_true, y_pred_np)
        
        # Extracting the activation_function(W1.T @ X + B1)
        hidden_output = model.hidden(X_tensor)
        if activation == 'relu':
            neuron_activations[activation] = torch.relu(hidden_output).numpy()
        elif activation == 'sigmoid':
            neuron_activations[activation] = torch.sigmoid(hidden_output).numpy()
        elif activation == 'sin':
            neuron_activations[activation] = torch.sin(hidden_output).numpy()

fig, axes = plt.subplots(2, 3, figsize=(20, 8))

for i, activation in enumerate(activations):
    ax = axes[0, i]
    ax.plot(x, y_true, 'k-', label=f'True', linewidth=3, alpha=0.8)
    ax.plot(x, predictions[activation], 'r--', label=f'{activation.upper()} Approximation', linewidth=2)
    
    loss_val = losses[activation]
    r2_val = r2_scores[activation]
    ax.set_title(f'{activation.upper()} Approximation\nMSE: {loss_val:.4f}, R²: {r2_val:.4f}', fontsize=12)
    ax.legend()
    ax.grid(True)

for i, activation in enumerate(activations):
    ax = axes[1, i]
    activations_data = neuron_activations[activation]
    for neuron_idx in range(min(6, activations_data.shape[1])):
        ax.plot(x, activations_data[:, neuron_idx], 
               alpha=0.7, label=f'Neuron {neuron_idx+1}', linewidth=2)
    ax.set_title(f'{activation.upper()} Neuron Activations')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()