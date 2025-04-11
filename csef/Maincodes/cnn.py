import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchinfo import summary

############################################
# 1. Define the CNN Model (Must Match Trained Model)
############################################
class CNN(nn.Module):
    def __init__(self, input_length):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Compute final feature length after conv1 and maxpool
        final_length = (input_length - 4) // 2
        if final_length <= 0:
            raise ValueError(f"Input sequence length {input_length} is too small for kernel=5 & pool=2.")

        self.fc1 = nn.Linear(32 * final_length, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid()
        return x

############################################
# 2. Load the Model Properly
############################################
def load_model(model_path, input_length):
    # Initialize model
    model = CNN(input_length)
    
    # Load weights correctly
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

############################################
# 3. Display Model Architecture
############################################
def display_model_summary(model, input_length):
    print("\n=== Model Summary ===")
    summary(model, input_size=(1, 1, input_length))

############################################
# 4. Visualize Filters (Conv1D Kernels)
############################################
def visualize_filters(model):
    conv1_weights = model.conv1.weight.data.cpu().numpy()  # Shape: (32, 1, 5)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))  # First 10 filters
    for i, ax in enumerate(axes.flat):
        if i < conv1_weights.shape[0]:
            ax.plot(conv1_weights[i, 0])  # Plot the 1D kernel
            ax.set_title(f"Filter {i+1}")
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()

############################################
# 5. Visualize Feature Maps (Layer Activations)
############################################
def visualize_feature_maps(model, input_length):
    X_test_sample = torch.randn(1, 1, input_length)  # Generate random test input

    with torch.no_grad():
        feature_maps = model.conv1(X_test_sample).squeeze().cpu().numpy()  # Shape: (32, new_length)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))  # First 10 feature maps
    for i, ax in enumerate(axes.flat):
        if i < feature_maps.shape[0]:
            ax.plot(feature_maps[i])  # Plot activation response
            ax.set_title(f"Feature Map {i+1}")
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()

############################################
# 6. Main Script Execution
############################################
if __name__ == "__main__":
    model_path = "CNN_Model.pth"  # Change if using a different model file
    csv_file = "synthetic_exoplanet_dataset_1000.csv"  # Change if needed
    
    # Determine input feature length from dataset
    df = pd.read_csv(csv_file)
    input_length = df.shape[1] - 1  # Assuming last column is label

    # Load Model
    model = load_model(model_path, input_length)

    # Show Model Summary
    display_model_summary(model, input_length)

    # Visualize Filters
    print("\n=== Visualizing Filters (Conv1D Kernels) ===")
    visualize_filters(model)

    # Visualize Feature Maps
    print("\n=== Visualizing Feature Maps ===")
    visualize_feature_maps(model, input_length)
