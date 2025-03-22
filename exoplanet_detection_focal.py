import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# Load the FOCAL dataset from Hugging Face
print("Loading FOCAL dataset...")
dataset = load_dataset('adsabs/FOCAL')

# Function to preprocess light curve data
def preprocess_light_curve(entry):
    time = np.array(entry['time'])
    flux = np.array(entry['flux'])
    # Normalize flux values
    flux = (flux - np.mean(flux)) / np.std(flux)
    return flux

# Extract and preprocess the data
labels = [entry['label'] for entry in dataset['train']]
light_curves = [preprocess_light_curve(entry) for entry in dataset['train']]

# Convert to NumPy arrays
light_curves = np.array(light_curves)
labels = np.array(labels)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(light_curves, labels, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define 1D CNN Model for Exoplanet Detection
class ExoplanetDetector(nn.Module):
    def __init__(self, input_length):
        super(ExoplanetDetector, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * (input_length // 4), 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * (x.shape[2] // 4))
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Define the model, loss function, and optimizer
input_length = X_train.shape[1]
model = ExoplanetDetector(input_length)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
print("Training the model...")
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), "exoplanet_detector.pth")
print("Model training complete and saved as 'exoplanet_detector.pth'.")

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_labels = (predictions > 0.5).float()
    accuracy = (predicted_labels == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Plot sample light curves with predictions
plt.figure(figsize=(12, 5))
plt.plot(X_test[0], label="Test Light Curve", alpha=0.7)
plt.title(f"Predicted: {'Exoplanet' if predicted_labels[0] == 1 else 'No Exoplanet'}")
plt.xlabel("Time")
plt.ylabel("Normalized Flux")
plt.legend()
plt.show()
