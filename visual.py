import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Load MNIST dataset
transform = transforms.ToTensor()
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Plot sample images
plt.figure(figsize=(10, 10))
for i in range(16):
    image, label = mnist_train[i]
    plt.subplot(4, 4, i + 1)
    plt.imshow(image.squeeze(), cmap='gray')  # Squeeze to remove channel dimension
    plt.title(f"Label: {label}")
    plt.axis('off')
plt.tight_layout()
plt.show()
