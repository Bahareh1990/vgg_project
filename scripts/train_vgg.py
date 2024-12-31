import sys
import os

# Add the parent directory (vgg_project) to Python's search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from models.vgg import vgg16

# Hyperparameters
num_classes = 10
learning_rate = 0.001
batch_size = 64  # Increase batch size for faster processing
num_epochs = 2  # Reduce epochs for quicker training

# Transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Smaller image size for quicker training,
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset and DataLoader (subset of CIFAR-10)
print("Loading datasets...")
train_dataset = datasets.CIFAR10(root="data", train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root="data", train=False, transform=transform, download=True)


# Subset the CIFAR-10 dataset to reduce size
subset_size = 1000  # Limit to 1000 samples for training
train_subset, _ = torch.utils.data.random_split(train_dataset, [subset_size, len(train_dataset) - subset_size])
train_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)


# Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vgg16(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
print("Starting training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 10 == 0:  # Print progress every 10 batches
               print(f"Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}")
               running_loss = 0.0

# Save the model
if not os.path.exists("outputs"):
       os.makedirs("outputs")
torch.save(model.state_dict(), "outputs/vgg16_cifar10_small.pth")
print("Training complete. Model saved to outputs/vgg16_cifar10.pth")
