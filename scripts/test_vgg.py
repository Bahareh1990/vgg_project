import sys
import os

# Add the parent directory (vgg_project) to Python's search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.vgg import vgg16

# Instantiate the VGG model
model = vgg16(num_classes=1000)
print(model)

# Test the model with random input
x = torch.randn(1, 3, 224, 224)  # Batch size: 1, Channels: 3, Image size: 224x224
output = model(x)
print("Output shape:", output.shape)
