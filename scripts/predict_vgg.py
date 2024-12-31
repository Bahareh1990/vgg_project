import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import torch
from torchvision import transforms

# Add the parent directory (vgg_project) to Python's search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vgg import vgg16

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vgg16(num_classes=10).to(device)
model.load_state_dict(torch.load("outputs/vgg16_cifar10.pth"))
model.eval()

# Define a transformation for the input image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image_path):
    # Open and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Make prediction
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

    # Map prediction to CIFAR-10 classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return classes[predicted.item()]

if __name__ == "__main__":
    # Update this path to the actual image in your examples directory
    image_path = "examples/2.jpg"
    try:
        result = predict(image_path)
        print(f"Predicted class: {result}")
    except FileNotFoundError as e:
        print(e)

