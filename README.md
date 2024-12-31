# VGG Neural Network from Scratch in PyTorch

This project demonstrates how to build a VGG-16 convolutional neural network (CNN) from scratch using PyTorch, train it on the CIFAR-10 dataset, and evaluate its performance. The project is designed for learning purposes and uses a smaller subset of the dataset for quick training on a CPU.

---

## Project Structure

```
vgg_project/
├── models/         # Contains the VGG-16 model definition
├── scripts/        # Training, evaluation, and prediction scripts
├── data/           # Contains the CIFAR-10 dataset (downloaded automatically)
├── outputs/        # Stores trained model files
├── examples/       # Contains example input images and outputs
└── README.md       # Documentation for the project
```

---

## Getting Started

### Prerequisites

- Python 3.8 or later
- PyTorch and torchvision

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/vgg_project.git
   cd vgg_project
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib
   ```

---

## Running the Project

### Training the Model

Run the following script to train the VGG-16 model on a subset of the CIFAR-10 dataset:

```bash
python scripts/train_vgg.py
```

- The trained model will be saved as `outputs/vgg16_cifar10.pth`.
- Training progress will be displayed in the terminal.

### Evaluating the Model

Use the following script to evaluate the trained model on the CIFAR-10 test set:

```bash
python scripts/evaluate_vgg.py
```

Expected Output:
```
Test Accuracy: ~20% (on a small subset)
```

### Making Predictions on New Images

You can use the trained model to predict the class of a new image. Ensure the image is placed in the `examples/` directory, e.g., `examples/test_image.jpg`. Run the following script:

```bash
python scripts/predict_vgg.py
```

- Replace `examples/test_image.jpg` with the path to your image.
- The script will output the predicted class for the image.

Example Output:
```
Predicted class: cat
```

---

## Features

- Implements the VGG-16 architecture from scratch.
- Trains on a small subset of the CIFAR-10 dataset for quick experimentation.
- Evaluates the trained model and outputs accuracy.
- Predicts classes for new images.

---

## Results

This project demonstrates how to:
- Build the VGG-16 network architecture from scratch.
- Train and evaluate a deep learning model on a real dataset.
- Save and load trained models.
- Predict the class of unseen images.

---
## Acknowledgments

- PyTorch for the deep learning framework.
- CIFAR-10 dataset for benchmarking.
