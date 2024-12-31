import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, architecture, num_classes=1000, input_size=64):
        super(VGG, self).__init__()
        self.in_channels = 3
        self.conv_layers = self.create_conv_layers(architecture)
        
        # Dynamically compute the size of the flattened feature map
        test_tensor = torch.zeros(1, 3, input_size, input_size)
        flattened_size = self._get_flattened_size(test_tensor)
        
        self.fcs = nn.Sequential(
            nn.Linear(flattened_size, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == int:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, stride=1, padding=1), nn.ReLU()]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)
    
    def _get_flattened_size(self, x):
        x = self.conv_layers(x)
        return x.view(x.shape[0], -1).shape[1]

VGG16_ARCHITECTURE = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

def vgg16(num_classes=1000, input_size=64):
    return VGG(VGG16_ARCHITECTURE, num_classes=num_classes, input_size=input_size)

