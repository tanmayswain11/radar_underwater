import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import os

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        data_path = "data/spectrograms/train"
        classes = sorted([
            c for c in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, c))
        ])

        num_classes = len(classes)
        print("✅ CNN Classes:", classes)

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)