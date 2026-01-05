"""
Test model architectures for development and testing.
These models are importable and can be used in tests.
"""

import torch.nn as nn


class CIFAR10TestModel(nn.Module):
    """Simple fully-connected model for CIFAR-10 (32x32x3 images)."""

    def __init__(self):
        super().__init__()
        # CIFAR-10 images are 32x32x3 = 3072 pixels
        self.fc1 = nn.Linear(3072, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)  # 10 classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Flatten the image
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class MNISTTestModel(nn.Module):
    """Simple fully-connected model for MNIST (28x28 images)."""

    def __init__(self):
        super().__init__()
        # MNIST images are 28x28 = 784 pixels
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # 10 classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Flatten the image
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
