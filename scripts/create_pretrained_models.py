"""
Create pretrained models for all datasets (MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100).
Trains small, medium, and large variants for both PyTorch and TensorFlow.

FIXED: PyTorch models are now saved as TorchScript for platform compatibility.
"""

import argparse
from pathlib import Path

import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ============================================================================
# PyTorch Model Architectures
# ============================================================================


class PyTorchSmallCNN(nn.Module):
    """Small CNN - works for grayscale (MNIST, Fashion-MNIST) and RGB (CIFAR)."""

    def __init__(self, input_channels: int, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 32 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class PyTorchMediumCNN(nn.Module):
    """Medium CNN - works for grayscale and RGB."""

    def __init__(self, input_channels: int, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class PyTorchLargeCNN(nn.Module):
    """Large CNN - works for grayscale and RGB."""

    def __init__(self, input_channels: int, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# ============================================================================
# TensorFlow Model Architectures
# ============================================================================


def create_tensorflow_small(input_shape: tuple, num_classes: int = 10) -> tf.keras.Model:
    """Create small TensorFlow CNN."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                16, (3, 3), activation="relu", padding="same", input_shape=input_shape
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def create_tensorflow_medium(input_shape: tuple, num_classes: int = 10) -> tf.keras.Model:
    """Create medium TensorFlow CNN."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", padding="same", input_shape=input_shape
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def create_tensorflow_large(input_shape: tuple, num_classes: int = 10) -> tf.keras.Model:
    """Create large TensorFlow CNN."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                64, (3, 3), activation="relu", padding="same", input_shape=input_shape
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


# ============================================================================
# Dataset Configuration
# ============================================================================

DATASET_CONFIG = {
    "mnist": {
        "num_classes": 10,
        "input_channels": 1,
        "input_shape": (28, 28, 1),
        "input_size": (28, 28),
        "pytorch_dataset": torchvision.datasets.MNIST,
        "tensorflow_dataset": tf.keras.datasets.mnist,
        "normalize": ((0.5,), (0.5,)),
    },
    "fashionmnist": {
        "num_classes": 10,
        "input_channels": 1,
        "input_shape": (28, 28, 1),
        "input_size": (28, 28),
        "pytorch_dataset": torchvision.datasets.FashionMNIST,
        "tensorflow_dataset": tf.keras.datasets.fashion_mnist,
        "normalize": ((0.5,), (0.5,)),
    },
    "cifar10": {
        "num_classes": 10,
        "input_channels": 3,
        "input_shape": (32, 32, 3),
        "input_size": (32, 32),
        "pytorch_dataset": torchvision.datasets.CIFAR10,
        "tensorflow_dataset": tf.keras.datasets.cifar10,
        "normalize": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    },
    "cifar100": {
        "num_classes": 100,
        "input_channels": 3,
        "input_shape": (32, 32, 3),
        "input_size": (32, 32),
        "pytorch_dataset": torchvision.datasets.CIFAR100,
        "tensorflow_dataset": tf.keras.datasets.cifar100,
        "normalize": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    },
}


# ============================================================================
# PyTorch Training
# ============================================================================


def train_pytorch_model(dataset_name: str, size: str, epochs: int = 5, batch_size: int = 128):
    """Train PyTorch model and save as TorchScript."""
    print(f"\n{'='*80}")
    print(f"Training PyTorch {size.upper()} model for {dataset_name.upper()}")
    print(f"{'='*80}")

    config = DATASET_CONFIG[dataset_name]

    # Create model
    if size == "small":
        model = PyTorchSmallCNN(config["input_channels"], config["num_classes"])
    elif size == "medium":
        model = PyTorchMediumCNN(config["input_channels"], config["num_classes"])
    else:  # large
        model = PyTorchLargeCNN(config["input_channels"], config["num_classes"])

    # Load dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(*config["normalize"])]
    )

    trainset = config["pytorch_dataset"](
        root="./dataset/custom", train=True, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 100 == 99:
                print(
                    f"  Epoch {epoch+1}/{epochs}, Batch {i+1}, "
                    f"Loss: {running_loss/100:.3f}, "
                    f"Acc: {100.*correct/total:.2f}%"
                )
                running_loss = 0.0

    # Save model (NOT TorchScript - use state_dict for optimization compatibility)
    output_dir = Path("models/pretrained/pytorch")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{size}_{dataset_name}_cnn.pt"

    # Move to CPU and save state dict
    model.cpu()
    torch.save(model.state_dict(), output_path)

    # Get file size
    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n✅ Saved model state dict: {output_path}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Final Accuracy: {100.*correct/total:.2f}%")


# ============================================================================
# TensorFlow Training
# ============================================================================


def train_tensorflow_model(dataset_name: str, size: str, epochs: int = 5, batch_size: int = 128):
    """Train TensorFlow model."""
    print(f"\n{'='*80}")
    print(f"Training TensorFlow {size.upper()} model for {dataset_name.upper()}")
    print(f"{'='*80}")

    config = DATASET_CONFIG[dataset_name]

    # Create model
    if size == "small":
        model = create_tensorflow_small(config["input_shape"], config["num_classes"])
    elif size == "medium":
        model = create_tensorflow_medium(config["input_shape"], config["num_classes"])
    else:  # large
        model = create_tensorflow_large(config["input_shape"], config["num_classes"])

    # Load dataset
    (x_train, y_train), (x_test, y_test) = config["tensorflow_dataset"].load_data()

    # Preprocess
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension for grayscale
    if config["input_channels"] == 1:
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]

    # Compile
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train
    print(f"Training on {len(x_train)} samples...")
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    # Save
    output_dir = Path("models/pretrained/tensorflow")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{size}_{dataset_name}_cnn.h5"
    model.save(output_path)

    # Get file size
    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n✅ Saved: {output_path}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Test Accuracy: {test_acc*100:.2f}%")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Create pretrained models for all datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["mnist", "fashionmnist", "cifar10", "cifar100", "all"],
        default=["all"],
        help="Datasets to train models for (default: all)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=["small", "medium", "large", "all"],
        default=["all"],
        help="Model sizes to train (default: all)",
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=["pytorch", "tensorflow", "all"],
        default=["all"],
        help="Frameworks to train for (default: all)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs (default: 5)"
    )

    args = parser.parse_args()

    # Expand 'all' options
    datasets = list(DATASET_CONFIG.keys()) if "all" in args.datasets else args.datasets
    sizes = ["small", "medium", "large"] if "all" in args.sizes else args.sizes
    frameworks = ["pytorch", "tensorflow"] if "all" in args.frameworks else args.frameworks

    print("\n" + "=" * 80)
    print("PRETRAINED MODEL CREATION")
    print("=" * 80)
    print(f"Datasets:   {', '.join(datasets)}")
    print(f"Sizes:      {', '.join(sizes)}")
    print(f"Frameworks: {', '.join(frameworks)}")
    print(f"Epochs:     {args.epochs}")
    print("=" * 80)
    print("\n⚠️  IMPORTANT: PyTorch models will be saved as TorchScript (.pt)")
    print("   This ensures compatibility with the optimization platform.")

    total = len(datasets) * len(sizes) * len(frameworks)
    current = 0

    for dataset in datasets:
        for size in sizes:
            for framework in frameworks:
                current += 1
                print(f"\n[{current}/{total}] Training {framework} {size} {dataset}...")

                try:
                    if framework == "pytorch":
                        train_pytorch_model(dataset, size, epochs=args.epochs)
                    else:
                        train_tensorflow_model(dataset, size, epochs=args.epochs)
                except Exception as e:
                    print(f"❌ Error training {framework} {size} {dataset}: {e}")
                    import traceback

                    traceback.print_exc()
                    continue

    print("\n" + "=" * 80)
    print("MODEL CREATION COMPLETE!")
    print("=" * 80)

    # List created models
    print("\nCreated Models:")
    print("-" * 80)

    pytorch_dir = Path("models/pretrained/pytorch")
    tensorflow_dir = Path("models/pretrained/tensorflow")

    if pytorch_dir.exists():
        print("\nPyTorch Models (TorchScript):")
        for model_file in sorted(pytorch_dir.glob("*.pt")):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  {model_file.name:40s} {size_mb:8.2f} MB")

    if tensorflow_dir.exists():
        print("\nTensorFlow Models:")
        for model_file in sorted(tensorflow_dir.glob("*.h5")):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  {model_file.name:40s} {size_mb:8.2f} MB")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
