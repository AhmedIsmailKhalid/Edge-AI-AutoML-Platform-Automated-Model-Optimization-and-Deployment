"""
Dataset loading utilities for preset and custom datasets.
"""

from pathlib import Path
from typing import Any

import tensorflow as tf
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.config import settings


def load_pytorch_preset_dataset(
    dataset_name: str, batch_size: int = 32, num_workers: int = 0
) -> tuple[DataLoader, DataLoader]:
    """
    Load a preset PyTorch dataset.

    Args:
        dataset_name: Name of dataset ('cifar10', 'cifar100', 'mnist', 'fashionmnist')
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Base path
    base_path = settings.preset_dataset_path
    base_path.mkdir(parents=True, exist_ok=True)

    # For CIFAR datasets, create an extra parent folder
    # For MNIST/FashionMNIST, use base_path directly
    if dataset_name in ["cifar10", "cifar100"]:
        # Create CIFAR10/ or CIFAR100/ folder
        dataset_root = base_path / dataset_name.upper()
        dataset_root.mkdir(parents=True, exist_ok=True)
    else:
        # MNIST and FashionMNIST use base_path directly
        dataset_root = base_path

    # Define transforms
    if dataset_name in ["cifar10", "cifar100"]:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    else:  # mnist, fashionmnist
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    # Load datasets
    if dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=str(dataset_root), train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=str(dataset_root), train=False, download=True, transform=transform
        )

    elif dataset_name == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=str(dataset_root), train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=str(dataset_root), train=False, download=True, transform=transform
        )

    elif dataset_name == "mnist":
        train_dataset = torchvision.datasets.MNIST(
            root=str(dataset_root), train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=str(dataset_root), train=False, download=True, transform=transform
        )

    elif dataset_name == "fashionmnist":
        train_dataset = torchvision.datasets.FashionMNIST(
            root=str(dataset_root), train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=str(dataset_root), train=False, download=True, transform=transform
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def load_tensorflow_preset_dataset(
    dataset_name: str, batch_size: int = 32
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Load a preset TensorFlow dataset.

    Args:
        dataset_name: Name of dataset ('cifar10', 'cifar100', 'mnist', 'fashionmnist')
        batch_size: Batch size

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # TensorFlow datasets are built-in, no need to download to specific path

    if dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif dataset_name == "cifar100":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    elif dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # Add channel dimension for grayscale
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]
    elif dataset_name == "fashionmnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        # Add channel dimension for grayscale
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Create tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset


def load_dataset(
    framework: str,
    dataset_type: str,
    dataset_name: str,
    dataset_path: str | None = None,
    batch_size: int = 32,
) -> Any:
    """
    Load dataset based on framework and type.

    Args:
        framework: 'pytorch' or 'tensorflow'
        dataset_type: 'preset' or 'custom'
        dataset_name: Name of the dataset
        dataset_path: Path to custom dataset (if custom)
        batch_size: Batch size

    Returns:
        Dataset loaders (test/validation set for evaluation)
    """
    if dataset_type == "preset":
        if framework == "pytorch":
            _, test_loader = load_pytorch_preset_dataset(dataset_name, batch_size)
            return test_loader
        else:  # tensorflow
            _, test_dataset = load_tensorflow_preset_dataset(dataset_name, batch_size)
            return test_dataset
    elif dataset_type == "custom":
        if not dataset_path:
            raise ValueError("dataset_path is required for custom datasets")

        if framework == "pytorch":
            return load_pytorch_custom_dataset(dataset_path, batch_size)
        else:  # tensorflow
            return load_tensorflow_custom_dataset(dataset_path, batch_size)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


def load_pytorch_custom_dataset(dataset_path: str, batch_size: int = 32) -> DataLoader:
    """
    Load custom PyTorch dataset from ImageFolder format.

    Args:
        dataset_path: Path to dataset root (contains class directories)
        batch_size: Batch size

    Returns:
        DataLoader for the custom dataset
    """

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Detect dataset characteristics by reading actual images
    is_grayscale = _detect_grayscale_dataset(dataset_path)
    image_size = _detect_actual_image_size(dataset_path)

    if is_grayscale:
        print(f"📊 Detected grayscale dataset (actual size: {image_size}x{image_size})")
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
    else:
        print(f"📊 Detected RGB dataset (actual size: {image_size}x{image_size})")
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    dataset = ImageFolder(root=str(dataset_path), transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    print(
        f"✅ Loaded PyTorch custom dataset: {len(dataset)} images, {len(dataset.classes)} classes"
    )

    return dataloader


def load_tensorflow_custom_dataset(dataset_path: str, batch_size: int = 32) -> tf.data.Dataset:
    """
    Load custom TensorFlow dataset from ImageFolder format.

    Args:
        dataset_path: Path to dataset root (contains class directories)
        batch_size: Batch size

    Returns:
        tf.data.Dataset for the custom dataset
    """

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Detect dataset characteristics by reading actual images
    is_grayscale = _detect_grayscale_dataset(dataset_path)
    image_size = _detect_actual_image_size(dataset_path)

    if is_grayscale:
        print(f"📊 Detected grayscale TensorFlow dataset (actual size: {image_size}x{image_size})")
        color_mode = "grayscale"
    else:
        print(f"📊 Detected RGB TensorFlow dataset (actual size: {image_size}x{image_size})")
        color_mode = "rgb"

    dataset = tf.keras.utils.image_dataset_from_directory(
        str(dataset_path),
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        color_mode=color_mode,
    )

    # Normalize images to [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    print(f"✅ Loaded TensorFlow custom dataset from {dataset_path}")

    return dataset


def _detect_grayscale_dataset(dataset_path: Path) -> bool:
    """
    Detect if dataset is grayscale by checking the first image.

    Args:
        dataset_path: Path to dataset root

    Returns:
        True if grayscale, False if RGB
    """

    # Find first image file
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    # Open image and check mode
                    img = Image.open(img_file)

                    # L = grayscale, RGB = color, RGBA = color with alpha
                    if img.mode == "L":
                        return True
                    elif img.mode in ["RGB", "RGBA"]:
                        return False

                    # If found an image, stop
                    break
            break

    # Default to RGB if can't determine
    return False


def _detect_actual_image_size(dataset_path: Path) -> int:
    """
    Detect the actual image size in the dataset by reading the first image.

    Args:
        dataset_path: Path to dataset root

    Returns:
        Image size (width/height, assumes square images)
    """

    # Find first image file
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    # Open image and get size
                    img = Image.open(img_file)
                    width, height = img.size

                    # Return the smaller dimension (in case not square)
                    # This ensures we don't upscale if images are rectangular
                    detected_size = min(width, height)

                    print(
                        f"🔍 Detected image dimensions: {width}x{height}, using {detected_size}x{detected_size}"
                    )

                    return detected_size

    # Default to 32 if can't detect
    print("⚠️  Could not detect image size, defaulting to 32x32")
    return 32
