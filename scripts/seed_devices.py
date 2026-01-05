"""
Seed script to create necessary directories and download preset models (optional).
"""

import torch
import torchvision.models as models

from src.config import settings


def create_directories():
    """Create all necessary directories."""
    print("Creating directory structure...")

    # Dataset directories
    settings.preset_dataset_path.mkdir(parents=True, exist_ok=True)
    (settings.custom_dataset_path / "pytorch").mkdir(parents=True, exist_ok=True)
    (settings.custom_dataset_path / "tensorflow").mkdir(parents=True, exist_ok=True)

    # Model directories
    (settings.pretrained_models_path / "pytorch").mkdir(parents=True, exist_ok=True)
    (settings.pretrained_models_path / "tensorflow").mkdir(parents=True, exist_ok=True)
    (settings.custom_models_path / "pytorch").mkdir(parents=True, exist_ok=True)
    (settings.custom_models_path / "tensorflow").mkdir(parents=True, exist_ok=True)
    settings.optimized_models_path.mkdir(parents=True, exist_ok=True)

    print("✅ Directory structure created")


def download_pretrained_models():
    """Download pretrained PyTorch models."""
    print("\nDownloading pretrained models...")

    pytorch_models_dir = settings.pretrained_models_path / "pytorch"

    models_to_download = {
        "resnet18": models.resnet18,
        "vgg16": models.vgg16,
        "mobilenet": models.mobilenet_v2,
    }

    for model_name, model_fn in models_to_download.items():
        model_path = pytorch_models_dir / f"{model_name}.pth"

        if model_path.exists():
            print(f"   {model_name} already exists, skipping")
            continue

        print(f"  Downloading {model_name}...")
        model = model_fn(pretrained=True)
        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved {model_name} to {model_path}")

    print("\n✅ All pretrained models downloaded")


if __name__ == "__main__":
    print("=" * 60)
    print("SEEDING PLATFORM")
    print("=" * 60)

    create_directories()

    # Optional: download pretrained models
    download_choice = input("\nDownload pretrained models? (y/n): ").lower()
    if download_choice == "y":
        download_pretrained_models()

    print("\n" + "=" * 60)
    print("✅ SEEDING COMPLETE")
    print("=" * 60)
