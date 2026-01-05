"""
Create test datasets in ImageFolder format for testing custom dataset upload.

Creates zipped versions of CIFAR-10, MNIST, and Fashion-MNIST in the proper format:
dataset/custom/{dataset_name}/{dataset_name}.zip

The zip contains:
{dataset_name}/
├── class1/
│   ├── image1.png
│   └── ...
├── class2/
│   └── ...
"""

import shutil
from pathlib import Path

import torchvision

# import torchvision.transforms as transforms
# from PIL import Image


def create_cifar10_imagefolder():
    """Create CIFAR-10 in ImageFolder format and zip it."""
    print("📦 Creating CIFAR-10 ImageFolder dataset...")

    # Download CIFAR-10
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=None)

    # Class names
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # Create output directory: dataset/custom/cifar10_custom/
    dataset_name = "cifar10_custom"
    base_dir = Path("dataset/custom") / dataset_name
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create the actual dataset folder inside
    output_dir = base_dir / dataset_name
    output_dir.mkdir(exist_ok=True)

    # Create class directories
    for class_name in classes:
        (output_dir / class_name).mkdir(exist_ok=True)

    # Save images (use first 1000 images for quick testing)
    print("  Saving images...")
    for idx in range(min(1000, len(dataset))):
        img, label = dataset[idx]
        class_name = classes[label]

        # Save image
        img_path = output_dir / class_name / f"{idx:05d}.png"
        img.save(img_path)

        if (idx + 1) % 200 == 0:
            print(f"  Progress: {idx + 1}/1000 images")

    # Zip it - creates cifar10_custom.zip inside dataset/custom/cifar10_custom/
    print("  Zipping dataset...")
    zip_path = base_dir / dataset_name
    shutil.make_archive(str(zip_path), "zip", output_dir.parent, output_dir.name)

    # Clean up the unzipped folder
    shutil.rmtree(output_dir)

    print(f"✅ Created: {zip_path}.zip")

    return f"{zip_path}.zip"


def create_mnist_imagefolder():
    """Create MNIST in ImageFolder format and zip it."""
    print("📦 Creating MNIST ImageFolder dataset...")

    # Download MNIST
    dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=None)

    # Class names (digits 0-9)
    classes = [str(i) for i in range(10)]

    # Create output directory: dataset/custom/mnist_custom/
    dataset_name = "mnist_custom"
    base_dir = Path("dataset/custom") / dataset_name
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create the actual dataset folder inside
    output_dir = base_dir / dataset_name
    output_dir.mkdir(exist_ok=True)

    # Create class directories
    for class_name in classes:
        (output_dir / class_name).mkdir(exist_ok=True)

    # Save images (use first 1000 images)
    print("  Saving images...")
    for idx in range(min(1000, len(dataset))):
        img, label = dataset[idx]
        class_name = classes[label]

        # Save image
        img_path = output_dir / class_name / f"{idx:05d}.png"
        img.save(img_path)

        if (idx + 1) % 200 == 0:
            print(f"  Progress: {idx + 1}/1000 images")

    # Zip it
    print("  Zipping dataset...")
    zip_path = base_dir / dataset_name
    shutil.make_archive(str(zip_path), "zip", output_dir.parent, output_dir.name)

    # Clean up the unzipped folder
    shutil.rmtree(output_dir)

    print(f"✅ Created: {zip_path}.zip")

    return f"{zip_path}.zip"


def create_fashionmnist_imagefolder():
    """Create Fashion-MNIST in ImageFolder format and zip it."""
    print("📦 Creating Fashion-MNIST ImageFolder dataset...")

    # Download Fashion-MNIST
    dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=None
    )

    # Class names
    classes = [
        "tshirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle_boot",
    ]

    # Create output directory: dataset/custom/fashionmnist_custom/
    dataset_name = "fashionmnist_custom"
    base_dir = Path("dataset/custom") / dataset_name
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create the actual dataset folder inside
    output_dir = base_dir / dataset_name
    output_dir.mkdir(exist_ok=True)

    # Create class directories
    for class_name in classes:
        (output_dir / class_name).mkdir(exist_ok=True)

    # Save images (use first 1000 images)
    print("  Saving images...")
    for idx in range(min(1000, len(dataset))):
        img, label = dataset[idx]
        class_name = classes[label]

        # Save image
        img_path = output_dir / class_name / f"{idx:05d}.png"
        img.save(img_path)

        if (idx + 1) % 200 == 0:
            print(f"  Progress: {idx + 1}/1000 images")

    # Zip it
    print("  Zipping dataset...")
    zip_path = base_dir / dataset_name
    shutil.make_archive(str(zip_path), "zip", output_dir.parent, output_dir.name)

    # Clean up the unzipped folder
    shutil.rmtree(output_dir)

    print(f"✅ Created: {zip_path}.zip")

    return f"{zip_path}.zip"


def verify_imagefolder_structure(zip_path: str):
    """Verify the zip has correct ImageFolder structure."""
    import zipfile

    print(f"\n🔍 Verifying {Path(zip_path).name}...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        files = zf.namelist()

        # Get structure info
        class_dirs = set()
        image_count = 0

        for file in files:
            parts = Path(file).parts
            if len(parts) >= 2:
                class_dirs.add(parts[1])
            if len(parts) >= 3 and file.endswith(".png"):
                image_count += 1

        print(f"  Classes found: {len(class_dirs)}")
        print(f"  Class names: {sorted(class_dirs)}")
        print(f"  Total images: {image_count}")

        if len(class_dirs) > 0 and image_count > 0:
            print("  ✅ Structure looks good!")
        else:
            print("  ❌ Structure might be incorrect!")


def main():
    """Create all test datasets."""
    print("=" * 80)
    print("CREATING TEST DATASETS FOR CUSTOM UPLOAD")
    print("=" * 80)
    print()

    # Create datasets
    zip_files = []

    try:
        zip_files.append(create_cifar10_imagefolder())
        print()
        zip_files.append(create_mnist_imagefolder())
        print()
        zip_files.append(create_fashionmnist_imagefolder())
        print()

    except Exception as e:
        print(f"\n❌ Error creating datasets: {e}")
        raise

    # Verify all
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    for zip_file in zip_files:
        verify_imagefolder_structure(zip_file)

    # Summary
    print("\n" + "=" * 80)
    print("✅ DONE!")
    print("=" * 80)
    print("\nTest datasets created:")
    for zip_file in zip_files:
        print(f"  • {zip_file}")
    print("\nYou can now upload these zip files to test custom dataset functionality!")
    print()


if __name__ == "__main__":
    main()
