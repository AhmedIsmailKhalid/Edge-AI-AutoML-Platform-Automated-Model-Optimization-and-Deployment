"""
File operations utility functions.
"""

import hashlib
import shutil
from pathlib import Path

from fastapi import HTTPException, UploadFile, status


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    size_bytes = file_path.stat().st_size
    return size_bytes / (1024 * 1024)


def calculate_checksum(file_path: Path) -> str:
    """
    Calculate SHA256 checksum of a file.

    Args:
        file_path: Path to file

    Returns:
        SHA256 checksum as hex string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def validate_model_file(filename: str, framework: str) -> tuple[bool, str | None]:
    """
    Validate model file extension based on framework.

    Args:
        filename: Name of the uploaded file
        framework: 'pytorch' or 'tensorflow'

    Returns:
        Tuple of (is_valid, error_message)
    """
    file_ext = Path(filename).suffix.lower()

    if framework == "pytorch":
        valid_extensions = [".pt", ".pth", ".pkl"]  # Added .pt for TorchScript
        if file_ext not in valid_extensions:
            return (
                False,
                f"Invalid PyTorch model file. Expected {', '.join(valid_extensions)}, got {file_ext}",
            )

    elif framework == "tensorflow":
        valid_extensions = [".pb", ".h5", ".keras"]
        if file_ext not in valid_extensions:
            return (
                False,
                f"Invalid TensorFlow model file. Expected {', '.join(valid_extensions)}, got {file_ext}",
            )

    else:
        return False, f"Invalid framework: {framework}"

    return True, None


def generate_unique_filename(base_name: str, directory: Path, extension: str) -> str:
    """
    Generate a unique filename by appending a number if the file already exists.

    Args:
        base_name: Base name for the file (without extension)
        directory: Directory where file will be saved
        extension: File extension (with dot, e.g., '.pth')

    Returns:
        Unique filename

    Example:
        If 'my_model.pth' exists, returns 'my_model_2.pth'
    """
    filename = f"{base_name}{extension}"
    file_path = directory / filename

    if not file_path.exists():
        return filename

    counter = 2
    while True:
        filename = f"{base_name}_{counter}{extension}"
        file_path = directory / filename
        if not file_path.exists():
            return filename
        counter += 1


async def save_uploaded_file(
    upload_file: UploadFile, destination_dir: Path, custom_name: str | None = None
) -> tuple[Path, float]:
    """
    Save an uploaded file to the destination directory.

    Args:
        upload_file: FastAPI UploadFile object
        destination_dir: Directory to save the file
        custom_name: Optional custom name (without extension)

    Returns:
        Tuple of (file_path, file_size_mb)

    Raises:
        HTTPException: If file save fails
    """
    try:
        # Create destination directory if it doesn't exist
        destination_dir.mkdir(parents=True, exist_ok=True)

        # Determine filename
        original_name = Path(upload_file.filename).stem
        extension = Path(upload_file.filename).suffix

        if custom_name:
            base_name = custom_name
        else:
            base_name = original_name

        # Generate unique filename
        unique_filename = generate_unique_filename(base_name, destination_dir, extension)
        file_path = destination_dir / unique_filename

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        # Get file size
        file_size_mb = get_file_size_mb(file_path)

        return file_path, file_size_mb

    except Exception as e:
        raise HTTPException(  # noqa: B904
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )
    finally:
        upload_file.file.close()


def validate_dataset_structure(dataset_path: Path, framework: str) -> tuple[bool, str | None]:
    """
    Validate dataset directory structure.

    Args:
        dataset_path: Path to dataset directory
        framework: 'pytorch' or 'tensorflow'

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not dataset_path.exists():
        return False, "Dataset path does not exist"

    if not dataset_path.is_dir():
        return False, "Dataset path is not a directory"

    # For PyTorch: expect train/ and test/ or val/ subdirectories
    if framework == "pytorch":
        has_train = (dataset_path / "train").exists()
        has_test = (dataset_path / "test").exists() or (dataset_path / "val").exists()

        if not (has_train and has_test):
            return (
                False,
                "PyTorch dataset must contain 'train/' and 'test/' (or 'val/') directories",
            )

    # For TensorFlow: more flexible, just check it's not empty
    elif framework == "tensorflow":
        if not any(dataset_path.iterdir()):
            return False, "TensorFlow dataset directory is empty"

    return True, None
