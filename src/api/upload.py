"""
Model and dataset upload API endpoints.
"""

import logging
import shutil
from pathlib import Path
from uuid import UUID

from fastapi import (APIRouter, Depends, File, Form, HTTPException, UploadFile,
                     status)
from sqlalchemy.orm import Session

from src.config import settings
from src.database import get_db
from src.models.experiment import Experiment
from src.models.model_file import ModelFile
from src.schemas.upload import ModelUploadResponse
from src.utils.file_handler import (calculate_checksum, save_uploaded_file,
                                    validate_model_file)

router = APIRouter()

logger = logging.getLogger(__name__)


@router.post(
    "/{experiment_id}/model",
    response_model=ModelUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_model(
    experiment_id: UUID,
    file: UploadFile = File(None),  # Optional - can be None for pretrained
    model_source: str = Form(...),  # 'pretrained' or 'custom'
    pretrained_model_name: str = Form(None),  # Required if model_source='pretrained'
    db: Session = Depends(get_db),
) -> ModelUploadResponse:
    """
    Upload a model file for an experiment.

    Supports two modes:
    1. Custom Model: User uploads their own model file
    2. Pretrained Model: System loads a pretrained model by name

    Args:
        experiment_id: Experiment UUID
        file: Uploaded model file (required for custom, ignored for pretrained)
        model_source: 'pretrained' or 'custom'
        pretrained_model_name: Name of pretrained model (e.g., 'small_mnist_cnn')
        db: Database session

    Returns:
        Upload confirmation with file details

    Raises:
        HTTPException: If experiment not found, validation fails, or pretrained model not found
    """
    # Get experiment
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id {experiment_id} not found",
        )

    if model_source == "pretrained":
        # Handle pretrained model loading
        if not pretrained_model_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="pretrained_model_name is required when model_source='pretrained'",
            )

        # Build pretrained model path - try both .pt and .pth extensions
        framework = experiment.framework

        if framework == "tensorflow":
            extensions = [".h5"]
        else:  # pytorch
            extensions = [".pth", ".pt"]

        # Try to find the model with any valid extension
        pretrained_path = None
        for ext in extensions:
            model_filename = f"{pretrained_model_name}{ext}"
            test_path = settings.pretrained_models_path / framework / model_filename

            if test_path.exists():
                pretrained_path = test_path
                break

        # Check if pretrained model exists
        if not pretrained_path:
            # Show which extensions were tried
            tried_extensions = ", ".join(extensions)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pretrained model '{pretrained_model_name}' not found for {framework}. "
                f"Tried extensions: {tried_extensions}. "
                f"Expected location: {settings.pretrained_models_path / framework}/",
            )

        # Copy pretrained model to experiment's custom directory
        # This ensures each experiment has its own copy for optimization
        destination_dir = settings.custom_models_path / framework / str(experiment_id)
        destination_dir.mkdir(parents=True, exist_ok=True)

        # Keep original extension
        destination_path = destination_dir / pretrained_path.name

        logger.info(f"Copying pretrained model from {pretrained_path} to {destination_path}")
        shutil.copy2(pretrained_path, destination_path)

        file_path = destination_path
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        model_name = pretrained_model_name

    elif model_source == "custom":
        # Handle custom model upload
        if not file:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="file is required when model_source='custom'",
            )

        # Validate model file
        is_valid, error_msg = validate_model_file(file.filename, experiment.framework)
        if not is_valid:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)

        # Infer model name from filename (without extension)
        model_name = Path(file.filename).stem

        # Save to custom directory
        destination_dir = settings.custom_models_path / experiment.framework / str(experiment_id)
        file_path, file_size_mb = await save_uploaded_file(
            upload_file=file, destination_dir=destination_dir, custom_name=None
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model_source: {model_source}. Must be 'pretrained' or 'custom'",
        )

    # Calculate checksum
    checksum = calculate_checksum(file_path)

    # Determine file format
    extension = file_path.suffix.lower()
    if experiment.framework == "pytorch":
        file_format = f"pytorch_{extension[1:]}"  # e.g., 'pytorch_pth'
    else:
        file_format = f"tensorflow_{extension[1:]}"  # e.g., 'tensorflow_h5'

    # Create model file record
    model_file = ModelFile(
        experiment_id=experiment_id,
        file_type="original",
        file_format=file_format,
        file_path=str(file_path),
        file_size_mb=file_size_mb,
        checksum=checksum,
    )

    db.add(model_file)

    # Update experiment with model name
    experiment.model_name = model_name

    db.commit()
    db.refresh(model_file)

    logger.info(
        f"Model uploaded successfully for experiment {experiment_id}: "
        f"{model_name} ({file_size_mb:.2f} MB)"
    )

    return ModelUploadResponse(
        experiment_id=experiment_id,
        model_file_id=model_file.id,
        model_name=model_name,
        framework=experiment.framework,
        file_size_mb=file_size_mb,
        file_path=str(file_path),
        message=f"{'Pretrained' if model_source == 'pretrained' else 'Custom'} model uploaded successfully",
    )


@router.post("/{experiment_id}/dataset")
async def upload_dataset(
    experiment_id: UUID,
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
    db: Session = Depends(get_db),
) -> dict:
    """
    Upload a custom dataset zip file.

    Expected structure inside zip:
    dataset_name/
    ├── class1/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── class2/
    │   └── ...

    Args:
        experiment_id: Experiment UUID
        file: Zip file containing dataset in ImageFolder format
        dataset_name: Name of the dataset
        db: Database session

    Returns:
        Upload confirmation with dataset info
    """
    import tempfile
    import zipfile
    from pathlib import Path

    # Get experiment
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    # Validate file is a zip
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a .zip file")

    # Validate dataset name (alphanumeric and underscores only)
    if not dataset_name.replace("_", "").isalnum():
        raise HTTPException(
            status_code=400,
            detail="Dataset name must contain only letters, numbers, and underscores",
        )

    dataset_dir = None  # Initialize for cleanup

    try:
        # SMART NAMING STRATEGY
        # 1. Check if this is a test dataset (cifar10_custom, mnist_custom, fashionmnist_custom)
        test_datasets = ["cifar10_custom", "mnist_custom", "fashionmnist_custom"]

        if dataset_name in test_datasets:
            # Test datasets: reuse if exists
            dataset_dir = settings.custom_datasets_path / dataset_name

            if dataset_dir.exists():
                # Dataset directory exists - check if it has valid extracted structure
                logger.info(f"Checking existing test dataset: {dataset_name}")

                # Validate ImageFolder structure (look for class directories)
                class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

                if len(class_dirs) > 0:
                    # Valid structure exists - count images and return
                    logger.info(
                        f"Found valid extracted dataset with {len(class_dirs)} classes. Reusing..."
                    )

                    total_images = 0
                    class_info = {}

                    for class_dir in class_dirs:
                        images = (
                            list(class_dir.glob("*.jpg"))
                            + list(class_dir.glob("*.jpeg"))
                            + list(class_dir.glob("*.png"))
                            + list(class_dir.glob("*.JPG"))
                            + list(class_dir.glob("*.JPEG"))
                            + list(class_dir.glob("*.PNG"))
                        )

                        class_info[class_dir.name] = len(images)
                        total_images += len(images)

                    if total_images > 0:
                        # Valid dataset with images - use it
                        experiment.dataset_name = dataset_name
                        experiment.dataset_type = "custom"
                        experiment.dataset_path = str(dataset_dir)
                        db.commit()
                        db.flush()
                        db.refresh(experiment)

                        logger.info(
                            f"Reused test dataset: {dataset_name} ({total_images} images, {len(class_dirs)} classes)"
                        )

                        return {
                            "message": f"Using existing test dataset: {dataset_name}",
                            "dataset_name": dataset_name,
                            "dataset_path": str(dataset_dir),
                            "num_classes": len(class_dirs),
                            "classes": list(class_info.keys()),
                            "total_images": total_images,
                            "images_per_class": class_info,
                            "reused": True,
                        }

                # No valid structure found - need to extract
                logger.info("No valid extracted structure found. Will extract from zip...")
            else:
                # Directory doesn't exist - create it
                dataset_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created new directory for test dataset: {dataset_name}")

        # 2. For custom user datasets: auto-increment if exists
        base_dataset_name = dataset_name
        dataset_dir = settings.custom_datasets_path / dataset_name
        counter = 2

        # Find unique name by appending _2, _3, etc.
        while dataset_dir.exists():
            dataset_name = f"{base_dataset_name}_{counter}"
            dataset_dir = settings.custom_datasets_path / dataset_name
            counter += 1

        if dataset_name != base_dataset_name:
            logger.info(
                f"Dataset name changed from '{base_dataset_name}' to '{dataset_name}' to avoid conflict"
            )

        # Create dataset directory
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
            tmp_path = Path(tmp_file.name)
            contents = await file.read()
            tmp_file.write(contents)

        # Extract zip file to dataset_dir
        logger.info(f"Extracting dataset to {dataset_dir}")

        with zipfile.ZipFile(tmp_path, "r") as zip_ref:
            # Extract everything to dataset_dir
            zip_ref.extractall(dataset_dir)

        # Clean up temp zip file
        tmp_path.unlink()

        # Check if we have a nested directory structure
        items_in_dir = list(dataset_dir.iterdir())

        if len(items_in_dir) == 1 and items_in_dir[0].is_dir():
            # There's a single subdirectory - move its contents up one level
            nested_dir = items_in_dir[0]

            logger.info(f"Found nested directory: {nested_dir.name}. Moving contents up...")

            # Move all items from nested directory to parent
            for item in nested_dir.iterdir():
                dest = dataset_dir / item.name
                shutil.move(str(item), str(dest))

            # Remove the now-empty nested directory
            nested_dir.rmdir()

            logger.info("Flattened nested structure")

        # Validate ImageFolder structure
        class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

        if len(class_dirs) == 0:
            # Cleanup and error
            shutil.rmtree(dataset_dir)
            raise HTTPException(
                status_code=400, detail="Invalid dataset structure. No class directories found."
            )

        # Count images in each class
        total_images = 0
        class_info = {}

        for class_dir in class_dirs:
            images = (
                list(class_dir.glob("*.jpg"))
                + list(class_dir.glob("*.jpeg"))
                + list(class_dir.glob("*.png"))
                + list(class_dir.glob("*.JPG"))
                + list(class_dir.glob("*.JPEG"))
                + list(class_dir.glob("*.PNG"))
            )

            class_info[class_dir.name] = len(images)
            total_images += len(images)

        if total_images == 0:
            # Cleanup and error
            shutil.rmtree(dataset_dir)
            raise HTTPException(
                status_code=400, detail="Invalid dataset. No images found in class directories."
            )

        # Update experiment with dataset info
        experiment.dataset_name = dataset_name
        experiment.dataset_type = "custom"
        experiment.dataset_path = str(dataset_dir)
        db.commit()
        db.flush()
        db.refresh(experiment)

        logger.info(
            f"Dataset uploaded successfully: {dataset_name} ({total_images} images, {len(class_dirs)} classes)"
        )

        return {
            "message": "Dataset uploaded successfully",
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_dir),
            "num_classes": len(class_dirs),
            "classes": list(class_info.keys()),
            "total_images": total_images,
            "images_per_class": class_info,
            "reused": False,
        }

    except HTTPException:
        raise
    except Exception as e:
        # Cleanup on error (only for non-test datasets)
        if dataset_dir and dataset_dir.exists() and dataset_name not in test_datasets:
            shutil.rmtree(dataset_dir)

        logger.error(f"Error uploading dataset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading dataset: {str(e)}") from e
