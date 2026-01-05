"""
Centralized error handling and validation utilities.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    """Standardized error codes."""

    EXPERIMENT_NOT_FOUND = "EXPERIMENT_NOT_FOUND"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    DATASET_NOT_FOUND = "DATASET_NOT_FOUND"
    INVALID_MODEL_FILE = "INVALID_MODEL_FILE"
    INVALID_DATASET_FILE = "INVALID_DATASET_FILE"
    CORRUPTED_MODEL = "CORRUPTED_MODEL"
    CORRUPTED_DATASET = "CORRUPTED_DATASET"
    OPTIMIZATION_FAILED = "OPTIMIZATION_FAILED"
    INSUFFICIENT_MEMORY = "INSUFFICIENT_MEMORY"
    UNSUPPORTED_FRAMEWORK = "UNSUPPORTED_FRAMEWORK"
    UNSUPPORTED_DEVICE = "UNSUPPORTED_DEVICE"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"


class ValidationError(Exception):
    """Custom validation error."""

    def __init__(self, message: str, code: ErrorCode, details: dict | None = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


def validate_file_exists(file_path: Path, file_type: str = "File") -> None:
    """
    Validate that a file exists.

    Args:
        file_path: Path to file
        file_type: Type of file for error message

    Raises:
        ValidationError: If file doesn't exist
    """
    if not file_path.exists():
        raise ValidationError(
            message=f"{file_type} not found: {file_path}",
            code=ErrorCode.MODEL_NOT_FOUND
            if "model" in file_type.lower()
            else ErrorCode.DATASET_NOT_FOUND,
            details={"file_path": str(file_path)},
        )


def validate_file_size(file_path: Path, max_size_mb: float = 500) -> None:
    """
    Validate file size.

    Args:
        file_path: Path to file
        max_size_mb: Maximum allowed size in MB

    Raises:
        ValidationError: If file is too large
    """
    if not file_path.exists():
        return

    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValidationError(
            message=f"File too large: {size_mb:.2f} MB (max: {max_size_mb} MB)",
            code=ErrorCode.FILE_TOO_LARGE,
            details={"size_mb": size_mb, "max_size_mb": max_size_mb},
        )


def validate_framework(framework: str) -> None:
    """
    Validate framework is supported.

    Args:
        framework: Framework name

    Raises:
        ValidationError: If framework not supported
    """
    valid_frameworks = ["pytorch", "tensorflow"]
    if framework.lower() not in valid_frameworks:
        raise ValidationError(
            message=f"Unsupported framework: {framework}",
            code=ErrorCode.UNSUPPORTED_FRAMEWORK,
            details={"framework": framework, "valid_frameworks": valid_frameworks},
        )


def validate_device(device: str) -> None:
    """
    Validate device is supported.

    Args:
        device: Device identifier

    Raises:
        ValidationError: If device not supported
    """
    valid_devices = [
        "raspberry_pi_3b",
        "raspberry_pi_4",
        "raspberry_pi_5",
        "jetson_nano",
        "jetson_xavier_nx",
        "coral_dev_board",
    ]
    if device.lower() not in valid_devices:
        raise ValidationError(
            message=f"Unsupported device: {device}",
            code=ErrorCode.UNSUPPORTED_DEVICE,
            details={"device": device, "valid_devices": valid_devices},
        )


def validate_model_file(file_path: Path, framework: str) -> None:
    """
    Validate model file format and integrity.

    Args:
        file_path: Path to model file
        framework: Expected framework

    Raises:
        ValidationError: If validation fails
    """
    validate_file_exists(file_path, "Model file")

    if framework == "pytorch":
        valid_extensions = [".pt", ".pth"]
        if file_path.suffix.lower() not in valid_extensions:
            raise ValidationError(
                message=f"Invalid PyTorch model file extension: {file_path.suffix}",
                code=ErrorCode.INVALID_MODEL_FILE,
                details={"extension": file_path.suffix, "valid_extensions": valid_extensions},
            )

        try:
            import torch

            torch.load(file_path, map_location="cpu")
        except Exception as e:
            raise ValidationError(  # noqa: B904
                message=f"Corrupted PyTorch model file: {str(e)}",
                code=ErrorCode.CORRUPTED_MODEL,
                details={"error": str(e)},
            )

    elif framework == "tensorflow":
        valid_extensions = [".h5", ".keras"]
        if file_path.suffix.lower() not in valid_extensions:
            raise ValidationError(
                message=f"Invalid TensorFlow model file extension: {file_path.suffix}",
                code=ErrorCode.INVALID_MODEL_FILE,
                details={"extension": file_path.suffix, "valid_extensions": valid_extensions},
            )

        try:
            import tensorflow as tf

            tf.keras.models.load_model(file_path, compile=False)
        except Exception as e:
            raise ValidationError(  # noqa: B904
                message=f"Corrupted TensorFlow model file: {str(e)}",
                code=ErrorCode.CORRUPTED_MODEL,
                details={"error": str(e)},
            )


def validate_constraints(
    accuracy_drop: float | None,
    max_size: float | None,
    max_latency: float | None,
) -> None:
    """
    Validate optimization constraints.

    Args:
        accuracy_drop: Max accuracy drop percentage
        max_size: Max model size in MB
        max_latency: Max latency in ms

    Raises:
        ValidationError: If constraints are invalid
    """
    if accuracy_drop is not None:
        if accuracy_drop < 0 or accuracy_drop > 100:
            raise ValidationError(
                message=f"Invalid accuracy drop: {accuracy_drop}% (must be 0-100)",
                code=ErrorCode.INVALID_CONFIGURATION,
                details={"accuracy_drop": accuracy_drop},
            )

    if max_size is not None:
        if max_size <= 0:
            raise ValidationError(
                message=f"Invalid max size: {max_size} MB (must be positive)",
                code=ErrorCode.INVALID_CONFIGURATION,
                details={"max_size": max_size},
            )

    if max_latency is not None:
        if max_latency <= 0:
            raise ValidationError(
                message=f"Invalid max latency: {max_latency} ms (must be positive)",
                code=ErrorCode.INVALID_CONFIGURATION,
                details={"max_latency": max_latency},
            )


def handle_validation_error(error: ValidationError) -> HTTPException:
    """
    Convert ValidationError to HTTPException.

    Args:
        error: ValidationError instance

    Returns:
        HTTPException with appropriate status and message
    """
    logger.error(f"Validation error: {error.code} - {error.message}", extra=error.details)

    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "error": error.code,
            "message": error.message,
            "details": error.details,
        },
    )


def safe_execute(
    func, *args, error_code: ErrorCode = ErrorCode.OPTIMIZATION_FAILED, **kwargs
) -> Any:
    """
    Safely execute a function with error handling.

    Args:
        func: Function to execute
        *args: Positional arguments
        error_code: Error code to use if execution fails
        **kwargs: Keyword arguments

    Returns:
        Function result

    Raises:
        ValidationError: If execution fails
    """
    try:
        return func(*args, **kwargs)
    except ValidationError:
        raise
    except Exception as e:
        logger.exception(f"Error executing {func.__name__}")
        raise ValidationError(  # noqa: B904
            message=f"Operation failed: {str(e)}",
            code=error_code,
            details={"function": func.__name__, "error": str(e)},
        )
