"""
Model loading utilities for PyTorch and TensorFlow.
"""

from pathlib import Path
from typing import Any

import tensorflow as tf
import torch

from scripts.create_pretrained_models import (PyTorchLargeCNN,
                                              PyTorchMediumCNN,
                                              PyTorchSmallCNN)


def load_model(framework: str, model_path: Path):
    """Load model from file."""
    if framework == "pytorch":
        # Load state dict and reconstruct model
        state_dict = torch.load(model_path, map_location="cpu")

        # Try to infer architecture from state dict
        total_params = sum(p.numel() for p in state_dict.values())

        # Detect input channels and num classes from state dict
        first_conv_weight = state_dict["conv1.weight"]
        input_channels = first_conv_weight.shape[1]

        last_fc_weight = [v for k, v in state_dict.items() if "fc" in k and "weight" in k][-1]
        num_classes = last_fc_weight.shape[0]

        # Select architecture based on param count
        if total_params < 200_000:
            model = PyTorchSmallCNN(input_channels, num_classes)
        elif total_params < 1_000_000:
            model = PyTorchMediumCNN(input_channels, num_classes)
        else:
            model = PyTorchLargeCNN(input_channels, num_classes)

        model.load_state_dict(state_dict)
        model.eval()
        return model

    elif framework == "tensorflow":
        # Load TensorFlow model
        return _load_tensorflow_model(model_path)

    else:
        raise ValueError(f"Unsupported framework: {framework}")


def _load_pytorch_model(model_path: Path) -> Any:
    """
    Load PyTorch model from file.

    Supports:
    - TorchScript (.pt)
    - State dict (.pth)
    - Pickled models (.pkl)

    Args:
        model_path: Path to PyTorch model file

    Returns:
        Loaded PyTorch model
    """
    extension = model_path.suffix.lower()

    # Try TorchScript first (preferred format)
    if extension == ".pt":
        try:
            model = torch.jit.load(str(model_path))
            model.eval()
            return model
        except Exception:
            # If TorchScript fails, try regular torch.load
            pass

    # Try loading as full model or state dict
    try:
        model = torch.load(model_path, map_location=torch.device("cpu"))

        # If it's a state dict, we can't use it without architecture
        if isinstance(model, dict):
            raise RuntimeError(
                "Model is a state_dict but architecture is not available. "
                "Please provide a TorchScript (.pt) model or full model object."
            )

        model.eval()
        return model

    except Exception as e:
        raise RuntimeError(
            f"Failed to load PyTorch model. Error: {e}\n"
            "The model class definition may not be available. "
            "Please ensure the model is saved as TorchScript (.pt) or "
            "as a complete object that can be loaded independently."
        ) from e


def _load_tensorflow_model(model_path: Path) -> Any:
    """
    Load TensorFlow model from file.

    Supports:
    - SavedModel format (.pb)
    - HDF5 format (.h5)
    - Keras format (.keras)

    Args:
        model_path: Path to TensorFlow model file

    Returns:
        Loaded TensorFlow model
    """
    extension = model_path.suffix.lower()

    try:
        if extension in [".h5", ".keras"]:
            # Load Keras model
            model = tf.keras.models.load_model(str(model_path))
        elif extension == ".pb":
            # Load SavedModel
            model = tf.saved_model.load(str(model_path))
        else:
            # Try to load as directory (SavedModel format)
            model = tf.saved_model.load(str(model_path))

        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load TensorFlow model: {e}") from e
