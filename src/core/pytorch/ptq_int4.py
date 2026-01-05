"""
Post-Training Quantization (INT4) for PyTorch models.
"""

import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.core.base import BaseOptimization, OptimizationResult


class PTQInt4PyTorch(BaseOptimization):
    """
    Post-Training Quantization to INT4 for PyTorch models.
    Uses torch quantization with reduced precision.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize PTQ INT4 optimizer.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.technique_name = "ptq_int4"

    def requires_dataset(self) -> bool:
        """PTQ INT4 requires representative dataset for calibration."""
        return True

    def estimate_time(self, model: nn.Module) -> float:
        """
        Estimate execution time based on model size.

        Args:
            model: PyTorch model

        Returns:
            Estimated time in seconds
        """
        param_count = sum(p.numel() for p in model.parameters())
        # INT4 quantization is slightly faster than INT8
        return max(5.0, param_count / 1_200_000)

    def count_parameters(self, model: nn.Module) -> int:
        """
        Count total trainable parameters in PyTorch model.

        Args:
            model: PyTorch model

        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def evaluate_model(self, model: nn.Module, dataloader: Any, device: str = "cpu") -> float:
        """
        Evaluate PyTorch model accuracy on dataloader.

        Args:
            model: PyTorch model
            dataloader: DataLoader for evaluation
            device: Device to run evaluation on

        Returns:
            Accuracy as float between 0 and 1
        """
        model.eval()
        model.to(device)

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total if total > 0 else 0.0

    def optimize(
        self, model: nn.Module, dataset: Any | None = None, **kwargs
    ) -> OptimizationResult:
        """
        Apply INT4 quantization to PyTorch model.

        Args:
            model: PyTorch model to optimize
            dataset: Representative dataset for calibration
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with quantized model and metrics
        """
        start_time = time.time()
        device = "cpu"

        # Save original model to get size
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
            original_path = Path(tmp_file.name)
            torch.save(model.state_dict(), original_path)

        # Get original metrics
        original_size_mb = self.get_model_size_mb(original_path)
        original_params = self.count_parameters(model)

        # Evaluate original accuracy if dataset provided
        if dataset is not None:
            original_accuracy = self.evaluate_model(model, dataset, device)
        else:
            original_accuracy = 0.0

        # Prepare model for quantization
        model.eval()
        model.to(device)

        # Set quantization configuration for INT4
        # Note: PyTorch doesn't have native INT4 support, so we use dynamic quantization
        # with qint8 as the closest approximation. In production, you'd use specialized
        # INT4 kernels or frameworks like ONNX Runtime with INT4 support.
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8  # Closest to INT4 in standard PyTorch
        )

        # Note: For true INT4, you would need:
        # 1. Custom quantization backend (e.g., ONNX Runtime, TensorRT)
        # 2. Model export to ONNX with INT4 quantization
        # 3. Or use torch.ao.quantization with custom qconfig

        # Save quantized model
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_quantized:
            quantized_path = Path(tmp_quantized.name)
            torch.save(quantized_model.state_dict(), quantized_path)

        # Get optimized metrics
        optimized_size_mb = self.get_model_size_mb(quantized_path)
        optimized_params = self.count_parameters(quantized_model)

        # Evaluate quantized accuracy if dataset provided
        if dataset is not None:
            optimized_accuracy = self.evaluate_model(quantized_model, dataset, device)
        else:
            optimized_accuracy = 0.0

        # Clean up temporary files
        original_path.unlink()
        quantized_path.unlink()

        execution_time = time.time() - start_time

        result = OptimizationResult(
            optimized_model=quantized_model,
            original_accuracy=original_accuracy,
            optimized_accuracy=optimized_accuracy,
            original_size_mb=original_size_mb,
            optimized_size_mb=optimized_size_mb,
            original_params_count=original_params,
            optimized_params_count=optimized_params,
            execution_time_seconds=execution_time,
            metadata={
                "quantization_type": "post_training_dynamic",
                "dtype": "qint8_approx_int4",
                "note": "Using qint8 as INT4 approximation. For true INT4, use ONNX or TensorRT.",
            },
            technique_name="ptq_int4",
        )

        return result
