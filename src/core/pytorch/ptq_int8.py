"""
Post-Training Quantization (INT8) for PyTorch models.
"""

import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.core.base import BaseOptimization, OptimizationResult


class PTQInt8PyTorch(BaseOptimization):
    """
    Post-Training Quantization to INT8 for PyTorch models.
    Uses dynamic quantization for simplicity and broad compatibility.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize PTQ INT8 optimizer.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.technique_name = "ptq_int8"

    def requires_dataset(self) -> bool:
        """PTQ INT8 does not require dataset for dynamic quantization."""
        return False

    def estimate_time(self, model: Any) -> float:
        """
        Estimate execution time based on model size.

        Args:
            model: PyTorch model

        Returns:
            Estimated time in seconds
        """
        param_count = self.count_parameters(model)
        # Rough estimate: ~1 second per million parameters
        return max(5.0, param_count / 1_000_000)

    def count_parameters(self, model: nn.Module) -> int:
        """
        Count total trainable parameters in PyTorch model.

        Args:
            model: PyTorch model

        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in model.parameters())

    def evaluate_model(self, model: nn.Module, dataset: Any, device: str = "cpu") -> float:
        """
        Evaluate PyTorch model accuracy on dataset.

        Args:
            model: PyTorch model
            dataset: DataLoader for evaluation
            device: Device to run evaluation on

        Returns:
            Accuracy as float between 0 and 1
        """
        model.eval()
        model.to(device)

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataset:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def optimize(
        self, model: nn.Module, dataset: Any | None = None, device: str = "cpu", **kwargs
    ) -> OptimizationResult:
        """
        Apply dynamic INT8 quantization to PyTorch model.

        Args:
            model: PyTorch model to optimize
            dataset: Optional evaluation dataset
            device: Device to run on
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with quantized model and metrics
        """
        start_time = time.time()

        # Move model to CPU for quantization (required)
        model.eval()
        model.to("cpu")

        # Save original model to get size
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_original:
            original_path = Path(tmp_original.name)
            torch.save(model.state_dict(), original_path)

        # Get original metrics
        original_size_mb = self.get_model_size_mb(original_path)
        original_params = self.count_parameters(model)

        # Evaluate original accuracy if dataset provided
        if dataset is not None:
            original_accuracy = self.evaluate_model(model, dataset, device)
        else:
            original_accuracy = 0.0  # Will be set by orchestrator

        # Apply dynamic quantization
        # Quantize Linear and LSTM layers to INT8
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8  # Layers to quantize
        )

        # Save quantized model to get size
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_quantized:
            quantized_path = Path(tmp_quantized.name)
            torch.save(quantized_model.state_dict(), quantized_path)

        # Get optimized metrics
        optimized_size_mb = self.get_model_size_mb(quantized_path)
        optimized_params = self.count_parameters(quantized_model)

        # Evaluate optimized accuracy if dataset provided
        if dataset is not None:
            optimized_accuracy = self.evaluate_model(quantized_model, dataset, device)
        else:
            optimized_accuracy = 0.0  # Will be set by orchestrator

        # Clean up temporary files
        original_path.unlink()
        quantized_path.unlink()

        execution_time = time.time() - start_time

        # Create result
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
                "quantization_type": "dynamic",
                "dtype": "qint8",
                "quantized_layers": ["Linear", "LSTM", "GRU"],
            },
            technique_name="ptq_int8",
        )

        return result
