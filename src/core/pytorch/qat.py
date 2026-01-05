"""
Quantization-Aware Training (QAT) for PyTorch models.
"""

import copy
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.core.base import BaseOptimization, OptimizationResult


class QuantizationAwareTrainingPyTorch(BaseOptimization):
    """
    Quantization-Aware Training for PyTorch models.
    Uses dynamic quantization with fine-tuning for better compatibility.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize QAT optimizer.

        Args:
            config: Configuration dictionary
                - epochs: Fine-tuning epochs (default: 3)
                - learning_rate: Learning rate (default: 0.0001)
        """
        super().__init__(config)
        self.technique_name = "quantization_aware_training"
        self.epochs = config.get("epochs", 3)
        self.learning_rate = config.get("learning_rate", 0.0001)

    def requires_dataset(self) -> bool:
        """QAT requires dataset for fine-tuning."""
        return True

    def estimate_time(self, model: nn.Module) -> float:
        """
        Estimate execution time based on model size and epochs.

        Args:
            model: PyTorch model

        Returns:
            Estimated time in seconds
        """
        param_count = sum(p.numel() for p in model.parameters())
        # Training takes time
        return max(10.0, (param_count / 800_000) * self.epochs)

    def count_parameters(self, model: nn.Module) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def evaluate_model(self, model: nn.Module, dataloader: Any, device: str = "cpu") -> float:
        """Evaluate PyTorch model accuracy."""
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
        Apply Quantization-Aware Training.

        Note: For PyTorch, we use fine-tuning followed by dynamic quantization
        for better compatibility across different model architectures.

        Args:
            model: PyTorch model to optimize
            dataset: Training dataset for fine-tuning
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with quantized model and metrics
        """
        start_time = time.time()
        device = "cpu"

        if dataset is None:
            raise ValueError("QAT requires a dataset for fine-tuning")

        # Save original model to get size
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
            original_path = Path(tmp_file.name)
            torch.save(model.state_dict(), original_path)

        # Get original metrics
        original_size_mb = self.get_model_size_mb(original_path)
        original_params = self.count_parameters(model)
        original_accuracy = self.evaluate_model(model, dataset, device)

        # Create a copy for fine-tuning
        finetuned_model = copy.deepcopy(model)
        finetuned_model.to(device)
        finetuned_model.train()

        # Fine-tune the model
        # Get trainable parameters
        trainable_params = [p for p in finetuned_model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found in model.")

        optimizer = torch.optim.Adam(trainable_params, lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.epochs):
            for inputs, labels in dataset:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = finetuned_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Set to eval mode
        finetuned_model.eval()

        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            finetuned_model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )

        # Save quantized model
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_quantized:
            quantized_path = Path(tmp_quantized.name)
            torch.save(quantized_model.state_dict(), quantized_path)

        # Get optimized metrics
        optimized_size_mb = self.get_model_size_mb(quantized_path)
        optimized_params = self.count_parameters(quantized_model)

        # Evaluate quantized accuracy
        optimized_accuracy = self.evaluate_model(quantized_model, dataset, device)

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
                "technique": "quantization_aware_training",
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "quantization_type": "dynamic_post_finetuning",
                "dtype": "qint8",
                "note": "Fine-tuned then dynamically quantized for better compatibility",
            },
            technique_name="quantization_aware_training",
        )

        return result
