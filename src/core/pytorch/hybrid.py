"""
Hybrid Optimization (Pruning + Quantization) for PyTorch models.
"""

import copy
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.core.base import BaseOptimization, OptimizationResult


class HybridPruneQuantizePyTorch(BaseOptimization):
    """
    Hybrid optimization combining pruning and quantization.
    First prunes the model, then applies quantization for maximum compression.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize hybrid optimizer.

        Args:
            config: Configuration dictionary
                - sparsity: Pruning sparsity (default: 0.5 = 50%)
                - fine_tune_epochs: Epochs for fine-tuning after pruning (default: 2)
        """
        super().__init__(config)
        self.technique_name = "hybrid_prune_quantize"
        self.sparsity = config.get("sparsity", 0.5)
        self.fine_tune_epochs = config.get("fine_tune_epochs", 2)

    def requires_dataset(self) -> bool:
        """Hybrid requires dataset for fine-tuning."""
        return True

    def estimate_time(self, model: nn.Module) -> float:
        """
        Estimate execution time.

        Args:
            model: PyTorch model

        Returns:
            Estimated time in seconds
        """
        param_count = sum(p.numel() for p in model.parameters())
        # Both pruning and quantization + fine-tuning
        return max(8.0, (param_count / 1_000_000) * self.fine_tune_epochs)

    def count_parameters(self, model: nn.Module) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def count_nonzero_parameters(self, model: nn.Module) -> int:
        """Count non-zero parameters (after pruning)."""
        return sum((p != 0).sum().item() for p in model.parameters() if p.requires_grad)

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
        Apply hybrid pruning + quantization.

        Args:
            model: PyTorch model to optimize
            dataset: Dataset for fine-tuning and evaluation
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with hybrid optimized model and metrics
        """
        start_time = time.time()
        device = "cpu"

        if dataset is None:
            raise ValueError("Hybrid optimization requires a dataset for fine-tuning")

        # Save original model to get size
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
            original_path = Path(tmp_file.name)
            torch.save(model.state_dict(), original_path)

        # Get original metrics
        original_size_mb = self.get_model_size_mb(original_path)
        original_params = self.count_parameters(model)
        original_accuracy = self.evaluate_model(model, dataset, device)

        # Step 1: Apply pruning
        pruned_model = copy.deepcopy(model)
        pruned_model.to(device)

        # Identify layers to prune
        parameters_to_prune = []
        for name, module in pruned_model.named_modules():  # noqa
            if isinstance(module, (nn.Linear, nn.Conv2d)):  # noqa
                parameters_to_prune.append((module, "weight"))

        # Check if we found any layers to prune
        if not parameters_to_prune:
            raise ValueError("No prunable layers (Conv2d/Linear) found in model.")

        # Apply global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.sparsity,
        )

        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        # Step 2: Fine-tune pruned model
        pruned_model.train()
        optimizer = torch.optim.Adam(pruned_model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.fine_tune_epochs):  # noqa
            for inputs, labels in dataset:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = pruned_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Set to eval mode
        pruned_model.eval()

        # Step 3: Apply quantization
        quantized_model = torch.quantization.quantize_dynamic(
            pruned_model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )

        # Save final hybrid model
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_hybrid:
            hybrid_path = Path(tmp_hybrid.name)
            torch.save(quantized_model.state_dict(), hybrid_path)

        # Get optimized metrics
        optimized_size_mb = self.get_model_size_mb(hybrid_path)
        optimized_params = self.count_parameters(
            pruned_model
        )  # Count from pruned model, not quantized
        nonzero_params = self.count_nonzero_parameters(pruned_model)

        # Evaluate final accuracy
        optimized_accuracy = self.evaluate_model(quantized_model, dataset, device)

        # Clean up temporary files
        original_path.unlink()
        hybrid_path.unlink()

        execution_time = time.time() - start_time

        # Calculate actual sparsity
        actual_sparsity = 1.0 - (nonzero_params / optimized_params) if optimized_params > 0 else 0.0

        result = OptimizationResult(
            optimized_model=quantized_model,
            original_accuracy=original_accuracy,
            optimized_accuracy=optimized_accuracy,
            original_size_mb=original_size_mb,
            optimized_size_mb=optimized_size_mb,
            original_params_count=original_params,
            optimized_params_count=nonzero_params,
            execution_time_seconds=execution_time,
            metadata={
                "technique": "hybrid_prune_quantize",
                "pruning_sparsity": self.sparsity,
                "actual_sparsity": actual_sparsity,
                "fine_tune_epochs": self.fine_tune_epochs,
                "quantization_dtype": "qint8",
                "pipeline": "prune -> fine-tune -> quantize",
            },
            technique_name="hybrid_prune_quantize",
        )

        return result
