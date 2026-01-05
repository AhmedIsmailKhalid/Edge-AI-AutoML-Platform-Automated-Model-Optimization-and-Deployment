"""
Magnitude-based Pruning for PyTorch models.
Includes both unstructured and structured pruning.
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


class PruningMagnitudeUnstructuredPyTorch(BaseOptimization):
    """
    Magnitude-based unstructured pruning for PyTorch models.
    Prunes individual weights based on their magnitude.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize pruning optimizer.

        Args:
            config: Configuration dictionary
                - sparsity: Target sparsity level (default: 0.5 = 50%)
        """
        super().__init__(config)
        self.technique_name = "pruning_magnitude_unstructured"
        self.sparsity = config.get("sparsity", 0.5)

    def requires_dataset(self) -> bool:
        """Pruning requires dataset for fine-tuning and evaluation."""
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
        # Pruning is relatively fast
        return max(3.0, param_count / 2_000_000)

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
        Apply magnitude-based unstructured pruning.

        Args:
            model: PyTorch model to optimize
            dataset: Dataset for evaluation
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with pruned model and metrics
        """
        start_time = time.time()
        device = "cpu"

        # Create a copy of the model to prune
        pruned_model = copy.deepcopy(model)
        pruned_model.to(device)

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

        # Apply global unstructured pruning
        parameters_to_prune = []
        for _name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):  # noqa
                parameters_to_prune.append((module, "weight"))

        # Check if we found any layers to prune
        if not parameters_to_prune:
            raise ValueError("No prunable layers (Conv2d/Linear) found in model.")

        # Apply pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.sparsity,
        )

        # Make pruning permanent (remove reparametrization)
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        # Save pruned model
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_pruned:
            pruned_path = Path(tmp_pruned.name)
            torch.save(pruned_model.state_dict(), pruned_path)

        # Get optimized metrics
        optimized_size_mb = self.get_model_size_mb(pruned_path)
        optimized_params = self.count_parameters(pruned_model)
        nonzero_params = self.count_nonzero_parameters(pruned_model)

        # Evaluate pruned accuracy if dataset provided
        if dataset is not None:
            optimized_accuracy = self.evaluate_model(pruned_model, dataset, device)
        else:
            optimized_accuracy = 0.0

        # Clean up temporary files
        original_path.unlink()
        pruned_path.unlink()

        execution_time = time.time() - start_time

        # Calculate actual sparsity
        actual_sparsity = 1.0 - (nonzero_params / optimized_params)

        result = OptimizationResult(
            optimized_model=pruned_model,
            original_accuracy=original_accuracy,
            optimized_accuracy=optimized_accuracy,
            original_size_mb=original_size_mb,
            optimized_size_mb=optimized_size_mb,
            original_params_count=original_params,
            optimized_params_count=nonzero_params,
            execution_time_seconds=execution_time,
            metadata={
                "pruning_type": "magnitude_unstructured",
                "target_sparsity": self.sparsity,
                "actual_sparsity": actual_sparsity,
                "total_params": optimized_params,
                "nonzero_params": nonzero_params,
            },
            technique_name="pruning_magnitude_unstructured",
        )

        return result


class PruningMagnitudeStructuredPyTorch(BaseOptimization):
    """
    Magnitude-based structured pruning for PyTorch models.
    Prunes entire channels/filters based on their L1 norm.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize structured pruning optimizer.

        Args:
            config: Configuration dictionary
                - sparsity: Target sparsity level (default: 0.3 = 30%)
        """
        super().__init__(config)
        self.technique_name = "pruning_magnitude_structured"
        self.sparsity = config.get("sparsity", 0.3)  # Lower default for structured

    def requires_dataset(self) -> bool:
        """Pruning requires dataset for evaluation."""
        return True

    def estimate_time(self, model: nn.Module) -> float:
        """Estimate execution time."""
        param_count = sum(p.numel() for p in model.parameters())
        return max(3.0, param_count / 2_000_000)

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
        Apply magnitude-based structured pruning.

        Args:
            model: PyTorch model to optimize
            dataset: Dataset for evaluation
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with pruned model and metrics
        """
        start_time = time.time()
        device = "cpu"

        # Create a copy of the model to prune
        pruned_model = copy.deepcopy(model)
        pruned_model.to(device)

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

        # Apply structured pruning to Conv2d layers (prune channels)
        pruned_layers = 0
        for _name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Prune channels based on L1 norm
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=self.sparsity,
                    n=1,  # L1 norm
                    dim=0,  # Prune output channels
                )
                prune.remove(module, "weight")
                pruned_layers += 1

        # Apply structured pruning to Linear layers (prune neurons)
        for _name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=self.sparsity,
                    n=1,  # L1 norm
                    dim=0,  # Prune output neurons
                )
                prune.remove(module, "weight")
                pruned_layers += 1

        # Save pruned model
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_pruned:
            pruned_path = Path(tmp_pruned.name)
            torch.save(pruned_model.state_dict(), pruned_path)

        # Get optimized metrics
        optimized_size_mb = self.get_model_size_mb(pruned_path)
        optimized_params = self.count_parameters(pruned_model)

        # Evaluate pruned accuracy if dataset provided
        if dataset is not None:
            optimized_accuracy = self.evaluate_model(pruned_model, dataset, device)
        else:
            optimized_accuracy = 0.0

        # Clean up temporary files
        original_path.unlink()
        pruned_path.unlink()

        execution_time = time.time() - start_time

        result = OptimizationResult(
            optimized_model=pruned_model,
            original_accuracy=original_accuracy,
            optimized_accuracy=optimized_accuracy,
            original_size_mb=original_size_mb,
            optimized_size_mb=optimized_size_mb,
            original_params_count=original_params,
            optimized_params_count=optimized_params,
            execution_time_seconds=execution_time,
            metadata={
                "pruning_type": "magnitude_structured",
                "target_sparsity": self.sparsity,
                "pruned_layers": pruned_layers,
            },
            technique_name="pruning_magnitude_structured",
        )

        return result
