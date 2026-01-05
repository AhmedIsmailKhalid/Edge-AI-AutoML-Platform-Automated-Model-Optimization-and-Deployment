"""
Knowledge Distillation for PyTorch models.
"""

import copy
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.base import BaseOptimization, OptimizationResult


class KnowledgeDistillationPyTorch(BaseOptimization):
    """
    Knowledge Distillation for PyTorch models.
    Trains a smaller student model to mimic a larger teacher model.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize knowledge distillation optimizer.

        Args:
            config: Configuration dictionary
                - student_scale: Scale factor for student model size (default: 0.5)
                - temperature: Distillation temperature (default: 3.0)
                - alpha: Weight for distillation loss (default: 0.7)
                - epochs: Training epochs (default: 5)
        """
        super().__init__(config)
        self.technique_name = "knowledge_distillation"
        self.student_scale = config.get("student_scale", 0.5)
        self.temperature = config.get("temperature", 3.0)
        self.alpha = config.get("alpha", 0.7)
        self.epochs = config.get("epochs", 5)

    def requires_dataset(self) -> bool:
        """Knowledge distillation requires dataset for training."""
        return True

    def estimate_time(self, model: nn.Module) -> float:
        """
        Estimate execution time based on model size and training epochs.

        Args:
            model: PyTorch model

        Returns:
            Estimated time in seconds
        """
        param_count = sum(p.numel() for p in model.parameters())
        # Training takes longer - estimate based on epochs
        return max(10.0, (param_count / 500_000) * self.epochs)

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

    def create_student_model(self, teacher_model: nn.Module) -> nn.Module:
        """
        Create a smaller student model based on teacher architecture.

        Args:
            teacher_model: Teacher model

        Returns:
            Student model with reduced capacity
        """
        # Simple heuristic: scale down linear/conv layers
        student_layers = []

        for name, module in teacher_model.named_children():  # noqa
            if isinstance(module, nn.Linear):
                # Scale down the hidden dimension
                in_features = module.in_features
                out_features = module.out_features

                # Don't scale the output layer (classification head)
                if out_features <= 20:  # Likely output layer
                    student_layers.append(
                        nn.Linear(int(in_features * self.student_scale), out_features)
                    )
                else:
                    student_layers.append(
                        nn.Linear(
                            in_features
                            if len(student_layers) == 0
                            else int(in_features * self.student_scale),
                            int(out_features * self.student_scale),
                        )
                    )

            elif isinstance(module, nn.Conv2d):
                # Scale down channels
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = module.kernel_size
                stride = module.stride
                padding = module.padding

                student_layers.append(
                    nn.Conv2d(
                        in_channels
                        if len(student_layers) == 0
                        else int(in_channels * self.student_scale),
                        int(out_channels * self.student_scale),
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )

            elif isinstance(
                module, nn.ReLU | nn.Dropout | nn.BatchNorm2d | nn.MaxPool2d | nn.Flatten
            ):
                # Keep activation/regularization layers as is
                student_layers.append(copy.deepcopy(module))

        return nn.Sequential(*student_layers)

    def distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        alpha: float,
    ) -> torch.Tensor:
        """
        Calculate distillation loss.

        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs
            labels: Ground truth labels
            temperature: Temperature for softening distributions
            alpha: Weight for distillation loss

        Returns:
            Combined loss
        """
        # Soft targets loss (KL divergence)
        soft_targets_loss = F.kl_div(
            F.log_softmax(student_outputs / temperature, dim=1),
            F.softmax(teacher_outputs / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature**2)

        # Hard targets loss (cross entropy)
        hard_targets_loss = F.cross_entropy(student_outputs, labels)

        # Combined loss
        return alpha * soft_targets_loss + (1 - alpha) * hard_targets_loss

    def optimize(
        self, model: nn.Module, dataset: Any | None = None, **kwargs
    ) -> OptimizationResult:
        """
        Apply knowledge distillation.

        Args:
            model: Teacher model (PyTorch)
            dataset: Training dataset
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with student model and metrics
        """
        start_time = time.time()
        device = "cpu"

        if dataset is None:
            raise ValueError("Knowledge distillation requires a dataset for training")

        # Save teacher model to get size
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
            teacher_path = Path(tmp_file.name)
            torch.save(model.state_dict(), teacher_path)

        # Get teacher metrics
        teacher_size_mb = self.get_model_size_mb(teacher_path)
        teacher_params = self.count_parameters(model)
        teacher_accuracy = self.evaluate_model(model, dataset, device)

        # Create student model
        student_model = self.create_student_model(model)
        student_model.to(device)

        # Set teacher to eval mode
        model.eval()
        model.to(device)

        # Train student model
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

        for epoch in range(self.epochs):  # noqa
            student_model.train()

            for inputs, labels in dataset:
                inputs, labels = inputs.to(device), labels.to(device)

                # Get teacher outputs (no gradient)
                with torch.no_grad():
                    teacher_outputs = model(inputs)

                # Get student outputs
                student_outputs = student_model(inputs)

                # Calculate distillation loss
                loss = self.distillation_loss(
                    student_outputs, teacher_outputs, labels, self.temperature, self.alpha
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Save student model
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_student:
            student_path = Path(tmp_student.name)
            torch.save(student_model.state_dict(), student_path)

        # Get student metrics
        student_size_mb = self.get_model_size_mb(student_path)
        student_params = self.count_parameters(student_model)
        student_accuracy = self.evaluate_model(student_model, dataset, device)

        # Clean up temporary files
        teacher_path.unlink()
        student_path.unlink()

        execution_time = time.time() - start_time

        result = OptimizationResult(
            optimized_model=student_model,
            original_accuracy=teacher_accuracy,
            optimized_accuracy=student_accuracy,
            original_size_mb=teacher_size_mb,
            optimized_size_mb=student_size_mb,
            original_params_count=teacher_params,
            optimized_params_count=student_params,
            execution_time_seconds=execution_time,
            metadata={
                "technique": "knowledge_distillation",
                "student_scale": self.student_scale,
                "temperature": self.temperature,
                "alpha": self.alpha,
                "epochs": self.epochs,
            },
            technique_name="distillation",
        )

        return result
