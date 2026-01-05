"""
Base optimization class defining the interface for all optimization techniques.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class OptimizationResult:
    """
    Result of an optimization technique execution.
    """

    optimized_model: Any  # torch.nn.Module or tf.keras.Model
    original_accuracy: float
    optimized_accuracy: float
    original_size_mb: float
    optimized_size_mb: float
    original_params_count: int
    optimized_params_count: int
    execution_time_seconds: float
    metadata: dict[str, Any]  # Additional technique-specific metadata
    technique_name: str


class BaseOptimization(ABC):
    """
    Abstract base class for all optimization techniques.
    All optimization techniques must inherit from this class and implement its methods.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize optimization technique.

        Args:
            config: Configuration dictionary for the technique
        """
        self.config = config
        self.technique_name = self.__class__.__name__

    @abstractmethod
    def optimize(self, model: Any, dataset: Any | None = None, **kwargs) -> OptimizationResult:
        """
        Apply optimization technique to the model.

        Args:
            model: Model to optimize (torch.nn.Module or tf.keras.Model)
            dataset: Dataset for evaluation/training (if required)
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with optimized model and metrics
        """
        pass

    @abstractmethod
    def requires_dataset(self) -> bool:
        """
        Check if this optimization technique requires a dataset.

        Returns:
            True if dataset is required, False otherwise
        """
        pass

    @abstractmethod
    def estimate_time(self, model: Any) -> float:
        """
        Estimate execution time for this optimization technique.

        Args:
            model: Model to be optimized

        Returns:
            Estimated time in seconds
        """
        pass

    def get_model_size_mb(self, model_path: Path) -> float:
        """
        Get model file size in megabytes.

        Args:
            model_path: Path to model file

        Returns:
            Model size in MB
        """
        size_bytes = model_path.stat().st_size
        return size_bytes / (1024 * 1024)

    def count_parameters(self, model: Any) -> int:
        """
        Count total parameters in model.
        Must be implemented by framework-specific subclasses.

        Args:
            model: Model to count parameters

        Returns:
            Total number of parameters
        """
        raise NotImplementedError("Must be implemented by framework-specific class")

    def evaluate_model(self, model: Any, dataset: Any, **kwargs) -> float:
        """
        Evaluate model accuracy on dataset.
        Must be implemented by framework-specific subclasses.

        Args:
            model: Model to evaluate
            dataset: Evaluation dataset
            **kwargs: Additional arguments

        Returns:
            Accuracy as float between 0 and 1
        """
        raise NotImplementedError("Must be implemented by framework-specific class")
