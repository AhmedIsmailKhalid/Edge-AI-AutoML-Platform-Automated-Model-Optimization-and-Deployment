"""
Unit tests for PyTorch Pruning optimization.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.core.pytorch.pruning import (PruningMagnitudeStructuredPyTorch,
                                      PruningMagnitudeUnstructuredPyTorch)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return SimpleModel()


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    inputs = torch.randn(100, 10)
    labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=32)


# ============================================================================
# UNSTRUCTURED PRUNING TESTS
# ============================================================================


def test_pruning_unstructured_initialization():
    """Test unstructured pruning initialization."""
    print("\n  Testing unstructured pruning initialization...")
    optimizer = PruningMagnitudeUnstructuredPyTorch(config={})
    assert optimizer.technique_name == "pruning_magnitude_unstructured"
    assert optimizer.sparsity == 0.5  # default
    print("✅ Initialization test passed")


def test_pruning_unstructured_custom_sparsity():
    """Test custom sparsity configuration."""
    print("\n  Testing custom sparsity...")
    optimizer = PruningMagnitudeUnstructuredPyTorch(config={"sparsity": 0.7})
    assert optimizer.sparsity == 0.7
    print("✅ Custom sparsity test passed")


def test_pruning_unstructured_requires_dataset():
    """Test that pruning requires dataset."""
    print("\n  Testing requires_dataset...")
    optimizer = PruningMagnitudeUnstructuredPyTorch(config={})
    assert optimizer.requires_dataset() is True
    print("✅ Requires dataset test passed")


def test_pruning_unstructured_count_parameters(sample_model):
    """Test parameter counting."""
    print("\n  Testing parameter counting...")
    optimizer = PruningMagnitudeUnstructuredPyTorch(config={})
    param_count = optimizer.count_parameters(sample_model)

    expected_params = (10 * 50 + 50) + (50 * 10 + 10)
    assert param_count == expected_params
    print(f"✅ Parameter count: {param_count}")


def test_pruning_unstructured_estimate_time(sample_model):
    """Test time estimation."""
    print("\n  Testing time estimation...")
    optimizer = PruningMagnitudeUnstructuredPyTorch(config={})
    estimated_time = optimizer.estimate_time(sample_model)

    assert estimated_time > 0
    print(f"✅ Estimated time: {estimated_time:.2f}s")


def test_pruning_unstructured_optimize(sample_model, sample_dataset):
    """Test unstructured pruning optimization."""
    print("\n  Testing unstructured pruning optimization...")
    optimizer = PruningMagnitudeUnstructuredPyTorch(config={"sparsity": 0.5})

    result = optimizer.optimize(sample_model, sample_dataset)

    # Verify result structure
    assert result.optimized_model is not None
    assert result.original_size_mb > 0
    assert result.optimized_size_mb > 0
    assert result.execution_time_seconds > 0

    # Verify sparsity achieved
    nonzero_params = result.optimized_params_count
    total_params = result.metadata["total_params"]
    actual_sparsity = result.metadata["actual_sparsity"]

    print(f"   Original size: {result.original_size_mb:.4f} MB")
    print(f"   Optimized size: {result.optimized_size_mb:.4f} MB")
    print(f"   Total params: {total_params}")
    print(f"   Non-zero params: {nonzero_params}")
    print(f"   Target sparsity: {optimizer.sparsity:.2%}")
    print(f"   Actual sparsity: {actual_sparsity:.2%}")
    print(f"   Execution time: {result.execution_time_seconds:.2f}s")

    # Verify sparsity is close to target (within 5%)
    assert abs(actual_sparsity - optimizer.sparsity) < 0.05

    # Verify metadata
    assert "pruning_type" in result.metadata
    assert result.metadata["pruning_type"] == "magnitude_unstructured"

    print("✅ Unstructured pruning optimization test passed")


def test_pruning_unstructured_optimize_without_dataset(sample_model):
    """Test unstructured pruning without dataset."""
    print("\n  Testing unstructured pruning without dataset...")
    optimizer = PruningMagnitudeUnstructuredPyTorch(config={})

    result = optimizer.optimize(sample_model, dataset=None)

    # Should work but accuracy will be 0
    assert result.optimized_model is not None
    assert result.original_accuracy == 0.0
    assert result.optimized_accuracy == 0.0

    print(f"   Sparsity achieved: {result.metadata['actual_sparsity']:.2%}")
    print("✅ Optimization without dataset test passed")


# ============================================================================
# STRUCTURED PRUNING TESTS
# ============================================================================


def test_pruning_structured_initialization():
    """Test structured pruning initialization."""
    print("\n  Testing structured pruning initialization...")
    optimizer = PruningMagnitudeStructuredPyTorch(config={})
    assert optimizer.technique_name == "pruning_magnitude_structured"
    assert optimizer.sparsity == 0.3  # default (lower for structured)
    print("✅ Initialization test passed")


def test_pruning_structured_custom_sparsity():
    """Test custom sparsity configuration."""
    print("\n  Testing custom sparsity...")
    optimizer = PruningMagnitudeStructuredPyTorch(config={"sparsity": 0.4})
    assert optimizer.sparsity == 0.4
    print("✅ Custom sparsity test passed")


def test_pruning_structured_requires_dataset():
    """Test that pruning requires dataset."""
    print("\n  Testing requires_dataset...")
    optimizer = PruningMagnitudeStructuredPyTorch(config={})
    assert optimizer.requires_dataset() is True
    print("✅ Requires dataset test passed")


def test_pruning_structured_optimize(sample_model, sample_dataset):
    """Test structured pruning optimization."""
    print("\n  Testing structured pruning optimization...")
    optimizer = PruningMagnitudeStructuredPyTorch(config={"sparsity": 0.3})

    result = optimizer.optimize(sample_model, sample_dataset)

    # Verify result structure
    assert result.optimized_model is not None
    assert result.original_size_mb > 0
    assert result.optimized_size_mb > 0
    assert result.execution_time_seconds > 0

    print(f"   Original size: {result.original_size_mb:.4f} MB")
    print(f"   Optimized size: {result.optimized_size_mb:.4f} MB")
    print(
        f"   Size reduction: {((result.original_size_mb - result.optimized_size_mb) / result.original_size_mb * 100):.2f}%"
    )
    print(f"   Pruned layers: {result.metadata['pruned_layers']}")
    print(f"   Execution time: {result.execution_time_seconds:.2f}s")

    # Verify metadata
    assert "pruning_type" in result.metadata
    assert result.metadata["pruning_type"] == "magnitude_structured"
    assert result.metadata["pruned_layers"] > 0

    print("✅ Structured pruning optimization test passed")


def test_pruning_structured_optimize_without_dataset(sample_model):
    """Test structured pruning without dataset."""
    print("\n  Testing structured pruning without dataset...")
    optimizer = PruningMagnitudeStructuredPyTorch(config={})

    result = optimizer.optimize(sample_model, dataset=None)

    # Should work but accuracy will be 0
    assert result.optimized_model is not None
    assert result.original_accuracy == 0.0
    assert result.optimized_accuracy == 0.0

    print(f"   Pruned layers: {result.metadata['pruned_layers']}")
    print("✅ Optimization without dataset test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
