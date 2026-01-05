"""
Unit tests for PyTorch Hybrid optimization.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.core.pytorch.hybrid import HybridPruneQuantizePyTorch


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


def test_hybrid_initialization():
    """Test hybrid initialization."""
    print("\n   Testing hybrid initialization...")
    optimizer = HybridPruneQuantizePyTorch(config={})
    assert optimizer.technique_name == "hybrid_prune_quantize"
    assert optimizer.sparsity == 0.5  # default
    assert optimizer.fine_tune_epochs == 2  # default
    print("✅ Initialization test passed")


def test_hybrid_custom_config():
    """Test custom configuration."""
    print("\n   Testing custom configuration...")
    optimizer = HybridPruneQuantizePyTorch(config={"sparsity": 0.6, "fine_tune_epochs": 3})
    assert optimizer.sparsity == 0.6
    assert optimizer.fine_tune_epochs == 3
    print("✅ Custom configuration test passed")


def test_hybrid_requires_dataset():
    """Test that hybrid requires dataset."""
    print("\n   Testing requires_dataset...")
    optimizer = HybridPruneQuantizePyTorch(config={})
    assert optimizer.requires_dataset() is True
    print("✅ Requires dataset test passed")


def test_hybrid_count_parameters(sample_model):
    """Test parameter counting."""
    print("\n   Testing parameter counting...")
    optimizer = HybridPruneQuantizePyTorch(config={})
    param_count = optimizer.count_parameters(sample_model)

    expected_params = (10 * 50 + 50) + (50 * 10 + 10)
    assert param_count == expected_params
    print(f"✅ Parameter count: {param_count}")


def test_hybrid_estimate_time(sample_model):
    """Test time estimation."""
    print("\n   Testing time estimation...")
    optimizer = HybridPruneQuantizePyTorch(config={"fine_tune_epochs": 2})
    estimated_time = optimizer.estimate_time(sample_model)

    assert estimated_time > 0
    print(f"✅ Estimated time: {estimated_time:.2f}s")


def test_hybrid_optimize(sample_model, sample_dataset):
    """Test hybrid optimization."""
    print("\n   Testing hybrid optimization...")
    optimizer = HybridPruneQuantizePyTorch(config={"sparsity": 0.5, "fine_tune_epochs": 2})

    result = optimizer.optimize(sample_model, sample_dataset)

    # Verify result structure
    assert result.optimized_model is not None
    assert result.original_size_mb > 0
    assert result.optimized_size_mb > 0
    assert result.optimized_size_mb < result.original_size_mb
    assert result.execution_time_seconds > 0

    # Verify parameter reduction (due to pruning)
    assert result.optimized_params_count < result.original_params_count

    print(f"   Original size: {result.original_size_mb:.4f} MB")
    print(f"   Hybrid size: {result.optimized_size_mb:.4f} MB")
    print(
        f"   Size reduction: {((result.original_size_mb - result.optimized_size_mb) / result.original_size_mb * 100):.2f}%"
    )
    print(f"   Original params: {result.original_params_count}")
    print(f"   Optimized params: {result.optimized_params_count}")
    print(
        f"   Param reduction: {((result.original_params_count - result.optimized_params_count) / result.original_params_count * 100):.2f}%"
    )
    print(f"   Execution time: {result.execution_time_seconds:.2f}s")

    # Verify metadata
    assert "technique" in result.metadata
    assert result.metadata["technique"] == "hybrid_prune_quantize"
    assert "actual_sparsity" in result.metadata
    assert "pipeline" in result.metadata

    print("✅ Hybrid optimization test passed")


def test_hybrid_optimize_without_dataset(sample_model):
    """Test that hybrid fails without dataset."""
    print("\n   Testing hybrid without dataset...")
    optimizer = HybridPruneQuantizePyTorch(config={})

    with pytest.raises(ValueError, match="requires a dataset"):
        optimizer.optimize(sample_model, dataset=None)

    print("✅ Correctly raises error without dataset")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
