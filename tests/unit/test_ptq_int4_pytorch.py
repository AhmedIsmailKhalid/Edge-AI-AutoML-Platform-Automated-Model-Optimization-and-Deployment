"""
Unit tests for PyTorch PTQ INT4 optimization.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.core.pytorch.ptq_int4 import PTQInt4PyTorch


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
    # Create dummy data
    inputs = torch.randn(100, 10)
    labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=32)


def test_ptq_int4_initialization():
    """Test PTQ INT4 optimizer initialization."""
    print("\n🧪 Testing PTQ INT4 initialization...")
    optimizer = PTQInt4PyTorch(config={})
    assert optimizer.technique_name == "ptq_int4"
    print("✅ Initialization test passed")


def test_ptq_int4_requires_dataset():
    """Test that PTQ INT4 requires dataset."""
    print("\n🧪 Testing requires_dataset...")
    optimizer = PTQInt4PyTorch(config={})
    assert optimizer.requires_dataset() is True
    print("✅ Requires dataset test passed")


def test_ptq_int4_count_parameters(sample_model):
    """Test parameter counting."""
    print("\n🧪 Testing parameter counting...")
    optimizer = PTQInt4PyTorch(config={})
    param_count = optimizer.count_parameters(sample_model)

    # Model has: fc1 (10*50 + 50) + fc2 (50*10 + 10) = 1110 parameters
    expected_params = (10 * 50 + 50) + (50 * 10 + 10)
    assert param_count == expected_params
    print(f"✅ Parameter count: {param_count}")


def test_ptq_int4_estimate_time(sample_model):
    """Test time estimation."""
    print("\n🧪 Testing time estimation...")
    optimizer = PTQInt4PyTorch(config={})
    estimated_time = optimizer.estimate_time(sample_model)

    assert estimated_time > 0
    print(f"✅ Estimated time: {estimated_time:.2f}s")


def test_ptq_int4_evaluate_model(sample_model, sample_dataset):
    """Test model evaluation."""
    print("\n🧪 Testing model evaluation...")
    optimizer = PTQInt4PyTorch(config={})
    accuracy = optimizer.evaluate_model(sample_model, sample_dataset)

    assert 0 <= accuracy <= 1
    print(f"✅ Accuracy: {accuracy:.4f}")


def test_ptq_int4_optimize(sample_model, sample_dataset):
    """Test PTQ INT4 optimization."""
    print("\n🧪 Testing PTQ INT4 optimization...")
    optimizer = PTQInt4PyTorch(config={})

    result = optimizer.optimize(sample_model, sample_dataset)

    # Verify result structure
    assert result.optimized_model is not None
    assert result.original_size_mb > 0
    assert result.optimized_size_mb > 0
    assert result.optimized_size_mb < result.original_size_mb
    assert result.execution_time_seconds > 0

    # Verify metrics
    print(f"   Original size: {result.original_size_mb:.4f} MB")
    print(f"   Optimized size: {result.optimized_size_mb:.4f} MB")
    print(
        f"   Size reduction: {((result.original_size_mb - result.optimized_size_mb) / result.original_size_mb * 100):.2f}%"
    )
    print(f"   Execution time: {result.execution_time_seconds:.2f}s")

    # Verify metadata
    assert "quantization_type" in result.metadata
    assert result.metadata["quantization_type"] == "post_training_dynamic"

    print("✅ Optimization test passed")


def test_ptq_int4_optimize_without_dataset(sample_model):
    """Test PTQ INT4 without dataset (should still work but with 0 accuracy)."""
    print("\n🧪 Testing PTQ INT4 without dataset...")
    optimizer = PTQInt4PyTorch(config={})

    result = optimizer.optimize(sample_model, dataset=None)

    # Should work but accuracy will be 0
    assert result.optimized_model is not None
    assert result.original_accuracy == 0.0
    assert result.optimized_accuracy == 0.0
    assert result.optimized_size_mb < result.original_size_mb

    print(
        f"   Size reduction: {((result.original_size_mb - result.optimized_size_mb) / result.original_size_mb * 100):.2f}%"
    )
    print("✅ Optimization without dataset test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
