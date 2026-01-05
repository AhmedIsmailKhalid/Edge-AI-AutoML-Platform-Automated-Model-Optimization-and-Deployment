"""
Unit tests for PyTorch Quantization-Aware Training.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.core.pytorch.qat import QuantizationAwareTrainingPyTorch


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


def test_qat_initialization():
    """Test QAT initialization."""
    print("\n🧪 Testing QAT initialization...")
    optimizer = QuantizationAwareTrainingPyTorch(config={})
    assert optimizer.technique_name == "quantization_aware_training"
    assert optimizer.epochs == 3  # default
    print("✅ Initialization test passed")


def test_qat_custom_config():
    """Test custom configuration."""
    print("\n🧪 Testing custom configuration...")
    optimizer = QuantizationAwareTrainingPyTorch(config={"epochs": 5, "learning_rate": 0.001})
    assert optimizer.epochs == 5
    assert optimizer.learning_rate == 0.001
    print("✅ Custom configuration test passed")


def test_qat_requires_dataset():
    """Test that QAT requires dataset."""
    print("\n🧪 Testing requires_dataset...")
    optimizer = QuantizationAwareTrainingPyTorch(config={})
    assert optimizer.requires_dataset() is True
    print("✅ Requires dataset test passed")


def test_qat_count_parameters(sample_model):
    """Test parameter counting."""
    print("\n🧪 Testing parameter counting...")
    optimizer = QuantizationAwareTrainingPyTorch(config={})
    param_count = optimizer.count_parameters(sample_model)

    expected_params = (10 * 50 + 50) + (50 * 10 + 10)
    assert param_count == expected_params
    print(f"✅ Parameter count: {param_count}")


def test_qat_estimate_time(sample_model):
    """Test time estimation."""
    print("\n🧪 Testing time estimation...")
    optimizer = QuantizationAwareTrainingPyTorch(config={"epochs": 3})
    estimated_time = optimizer.estimate_time(sample_model)

    assert estimated_time > 0
    print(f"✅ Estimated time: {estimated_time:.2f}s")


def test_qat_optimize(sample_model, sample_dataset):
    """Test QAT optimization."""
    print("\n🧪 Testing QAT optimization...")
    optimizer = QuantizationAwareTrainingPyTorch(
        config={"epochs": 2}  # Fewer epochs for faster testing
    )

    result = optimizer.optimize(sample_model, sample_dataset)

    # Verify result structure
    assert result.optimized_model is not None
    assert result.original_size_mb > 0
    assert result.optimized_size_mb > 0
    assert result.optimized_size_mb < result.original_size_mb
    assert result.execution_time_seconds > 0

    print(f"   Original size: {result.original_size_mb:.4f} MB")
    print(f"   Quantized size: {result.optimized_size_mb:.4f} MB")
    print(
        f"   Size reduction: {((result.original_size_mb - result.optimized_size_mb) / result.original_size_mb * 100):.2f}%"
    )
    print(f"   Execution time: {result.execution_time_seconds:.2f}s")

    # Verify metadata
    assert "technique" in result.metadata
    assert result.metadata["technique"] == "quantization_aware_training"

    print("✅ QAT optimization test passed")


def test_qat_optimize_without_dataset(sample_model):
    """Test that QAT fails without dataset."""
    print("\n🧪 Testing QAT without dataset...")
    optimizer = QuantizationAwareTrainingPyTorch(config={})

    with pytest.raises(ValueError, match="requires a dataset"):
        optimizer.optimize(sample_model, dataset=None)

    print("✅ Correctly raises error without dataset")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
