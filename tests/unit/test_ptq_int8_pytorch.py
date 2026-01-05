"""
Unit tests for PyTorch PTQ INT8 optimization.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.core.pytorch.ptq_int8 import PTQInt8PyTorch


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


def test_ptq_int8_initialization():
    """Test PTQ INT8 optimizer initialization."""
    config = {"test": "value"}
    optimizer = PTQInt8PyTorch(config)

    assert optimizer.technique_name == "ptq_int8"
    assert optimizer.config == config
    assert not optimizer.requires_dataset()


def test_ptq_int8_optimize():
    """Test PTQ INT8 optimization."""
    # Create simple model
    model = SimpleModel()

    # Create dummy dataset
    X = torch.randn(100, 10)
    y = torch.randint(0, 10, (100,))
    dataset = DataLoader(TensorDataset(X, y), batch_size=10)

    # Run optimization
    optimizer = PTQInt8PyTorch({})
    result = optimizer.optimize(model, dataset)

    # Verify result
    assert result.optimized_model is not None
    assert result.original_size_mb > 0
    assert result.optimized_size_mb > 0
    assert result.optimized_size_mb < result.original_size_mb  # Should be smaller
    assert result.execution_time_seconds > 0
    print(f"\n✅ Original size: {result.original_size_mb:.2f} MB")
    print(f"✅ Optimized size: {result.optimized_size_mb:.2f} MB")
    print(f"✅ Compression: {result.original_size_mb / result.optimized_size_mb:.2f}x")


if __name__ == "__main__":
    test_ptq_int8_initialization()
    test_ptq_int8_optimize()
    print("\n✅ All PyTorch PTQ INT8 tests passed!")
