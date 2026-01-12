"""
Unit tests for TensorFlow Pruning optimization.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.core.tensorflow.pruning import (
    PruningMagnitudeStructuredTensorFlow,
    PruningMagnitudeUnstructuredTensorFlow)


@pytest.fixture
def sample_model():
    """Create a sample TensorFlow model for testing."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


@pytest.fixture
def sample_dataset():
    """Create a sample TensorFlow dataset for testing."""
    inputs = np.random.randn(100, 10).astype(np.float32)
    labels = np.random.randint(0, 10, 100).astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset = dataset.batch(32)

    return dataset


# ============================================================================
# UNSTRUCTURED PRUNING TESTS
# ============================================================================


def test_pruning_unstructured_initialization():
    """Test unstructured pruning initialization."""
    print("\n🧪 Testing unstructured pruning initialization...")
    optimizer = PruningMagnitudeUnstructuredTensorFlow(config={})
    assert optimizer.technique_name == "pruning_magnitude_unstructured"
    assert optimizer.sparsity == 0.5  # default
    print("✅ Initialization test passed")


def test_pruning_unstructured_custom_sparsity():
    """Test custom sparsity configuration."""
    print("\n🧪 Testing custom sparsity...")
    optimizer = PruningMagnitudeUnstructuredTensorFlow(config={"sparsity": 0.7})
    assert optimizer.sparsity == 0.7
    print("✅ Custom sparsity test passed")


def test_pruning_unstructured_requires_dataset():
    """Test that pruning requires dataset."""
    print("\n🧪 Testing requires_dataset...")
    optimizer = PruningMagnitudeUnstructuredTensorFlow(config={})
    assert optimizer.requires_dataset() is True
    print("✅ Requires dataset test passed")


def test_pruning_unstructured_count_parameters(sample_model):
    """Test parameter counting."""
    print("\n🧪 Testing parameter counting...")
    optimizer = PruningMagnitudeUnstructuredTensorFlow(config={})
    param_count = optimizer.count_parameters(sample_model)

    expected_params = (10 * 50 + 50) + (50 * 10 + 10)
    assert param_count == expected_params
    print(f"✅ Parameter count: {param_count}")


def test_pruning_unstructured_estimate_time(sample_model):
    """Test time estimation."""
    print("\n🧪 Testing time estimation...")
    optimizer = PruningMagnitudeUnstructuredTensorFlow(config={})
    estimated_time = optimizer.estimate_time(sample_model)

    assert estimated_time > 0
    print(f"✅ Estimated time: {estimated_time:.2f}s")


def test_pruning_unstructured_optimize(sample_model, sample_dataset):
    """Test unstructured pruning optimization."""
    print("\n🧪 Testing unstructured pruning optimization...")
    optimizer = PruningMagnitudeUnstructuredTensorFlow(config={"sparsity": 0.5})

    result = optimizer.optimize(sample_model, sample_dataset)

    # Verify result structure
    assert result.optimized_model is not None
    assert result.original_size_mb > 0
    assert result.optimized_size_mb > 0
    assert result.optimized_size_mb < result.original_size_mb
    assert result.execution_time_seconds > 0

    print(f"   Original size: {result.original_size_mb:.4f} MB")
    print(f"   Optimized size: {result.optimized_size_mb:.4f} MB")
    print(
        f"   Size reduction: {((result.original_size_mb - result.optimized_size_mb) / result.original_size_mb * 100):.2f}%"
    )
    print(f"   Target sparsity: {optimizer.sparsity:.2%}")
    print(f"   Execution time: {result.execution_time_seconds:.2f}s")

    # Verify metadata
    assert "pruning_type" in result.metadata
    assert result.metadata["pruning_type"] == "magnitude_unstructured"
    assert result.metadata["format"] == "tflite"

    print("✅ Unstructured pruning optimization test passed")


def test_pruning_unstructured_optimize_without_dataset(sample_model):
    """Test unstructured pruning without dataset."""
    print("\n🧪 Testing unstructured pruning without dataset...")
    optimizer = PruningMagnitudeUnstructuredTensorFlow(config={})

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


# ============================================================================
# STRUCTURED PRUNING TESTS
# ============================================================================


def test_pruning_structured_initialization():
    """Test structured pruning initialization."""
    print("\n🧪 Testing structured pruning initialization...")
    optimizer = PruningMagnitudeStructuredTensorFlow(config={})
    assert optimizer.technique_name == "pruning_magnitude_structured"
    assert optimizer.sparsity == 0.3  # default (lower for structured)
    print("✅ Initialization test passed")


def test_pruning_structured_custom_sparsity():
    """Test custom sparsity configuration."""
    print("\n🧪 Testing custom sparsity...")
    optimizer = PruningMagnitudeStructuredTensorFlow(config={"sparsity": 0.4})
    assert optimizer.sparsity == 0.4
    print("✅ Custom sparsity test passed")


def test_pruning_structured_requires_dataset():
    """Test that pruning requires dataset."""
    print("\n🧪 Testing requires_dataset...")
    optimizer = PruningMagnitudeStructuredTensorFlow(config={})
    assert optimizer.requires_dataset() is True
    print("✅ Requires dataset test passed")


def test_pruning_structured_optimize(sample_model, sample_dataset):
    """Test structured pruning optimization."""
    print("\n🧪 Testing structured pruning optimization...")
    optimizer = PruningMagnitudeStructuredTensorFlow(config={"sparsity": 0.3})

    result = optimizer.optimize(sample_model, sample_dataset)

    # Verify result structure
    assert result.optimized_model is not None
    assert result.original_size_mb > 0
    assert result.optimized_size_mb > 0
    assert result.optimized_size_mb < result.original_size_mb
    assert result.execution_time_seconds > 0

    print(f"   Original size: {result.original_size_mb:.4f} MB")
    print(f"   Optimized size: {result.optimized_size_mb:.4f} MB")
    print(
        f"   Size reduction: {((result.original_size_mb - result.optimized_size_mb) / result.original_size_mb * 100):.2f}%"
    )
    print(f"   Execution time: {result.execution_time_seconds:.2f}s")

    # Verify metadata
    assert "pruning_type" in result.metadata
    assert "structured" in result.metadata["pruning_type"]
    assert result.metadata["format"] == "tflite"

    print("✅ Structured pruning optimization test passed")


def test_pruning_structured_optimize_without_dataset(sample_model):
    """Test structured pruning without dataset."""
    print("\n🧪 Testing structured pruning without dataset...")
    optimizer = PruningMagnitudeStructuredTensorFlow(config={})

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
