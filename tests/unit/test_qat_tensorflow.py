"""
Unit tests for TensorFlow Quantization-Aware Training.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.core.tensorflow.qat import QuantizationAwareTrainingTensorFlow


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


def test_qat_initialization():
    """Test QAT initialization."""
    print("\n🧪 Testing QAT initialization...")
    optimizer = QuantizationAwareTrainingTensorFlow(config={})
    assert optimizer.technique_name == "quantization_aware_training"
    assert optimizer.epochs == 3  # default
    print("✅ Initialization test passed")


def test_qat_custom_config():
    """Test custom configuration."""
    print("\n🧪 Testing custom configuration...")
    optimizer = QuantizationAwareTrainingTensorFlow(config={"epochs": 5, "learning_rate": 0.001})
    assert optimizer.epochs == 5
    assert optimizer.learning_rate == 0.001
    print("✅ Custom configuration test passed")


def test_qat_requires_dataset():
    """Test that QAT requires dataset."""
    print("\n🧪 Testing requires_dataset...")
    optimizer = QuantizationAwareTrainingTensorFlow(config={})
    assert optimizer.requires_dataset() is True
    print("✅ Requires dataset test passed")


def test_qat_count_parameters(sample_model):
    """Test parameter counting."""
    print("\n🧪 Testing parameter counting...")
    optimizer = QuantizationAwareTrainingTensorFlow(config={})
    param_count = optimizer.count_parameters(sample_model)

    expected_params = (10 * 50 + 50) + (50 * 10 + 10)
    assert param_count == expected_params
    print(f"✅ Parameter count: {param_count}")


def test_qat_estimate_time(sample_model):
    """Test time estimation."""
    print("\n🧪 Testing time estimation...")
    optimizer = QuantizationAwareTrainingTensorFlow(config={"epochs": 3})
    estimated_time = optimizer.estimate_time(sample_model)

    assert estimated_time > 0
    print(f"✅ Estimated time: {estimated_time:.2f}s")


def test_qat_optimize(sample_model, sample_dataset):
    """Test QAT optimization."""
    print("\n🧪 Testing QAT optimization...")
    optimizer = QuantizationAwareTrainingTensorFlow(
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
    assert result.metadata["format"] == "tflite"

    print("✅ QAT optimization test passed")


def test_qat_optimize_without_dataset(sample_model):
    """Test that QAT fails without dataset."""
    print("\n🧪 Testing QAT without dataset...")
    optimizer = QuantizationAwareTrainingTensorFlow(config={})

    with pytest.raises(ValueError, match="requires a dataset"):
        optimizer.optimize(sample_model, dataset=None)

    print("✅ Correctly raises error without dataset")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
