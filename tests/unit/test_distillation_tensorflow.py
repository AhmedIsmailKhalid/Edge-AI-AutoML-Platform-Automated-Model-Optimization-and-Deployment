"""
Unit tests for TensorFlow Knowledge Distillation optimization.
"""

import numpy as np
import pytest
import tensorflow as tf

from src.core.tensorflow.distillation import KnowledgeDistillationTensorFlow


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


def test_distillation_initialization():
    """Test knowledge distillation initialization."""
    print("\n🧪 Testing knowledge distillation initialization...")
    optimizer = KnowledgeDistillationTensorFlow(config={})
    assert optimizer.technique_name == "knowledge_distillation"
    assert optimizer.student_scale == 0.5  # default
    assert optimizer.temperature == 3.0  # default
    assert optimizer.alpha == 0.7  # default
    print("✅ Initialization test passed")


def test_distillation_custom_config():
    """Test custom configuration."""
    print("\n🧪 Testing custom configuration...")
    optimizer = KnowledgeDistillationTensorFlow(
        config={"student_scale": 0.3, "temperature": 5.0, "alpha": 0.8, "epochs": 3}
    )
    assert optimizer.student_scale == 0.3
    assert optimizer.temperature == 5.0
    assert optimizer.alpha == 0.8
    assert optimizer.epochs == 3
    print("✅ Custom configuration test passed")


def test_distillation_requires_dataset():
    """Test that distillation requires dataset."""
    print("\n🧪 Testing requires_dataset...")
    optimizer = KnowledgeDistillationTensorFlow(config={})
    assert optimizer.requires_dataset() is True
    print("✅ Requires dataset test passed")


def test_distillation_count_parameters(sample_model):
    """Test parameter counting."""
    print("\n🧪 Testing parameter counting...")
    optimizer = KnowledgeDistillationTensorFlow(config={})
    param_count = optimizer.count_parameters(sample_model)

    expected_params = (10 * 50 + 50) + (50 * 10 + 10)
    assert param_count == expected_params
    print(f"✅ Parameter count: {param_count}")


def test_distillation_estimate_time(sample_model):
    """Test time estimation."""
    print("\n🧪 Testing time estimation...")
    optimizer = KnowledgeDistillationTensorFlow(config={"epochs": 5})
    estimated_time = optimizer.estimate_time(sample_model)

    assert estimated_time > 0
    print(f"✅ Estimated time: {estimated_time:.2f}s")


def test_distillation_create_student(sample_model):
    """Test student model creation."""
    print("\n🧪 Testing student model creation...")
    optimizer = KnowledgeDistillationTensorFlow(config={"student_scale": 0.5})
    student_model = optimizer.create_student_model(sample_model)

    # Student should have fewer parameters
    teacher_params = optimizer.count_parameters(sample_model)
    student_params = optimizer.count_parameters(student_model)

    assert student_params < teacher_params
    print(f"   Teacher params: {teacher_params}")
    print(f"   Student params: {student_params}")
    print(f"   Reduction: {((teacher_params - student_params) / teacher_params * 100):.2f}%")
    print("✅ Student model creation test passed")


def test_distillation_optimize(sample_model, sample_dataset):
    """Test knowledge distillation optimization."""
    print("\n🧪 Testing knowledge distillation optimization...")
    optimizer = KnowledgeDistillationTensorFlow(
        config={"student_scale": 0.5, "epochs": 2}  # Fewer epochs for faster testing
    )

    result = optimizer.optimize(sample_model, sample_dataset)

    # Verify result structure
    assert result.optimized_model is not None
    assert result.original_size_mb > 0
    assert result.optimized_size_mb > 0
    assert result.optimized_size_mb < result.original_size_mb
    assert result.execution_time_seconds > 0

    # Verify parameter reduction
    assert result.optimized_params_count < result.original_params_count

    print(f"   Teacher size: {result.original_size_mb:.4f} MB")
    print(f"   Student size: {result.optimized_size_mb:.4f} MB")
    print(
        f"   Size reduction: {((result.original_size_mb - result.optimized_size_mb) / result.original_size_mb * 100):.2f}%"
    )
    print(f"   Teacher params: {result.original_params_count}")
    print(f"   Student params: {result.optimized_params_count}")
    print(
        f"   Param reduction: {((result.original_params_count - result.optimized_params_count) / result.original_params_count * 100):.2f}%"
    )
    print(f"   Execution time: {result.execution_time_seconds:.2f}s")

    # Verify metadata
    assert "technique" in result.metadata
    assert result.metadata["technique"] == "knowledge_distillation"
    assert result.metadata["format"] == "tflite"

    print("✅ Knowledge distillation optimization test passed")


def test_distillation_optimize_without_dataset(sample_model):
    """Test that distillation fails without dataset."""
    print("\n🧪 Testing distillation without dataset...")
    optimizer = KnowledgeDistillationTensorFlow(config={})

    with pytest.raises(ValueError, match="requires a dataset"):
        optimizer.optimize(sample_model, dataset=None)

    print("✅ Correctly raises error without dataset")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
