"""
Unit tests for TensorFlow PTQ INT8 optimization.
"""

import numpy as np
import tensorflow as tf

from src.core.tensorflow.ptq_int8 import PTQInt8TensorFlow


def create_simple_model():
    """Create a simple TensorFlow model for testing."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def create_dummy_dataset(num_samples=100, batch_size=10):
    """Create a dummy dataset for testing."""
    X = np.random.randn(num_samples, 10).astype(np.float32)
    y = np.random.randint(0, 10, size=(num_samples,))

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size)

    return dataset


def test_ptq_int8_initialization():
    """Test PTQ INT8 optimizer initialization."""
    config = {"test": "value"}
    optimizer = PTQInt8TensorFlow(config)

    assert optimizer.technique_name == "ptq_int8"
    assert optimizer.config == config
    assert optimizer.requires_dataset()  # TensorFlow PTQ requires dataset
    print("✅ Initialization test passed")


def test_ptq_int8_count_parameters():
    """Test parameter counting."""
    model = create_simple_model()
    optimizer = PTQInt8TensorFlow({})

    param_count = optimizer.count_parameters(model)

    assert param_count > 0
    print(f"✅ Parameter count: {param_count:,}")


def test_ptq_int8_estimate_time():
    """Test time estimation."""
    model = create_simple_model()
    optimizer = PTQInt8TensorFlow({})

    estimated_time = optimizer.estimate_time(model)

    assert estimated_time > 0
    print(f"✅ Estimated time: {estimated_time:.2f} seconds")


def test_ptq_int8_evaluate_model():
    """Test model evaluation."""
    model = create_simple_model()
    dataset = create_dummy_dataset()
    optimizer = PTQInt8TensorFlow({})

    accuracy = optimizer.evaluate_model(model, dataset)

    assert 0.0 <= accuracy <= 1.0
    print(f"✅ Model accuracy: {accuracy:.4f}")


def test_ptq_int8_optimize():
    """Test PTQ INT8 optimization."""
    print("\n🧪 Testing PTQ INT8 optimization...")

    # Create simple model
    model = create_simple_model()

    # Create dummy dataset
    dataset = create_dummy_dataset()

    # Run optimization
    optimizer = PTQInt8TensorFlow({})
    result = optimizer.optimize(model, dataset)

    # Verify result
    assert result.optimized_model is not None
    assert isinstance(result.optimized_model, bytes)  # TFLite model is bytes
    assert result.original_size_mb > 0
    assert result.optimized_size_mb > 0
    assert result.optimized_size_mb < result.original_size_mb  # Should be smaller
    assert result.execution_time_seconds > 0
    assert 0.0 <= result.original_accuracy <= 1.0
    assert 0.0 <= result.optimized_accuracy <= 1.0

    compression_ratio = result.original_size_mb / result.optimized_size_mb

    print(f"\n✅ Original size: {result.original_size_mb:.2f} MB")
    print(f"✅ Optimized size: {result.optimized_size_mb:.2f} MB")
    print(f"✅ Compression: {compression_ratio:.2f}x")
    print(f"✅ Original accuracy: {result.original_accuracy:.4f}")
    print(f"✅ Optimized accuracy: {result.optimized_accuracy:.4f}")
    print(f"✅ Execution time: {result.execution_time_seconds:.2f}s")

    # Verify reasonable compression (should be ~4x for INT8)
    assert compression_ratio >= 2.0, "Compression ratio should be at least 2x"

    print("✅ PTQ INT8 optimization test passed!")


def test_ptq_int8_optimize_without_dataset():
    """Test PTQ INT8 optimization without dataset (should still work but with default accuracy)."""
    print("\n🧪 Testing PTQ INT8 without dataset...")

    model = create_simple_model()
    optimizer = PTQInt8TensorFlow({})

    # This should work but produce 0.0 accuracy values
    result = optimizer.optimize(model, dataset=None)

    assert result.optimized_model is not None
    assert result.original_accuracy == 0.0
    assert result.optimized_accuracy == 0.0
    print("✅ Optimization without dataset test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("TENSORFLOW PTQ INT8 TEST SUITE")
    print("=" * 60)

    try:
        test_ptq_int8_initialization()
        test_ptq_int8_count_parameters()
        test_ptq_int8_estimate_time()
        test_ptq_int8_evaluate_model()
        test_ptq_int8_optimize()
        test_ptq_int8_optimize_without_dataset()

        print("\n" + "=" * 60)
        print("✅ ALL TENSORFLOW PTQ INT8 TESTS PASSED!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
