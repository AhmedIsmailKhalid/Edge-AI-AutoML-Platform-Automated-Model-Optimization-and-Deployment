"""
Hybrid Optimization (Pruning + Quantization) for TensorFlow models.
"""

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.core.base import BaseOptimization, OptimizationResult


class HybridPruneQuantizeTensorFlow(BaseOptimization):
    """
    Hybrid optimization combining pruning and quantization.
    First prunes the model, then applies quantization for maximum compression.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize hybrid optimizer.

        Args:
            config: Configuration dictionary
                - sparsity: Pruning sparsity (default: 0.5 = 50%)
                - fine_tune_epochs: Epochs for fine-tuning after pruning (default: 2)
        """
        super().__init__(config)
        self.technique_name = "hybrid_prune_quantize"
        self.sparsity = config.get("sparsity", 0.5)
        self.fine_tune_epochs = config.get("fine_tune_epochs", 2)

    def requires_dataset(self) -> bool:
        """Hybrid requires dataset for fine-tuning."""
        return True

    def estimate_time(self, model: Any) -> float:
        """
        Estimate execution time.

        Args:
            model: TensorFlow model

        Returns:
            Estimated time in seconds
        """
        param_count = self.count_parameters(model)
        return max(8.0, (param_count / 800_000) * self.fine_tune_epochs)

    def count_parameters(self, model: tf.keras.Model) -> int:
        """Count total trainable parameters."""
        return int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))

    def evaluate_model(self, model: tf.keras.Model, dataset: Any, **kwargs) -> float:
        """Evaluate TensorFlow model accuracy."""
        results = model.evaluate(dataset, verbose=0)
        if isinstance(results, list) and len(results) > 1:
            return float(results[1])  # accuracy
        return 0.0

    def apply_magnitude_pruning(self, model: tf.keras.Model, sparsity: float) -> tf.keras.Model:
        """
        Apply magnitude-based pruning to model weights.

        Args:
            model: TensorFlow model
            sparsity: Target sparsity

        Returns:
            Pruned model
        """
        import copy

        pruned_model = copy.deepcopy(model)

        for layer in pruned_model.layers:
            if hasattr(layer, "kernel"):
                weights = layer.get_weights()
                if len(weights) > 0:
                    kernel = weights[0]

                    # Calculate threshold for pruning
                    threshold = np.percentile(np.abs(kernel), sparsity * 100)

                    # Create mask: set weights below threshold to zero
                    mask = np.abs(kernel) >= threshold
                    pruned_kernel = kernel * mask

                    # Set pruned weights back
                    weights[0] = pruned_kernel
                    layer.set_weights(weights)

        return pruned_model

    def optimize(
        self, model: tf.keras.Model, dataset: Any | None = None, **kwargs
    ) -> OptimizationResult:
        """
        Apply hybrid pruning + quantization.

        Args:
            model: TensorFlow model to optimize
            dataset: Dataset for fine-tuning and evaluation
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with hybrid optimized model and metrics
        """
        start_time = time.time()

        if dataset is None:
            raise ValueError("Hybrid optimization requires a dataset for fine-tuning")

        # Save original model to get size
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
            original_path = Path(tmp_file.name)
            model.save(original_path)

        # Get original metrics
        original_size_mb = self.get_model_size_mb(original_path)
        original_params = self.count_parameters(model)
        original_accuracy = self.evaluate_model(model, dataset)

        # Step 1: Apply magnitude-based pruning
        pruned_model = self.apply_magnitude_pruning(model, self.sparsity)

        # Step 2: Compile and fine-tune pruned model
        pruned_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        pruned_model.fit(dataset, epochs=self.fine_tune_epochs, verbose=0)

        # Step 3: Apply quantization via TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Use representative dataset for better quantization
        def representative_dataset_gen():
            for data in dataset.take(100):
                if isinstance(data, tuple):
                    yield [data[0]]
                else:
                    yield [data]

        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        # Convert to quantized TFLite
        hybrid_tflite_model = converter.convert()

        # Save hybrid model to get size
        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as tmp_hybrid:
            hybrid_path = Path(tmp_hybrid.name)
            hybrid_path.write_bytes(hybrid_tflite_model)

        optimized_size_mb = self.get_model_size_mb(hybrid_path)

        # Load hybrid model for evaluation
        interpreter = tf.lite.Interpreter(model_content=hybrid_tflite_model)
        interpreter.allocate_tensors()

        # Evaluate final accuracy
        optimized_accuracy = self._evaluate_tflite_model(interpreter, dataset)

        # Clean up temporary files
        original_path.unlink()
        hybrid_path.unlink()

        execution_time = time.time() - start_time

        result = OptimizationResult(
            optimized_model=hybrid_tflite_model,
            original_accuracy=original_accuracy,
            optimized_accuracy=optimized_accuracy,
            original_size_mb=original_size_mb,
            optimized_size_mb=optimized_size_mb,
            original_params_count=original_params,
            optimized_params_count=original_params,
            execution_time_seconds=execution_time,
            metadata={
                "technique": "hybrid_prune_quantize",
                "pruning_sparsity": self.sparsity,
                "fine_tune_epochs": self.fine_tune_epochs,
                "quantization_dtype": "int8",
                "format": "tflite",
                "pipeline": "prune -> fine-tune -> quantize",
            },
            technique_name="hyrbrid_prune_quantize",
        )

        return result

    def _evaluate_tflite_model(self, interpreter: tf.lite.Interpreter, dataset: Any) -> float:
        """Evaluate TFLite model accuracy."""
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        correct = 0
        total = 0

        for data in dataset:
            if isinstance(data, tuple):
                inputs, labels = data
            else:
                continue

            for i in range(inputs.shape[0]):
                input_data = np.expand_dims(inputs[i], axis=0).astype(input_details[0]["dtype"])
                interpreter.set_tensor(input_details[0]["index"], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]["index"])

                predicted = np.argmax(output_data)
                actual = labels[i].numpy() if hasattr(labels[i], "numpy") else labels[i]

                if predicted == actual:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0
