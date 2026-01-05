"""
Magnitude-based Pruning for TensorFlow models.
Uses TFLite optimization without tensorflow-model-optimization package.
"""

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.core.base import BaseOptimization, OptimizationResult


class PruningMagnitudeUnstructuredTensorFlow(BaseOptimization):
    """
    Magnitude-based unstructured pruning for TensorFlow models.
    Uses custom pruning implementation + TFLite optimization.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize pruning optimizer.

        Args:
            config: Configuration dictionary
                - sparsity: Target sparsity level (default: 0.5 = 50%)
        """
        super().__init__(config)
        self.technique_name = "pruning_magnitude_unstructured"
        self.sparsity = config.get("sparsity", 0.5)

    def requires_dataset(self) -> bool:
        """Pruning requires dataset for evaluation."""
        return True

    def estimate_time(self, model: Any) -> float:
        """Estimate execution time based on model size."""
        param_count = self.count_parameters(model)
        return max(5.0, param_count / 1_500_000)

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
            sparsity: Target sparsity (0.5 = 50% of weights set to zero)

        Returns:
            Pruned model
        """
        import copy

        pruned_model = copy.deepcopy(model)

        for layer in pruned_model.layers:
            if hasattr(layer, "kernel"):
                # Get weights
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
        Apply magnitude-based unstructured pruning.

        Args:
            model: TensorFlow model to optimize
            dataset: Dataset for evaluation
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with pruned model and metrics
        """
        start_time = time.time()

        # Save original model to get size
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
            original_path = Path(tmp_file.name)
            model.save(original_path)

        # Get original metrics
        original_size_mb = self.get_model_size_mb(original_path)
        original_params = self.count_parameters(model)

        # Evaluate original accuracy if dataset provided
        if dataset is not None:
            original_accuracy = self.evaluate_model(model, dataset)
        else:
            original_accuracy = 0.0

        # Apply magnitude-based pruning
        pruned_model = self.apply_magnitude_pruning(model, self.sparsity)

        # Compile pruned model
        pruned_model.compile(
            optimizer="adam",
            loss=model.loss if hasattr(model, "loss") else "sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Convert to TFLite with optimization
        converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        pruned_tflite_model = converter.convert()

        # Save pruned model to get size
        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as tmp_pruned:
            pruned_path = Path(tmp_pruned.name)
            pruned_path.write_bytes(pruned_tflite_model)

        optimized_size_mb = self.get_model_size_mb(pruned_path)

        # Load pruned model for evaluation
        interpreter = tf.lite.Interpreter(model_content=pruned_tflite_model)
        interpreter.allocate_tensors()

        # Evaluate pruned accuracy if dataset provided
        if dataset is not None:
            optimized_accuracy = self._evaluate_tflite_model(interpreter, dataset)
        else:
            optimized_accuracy = 0.0

        # Clean up temporary files
        original_path.unlink()
        pruned_path.unlink()

        execution_time = time.time() - start_time

        result = OptimizationResult(
            optimized_model=pruned_tflite_model,
            original_accuracy=original_accuracy,
            optimized_accuracy=optimized_accuracy,
            original_size_mb=original_size_mb,
            optimized_size_mb=optimized_size_mb,
            original_params_count=original_params,
            optimized_params_count=original_params,
            execution_time_seconds=execution_time,
            metadata={
                "pruning_type": "magnitude_unstructured",
                "target_sparsity": self.sparsity,
                "format": "tflite",
                "implementation": "custom_magnitude_pruning",
            },
            technique_name="pruning_magnitude_unstructured",
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


class PruningMagnitudeStructuredTensorFlow(BaseOptimization):
    """
    Magnitude-based structured pruning for TensorFlow models.
    Since we can't change layer dimensions in-place, we use unstructured pruning
    with higher sparsity and let TFLite optimization compress it.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize structured pruning optimizer.

        Args:
            config: Configuration dictionary
                - sparsity: Target sparsity level (default: 0.3 = 30%)
        """
        super().__init__(config)
        self.technique_name = "pruning_magnitude_structured"
        self.sparsity = config.get("sparsity", 0.3)

    def requires_dataset(self) -> bool:
        """Pruning requires dataset for evaluation."""
        return True

    def estimate_time(self, model: Any) -> float:
        """Estimate execution time."""
        param_count = self.count_parameters(model)
        return max(5.0, param_count / 1_500_000)

    def count_parameters(self, model: tf.keras.Model) -> int:
        """Count total trainable parameters."""
        return int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))

    def evaluate_model(self, model: tf.keras.Model, dataset: Any, **kwargs) -> float:
        """Evaluate TensorFlow model accuracy."""
        results = model.evaluate(dataset, verbose=0)
        if isinstance(results, list) and len(results) > 1:
            return float(results[1])
        return 0.0

    def apply_channel_wise_pruning(self, model: tf.keras.Model, sparsity: float) -> tf.keras.Model:
        """
        Apply channel-wise magnitude pruning.
        Prunes entire channels/filters with low L1 norm.

        Note: Since Keras doesn't allow changing layer shapes in-place,
        we apply block-wise pruning that approximates structured pruning.

        Args:
            model: TensorFlow model
            sparsity: Target sparsity

        Returns:
            Pruned model
        """
        import copy

        pruned_model = copy.deepcopy(model)

        for layer in pruned_model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D | tf.keras.layers.Dense):
                weights = layer.get_weights()
                if len(weights) > 0:
                    kernel = weights[0]

                    # Apply channel-wise pruning based on L1 norm
                    # For Conv2D: kernel shape is (height, width, in_channels, out_channels)
                    # For Dense: kernel shape is (in_features, out_features)

                    if len(kernel.shape) == 4:  # Conv2D
                        # Calculate L1 norm for each output channel
                        channel_norms = np.sum(np.abs(kernel), axis=(0, 1, 2))
                        threshold = np.percentile(channel_norms, sparsity * 100)

                        # Zero out entire channels below threshold
                        for ch_idx in range(kernel.shape[3]):
                            if channel_norms[ch_idx] < threshold:
                                kernel[:, :, :, ch_idx] = 0

                    else:  # Dense
                        # Calculate L1 norm for each output neuron
                        neuron_norms = np.sum(np.abs(kernel), axis=0)
                        threshold = np.percentile(neuron_norms, sparsity * 100)

                        # Zero out entire neurons below threshold
                        for neuron_idx in range(kernel.shape[1]):
                            if neuron_norms[neuron_idx] < threshold:
                                kernel[:, neuron_idx] = 0

                    # Set pruned weights
                    weights[0] = kernel
                    layer.set_weights(weights)

        return pruned_model

    def optimize(
        self, model: tf.keras.Model, dataset: Any | None = None, **kwargs
    ) -> OptimizationResult:
        """
        Apply magnitude-based structured pruning.

        Args:
            model: TensorFlow model to optimize
            dataset: Dataset for evaluation
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with pruned model and metrics
        """
        start_time = time.time()

        # Save original model to get size
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
            original_path = Path(tmp_file.name)
            model.save(original_path)

        # Get original metrics
        original_size_mb = self.get_model_size_mb(original_path)
        original_params = self.count_parameters(model)

        # Evaluate original accuracy if dataset provided
        if dataset is not None:
            original_accuracy = self.evaluate_model(model, dataset)
        else:
            original_accuracy = 0.0

        # Apply channel-wise pruning
        pruned_model = self.apply_channel_wise_pruning(model, self.sparsity)

        # Compile pruned model
        pruned_model.compile(
            optimizer="adam",
            loss=model.loss if hasattr(model, "loss") else "sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Convert to TFLite with optimization
        converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        pruned_tflite_model = converter.convert()

        # Save pruned model to get size
        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as tmp_pruned:
            pruned_path = Path(tmp_pruned.name)
            pruned_path.write_bytes(pruned_tflite_model)

        optimized_size_mb = self.get_model_size_mb(pruned_path)

        # Load pruned model for evaluation
        interpreter = tf.lite.Interpreter(model_content=pruned_tflite_model)
        interpreter.allocate_tensors()

        # Evaluate pruned accuracy if dataset provided
        if dataset is not None:
            optimized_accuracy = self._evaluate_tflite_model(interpreter, dataset)
        else:
            optimized_accuracy = 0.0

        # Clean up temporary files
        original_path.unlink()
        pruned_path.unlink()

        execution_time = time.time() - start_time

        result = OptimizationResult(
            optimized_model=pruned_tflite_model,
            original_accuracy=original_accuracy,
            optimized_accuracy=optimized_accuracy,
            original_size_mb=original_size_mb,
            optimized_size_mb=optimized_size_mb,
            original_params_count=original_params,
            optimized_params_count=original_params,
            execution_time_seconds=execution_time,
            metadata={
                "pruning_type": "magnitude_structured_channelwise",
                "target_sparsity": self.sparsity,
                "format": "tflite",
                "implementation": "channel_wise_pruning",
                "note": "Zeros out entire channels/filters based on L1 norm",
            },
            technique_name="pruning_magnitude_structured",
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
