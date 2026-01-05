"""
Post-Training Quantization (INT4) for TensorFlow models.
"""

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.core.base import BaseOptimization, OptimizationResult


class PTQInt4TensorFlow(BaseOptimization):
    """
    Post-Training Quantization to INT4 for TensorFlow models.
    Uses TFLite converter with INT4 quantization.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize PTQ INT4 optimizer.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.technique_name = "ptq_int4"

    def requires_dataset(self) -> bool:
        """PTQ INT4 requires representative dataset for calibration."""
        return True

    def estimate_time(self, model: Any) -> float:
        """
        Estimate execution time based on model size.

        Args:
            model: TensorFlow model

        Returns:
            Estimated time in seconds
        """
        param_count = self.count_parameters(model)
        # INT4 quantization is slightly faster than INT8
        return max(5.0, param_count / 600_000)

    def count_parameters(self, model: tf.keras.Model) -> int:
        """
        Count total trainable parameters in TensorFlow model.

        Args:
            model: TensorFlow model

        Returns:
            Total number of parameters
        """
        return int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))

    def evaluate_model(self, model: tf.keras.Model, dataset: Any, **kwargs) -> float:
        """
        Evaluate TensorFlow model accuracy on dataset.

        Args:
            model: TensorFlow model
            dataset: tf.data.Dataset for evaluation

        Returns:
            Accuracy as float between 0 and 1
        """
        results = model.evaluate(dataset, verbose=0)
        # results is [loss, accuracy] for compiled models
        if isinstance(results, list) and len(results) > 1:
            return float(results[1])  # accuracy
        return 0.0

    def representative_dataset_gen(self, dataset: Any, num_samples: int = 100):
        """
        Generator for representative dataset samples for calibration.

        Args:
            dataset: tf.data.Dataset
            num_samples: Number of samples to use

        Yields:
            Single batch of data
        """
        count = 0
        for data in dataset.take(num_samples):
            if isinstance(data, tuple):
                # (inputs, labels) format
                yield [data[0]]
            else:
                # Just inputs
                yield [data]
            count += 1
            if count >= num_samples:
                break

    def optimize(
        self, model: tf.keras.Model, dataset: Any | None = None, **kwargs
    ) -> OptimizationResult:
        """
        Apply INT4 quantization to TensorFlow model using TFLite.

        Args:
            model: TensorFlow model to optimize
            dataset: Representative dataset for calibration
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with quantized model and metrics
        """
        start_time = time.time()

        # Save original model to get size (use .keras extension for Keras 3.x)
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
            original_path = Path(tmp_file.name)
            model.save(original_path)

        # Get original size
        original_size_mb = self.get_model_size_mb(original_path)
        original_params = self.count_parameters(model)

        # Evaluate original accuracy if dataset provided
        if dataset is not None:
            original_accuracy = self.evaluate_model(model, dataset)
        else:
            original_accuracy = 0.0

        # Convert to TFLite with INT4 quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Set INT4 quantization (experimental feature in TFLite)
        if dataset is not None:
            converter.representative_dataset = lambda: self.representative_dataset_gen(dataset)
            # INT4 quantization for weights (experimental)
            # Note: Full INT4 support is limited in TFLite, this uses mixed precision
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS,
            ]
            # Enable experimental INT4 weights
            converter._experimental_low_bit_qat = True

        # Convert model
        quantized_tflite_model = converter.convert()

        # Save quantized model to get size
        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as tmp_quantized:
            quantized_path = Path(tmp_quantized.name)
            quantized_path.write_bytes(quantized_tflite_model)

        optimized_size_mb = self.get_model_size_mb(quantized_path)

        # Load quantized model for evaluation
        interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
        interpreter.allocate_tensors()

        # Evaluate quantized model accuracy if dataset provided
        if dataset is not None:
            optimized_accuracy = self._evaluate_tflite_model(interpreter, dataset)
        else:
            optimized_accuracy = 0.0

        # Clean up temporary files
        original_path.unlink()
        quantized_path.unlink()

        execution_time = time.time() - start_time

        # Store quantized model bytes in metadata for later saving
        result = OptimizationResult(
            optimized_model=quantized_tflite_model,  # Store bytes
            original_accuracy=original_accuracy,
            optimized_accuracy=optimized_accuracy,
            original_size_mb=original_size_mb,
            optimized_size_mb=optimized_size_mb,
            original_params_count=original_params,
            optimized_params_count=original_params,  # TFLite doesn't expose param count easily
            execution_time_seconds=execution_time,
            metadata={
                "quantization_type": "post_training",
                "dtype": "int4_experimental",
                "format": "tflite",
                "note": "INT4 weight quantization with mixed precision activations",
            },
            technique_name="ptq_int4",
        )

        return result

    def _evaluate_tflite_model(self, interpreter: tf.lite.Interpreter, dataset: Any) -> float:
        """
        Evaluate TFLite model accuracy.

        Args:
            interpreter: TFLite interpreter
            dataset: Evaluation dataset

        Returns:
            Accuracy as float
        """
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        correct = 0
        total = 0

        for data in dataset:
            if isinstance(data, tuple):
                inputs, labels = data
            else:
                continue  # Skip if no labels

            # Process batch
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
