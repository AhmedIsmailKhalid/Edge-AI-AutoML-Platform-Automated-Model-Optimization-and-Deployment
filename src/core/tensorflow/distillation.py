"""
Knowledge Distillation for TensorFlow models.
"""

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from src.core.base import BaseOptimization, OptimizationResult


class KnowledgeDistillationTensorFlow(BaseOptimization):
    """
    Knowledge Distillation for TensorFlow models.
    Trains a smaller student model to mimic a larger teacher model.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize knowledge distillation optimizer.

        Args:
            config: Configuration dictionary
                - student_scale: Scale factor for student model size (default: 0.5)
                - temperature: Distillation temperature (default: 3.0)
                - alpha: Weight for distillation loss (default: 0.7)
                - epochs: Training epochs (default: 5)
        """
        super().__init__(config)
        self.technique_name = "knowledge_distillation"
        self.student_scale = config.get("student_scale", 0.5)
        self.temperature = config.get("temperature", 3.0)
        self.alpha = config.get("alpha", 0.7)
        self.epochs = config.get("epochs", 5)

    def requires_dataset(self) -> bool:
        """Knowledge distillation requires dataset for training."""
        return True

    def estimate_time(self, model: Any) -> float:
        """
        Estimate execution time based on model size and training epochs.

        Args:
            model: TensorFlow model

        Returns:
            Estimated time in seconds
        """
        param_count = self.count_parameters(model)
        return max(10.0, (param_count / 400_000) * self.epochs)

    def count_parameters(self, model: tf.keras.Model) -> int:
        """Count total trainable parameters."""
        return int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))

    def evaluate_model(self, model: tf.keras.Model, dataset: Any, **kwargs) -> float:
        """Evaluate TensorFlow model accuracy."""
        results = model.evaluate(dataset, verbose=0)
        if isinstance(results, list) and len(results) > 1:
            return float(results[1])  # accuracy
        return 0.0

    def create_student_model(self, teacher_model: tf.keras.Model) -> tf.keras.Model:
        """
        Create a smaller student model based on teacher architecture.

        Args:
            teacher_model: Teacher model

        Returns:
            Student model with reduced capacity
        """
        student_layers = []

        for layer in teacher_model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                student_layers.append(layer)

            elif isinstance(layer, tf.keras.layers.Dense):
                units = layer.units
                activation = layer.activation

                # Don't scale the output layer (classification head)
                if units <= 20:  # Likely output layer
                    student_layers.append(tf.keras.layers.Dense(units, activation=activation))
                else:
                    student_layers.append(
                        tf.keras.layers.Dense(
                            int(units * self.student_scale), activation=activation
                        )
                    )

            elif isinstance(layer, tf.keras.layers.Conv2D):
                filters = layer.filters
                kernel_size = layer.kernel_size
                strides = layer.strides
                padding = layer.padding
                activation = layer.activation

                student_layers.append(
                    tf.keras.layers.Conv2D(
                        int(filters * self.student_scale),
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        activation=activation,
                    )
                )

            elif isinstance(
                layer,
                tf.keras.layers.Dropout
                | tf.keras.layers.Flatten
                | tf.keras.layers.MaxPooling2D
                | tf.keras.layers.BatchNormalization,
            ):
                # Keep regularization/pooling layers as is
                student_layers.append(layer.__class__.from_config(layer.get_config()))

        return tf.keras.Sequential(student_layers)

    def distillation_loss(
        self,
        student_outputs: tf.Tensor,
        teacher_outputs: tf.Tensor,
        labels: tf.Tensor,
        temperature: float,
        alpha: float,
    ) -> tf.Tensor:
        """
        Calculate distillation loss.

        Args:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs
            labels: Ground truth labels
            temperature: Temperature for softening distributions
            alpha: Weight for distillation loss

        Returns:
            Combined loss
        """
        # Soft targets loss (KL divergence)
        soft_student = tf.nn.softmax(student_outputs / temperature)
        soft_teacher = tf.nn.softmax(teacher_outputs / temperature)

        soft_targets_loss = tf.reduce_mean(
            tf.keras.losses.kullback_leibler_divergence(soft_teacher, soft_student)
        ) * (temperature**2)

        # Hard targets loss (cross entropy)
        hard_targets_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, student_outputs, from_logits=True
        )
        hard_targets_loss = tf.reduce_mean(hard_targets_loss)

        # Combined loss
        return alpha * soft_targets_loss + (1 - alpha) * hard_targets_loss

    class DistillationTrainer(tf.keras.Model):
        """Custom training loop for distillation."""

        def __init__(self, student, teacher, temperature, alpha):
            super().__init__()
            self.student = student
            self.teacher = teacher
            self.temperature = temperature
            self.alpha = alpha

        def call(self, x):
            return self.student(x)

        def train_step(self, data):
            x, y = data

            # Get teacher predictions
            teacher_predictions = self.teacher(x, training=False)

            with tf.GradientTape() as tape:
                # Get student predictions
                student_predictions = self.student(x, training=True)

                # Calculate distillation loss
                loss = self.distillation_loss(student_predictions, teacher_predictions, y)

            # Update student weights
            trainable_vars = self.student.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars, strict=False))

            # Calculate accuracy
            accuracy = tf.keras.metrics.sparse_categorical_accuracy(y, student_predictions)

            return {"loss": loss, "accuracy": tf.reduce_mean(accuracy)}

        def distillation_loss(self, student_outputs, teacher_outputs, labels):
            # Soft targets loss
            soft_student = tf.nn.softmax(student_outputs / self.temperature)
            soft_teacher = tf.nn.softmax(teacher_outputs / self.temperature)

            soft_loss = tf.reduce_mean(
                tf.keras.losses.kullback_leibler_divergence(soft_teacher, soft_student)
            ) * (self.temperature**2)

            # Hard targets loss
            hard_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    labels, student_outputs, from_logits=True
                )
            )

            return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

    def optimize(
        self, model: tf.keras.Model, dataset: Any | None = None, **kwargs
    ) -> OptimizationResult:
        """
        Apply knowledge distillation.

        Args:
            model: Teacher model (TensorFlow)
            dataset: Training dataset
            **kwargs: Additional arguments

        Returns:
            OptimizationResult with student model and metrics
        """
        start_time = time.time()

        if dataset is None:
            raise ValueError("Knowledge distillation requires a dataset for training")

        # Save teacher model to get size
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
            teacher_path = Path(tmp_file.name)
            model.save(teacher_path)

        # Get teacher metrics
        teacher_size_mb = self.get_model_size_mb(teacher_path)
        teacher_params = self.count_parameters(model)
        teacher_accuracy = self.evaluate_model(model, dataset)

        # Create student model
        student_model = self.create_student_model(model)

        # Create distillation trainer
        distillation_trainer = self.DistillationTrainer(
            student_model, model, self.temperature, self.alpha
        )

        # Compile trainer
        distillation_trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

        # Train student model
        distillation_trainer.fit(dataset, epochs=self.epochs, verbose=0)

        # Compile student model for evaluation
        student_model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(student_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        student_tflite_model = converter.convert()

        # Save student model to get size
        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as tmp_student:
            student_path = Path(tmp_student.name)
            student_path.write_bytes(student_tflite_model)

        # Get student metrics
        student_size_mb = self.get_model_size_mb(student_path)
        student_params = self.count_parameters(student_model)

        # Evaluate student model
        interpreter = tf.lite.Interpreter(model_content=student_tflite_model)
        interpreter.allocate_tensors()
        student_accuracy = self._evaluate_tflite_model(interpreter, dataset)

        # Clean up temporary files
        teacher_path.unlink()
        student_path.unlink()

        execution_time = time.time() - start_time

        result = OptimizationResult(
            optimized_model=student_tflite_model,
            original_accuracy=teacher_accuracy,
            optimized_accuracy=student_accuracy,
            original_size_mb=teacher_size_mb,
            optimized_size_mb=student_size_mb,
            original_params_count=teacher_params,
            optimized_params_count=student_params,
            execution_time_seconds=execution_time,
            metadata={
                "technique": "knowledge_distillation",
                "student_scale": self.student_scale,
                "temperature": self.temperature,
                "alpha": self.alpha,
                "epochs": self.epochs,
                "format": "tflite",
            },
            technique_name="distillation",
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
