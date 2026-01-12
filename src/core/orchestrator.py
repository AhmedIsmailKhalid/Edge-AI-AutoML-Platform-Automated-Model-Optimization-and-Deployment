"""
Experiment orchestrator for sequential optimization execution.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from src.config import settings
from src.core.base import OptimizationResult
from src.core.performance_estimator import DeviceType, PerformanceEstimator
from src.core.pytorch.hybrid import HybridPruneQuantizePyTorch
from src.core.pytorch.pruning import (PruningMagnitudeStructuredPyTorch,
                                      PruningMagnitudeUnstructuredPyTorch)
from src.core.pytorch.ptq_int4 import PTQInt4PyTorch
from src.core.pytorch.ptq_int8 import PTQInt8PyTorch
from src.core.pytorch.qat import QuantizationAwareTrainingPyTorch
from src.core.tensorflow.hybrid import HybridPruneQuantizeTensorFlow
from src.core.tensorflow.pruning import (
    PruningMagnitudeStructuredTensorFlow,
    PruningMagnitudeUnstructuredTensorFlow)
from src.core.tensorflow.ptq_int4 import PTQInt4TensorFlow
from src.core.tensorflow.ptq_int8 import PTQInt8TensorFlow
from src.core.tensorflow.qat import QuantizationAwareTrainingTensorFlow
from src.models.experiment import Experiment, ExperimentStatus
from src.models.model_file import ModelFile
from src.models.optimization_run import OptimizationRun, OptimizationStatus
from src.utils.dataset_loader import load_dataset
from src.utils.model_loader import load_model

logger = logging.getLogger(__name__)


class ExperimentOrchestrator:
    """
    Orchestrator for managing sequential execution of optimization techniques.
    """

    # Map technique names to their implementations
    PYTORCH_TECHNIQUES = {
        "ptq_int8": PTQInt8PyTorch,
        "ptq_int4": PTQInt4PyTorch,
        "pruning_magnitude_unstructured": PruningMagnitudeUnstructuredPyTorch,
        "pruning_magnitude_structured": PruningMagnitudeStructuredPyTorch,
        "quantization_aware_training": QuantizationAwareTrainingPyTorch,
        "hybrid_prune_quantize": HybridPruneQuantizePyTorch,
    }

    TENSORFLOW_TECHNIQUES = {
        "ptq_int8": PTQInt8TensorFlow,
        "ptq_int4": PTQInt4TensorFlow,
        "pruning_magnitude_unstructured": PruningMagnitudeUnstructuredTensorFlow,
        "pruning_magnitude_structured": PruningMagnitudeStructuredTensorFlow,
        "quantization_aware_training": QuantizationAwareTrainingTensorFlow,
        "hybrid_prune_quantize": HybridPruneQuantizeTensorFlow,
    }

    def __init__(self, experiment_id: UUID, db: Session):
        """
        Initialize orchestrator.

        Args:
            experiment_id: Experiment UUID
            db: Database session
        """
        self.experiment_id = experiment_id
        self.db = db
        self.experiment: Experiment | None = None
        self.model: Any | None = None
        self.dataset: Any | None = None
        self.techniques: list[str] = []

    def _get_techniques_for_framework(self, framework: str) -> dict[str, type]:
        """Get available techniques for framework."""
        if framework == "pytorch":
            return self.PYTORCH_TECHNIQUES
        elif framework == "tensorflow":
            return self.TENSORFLOW_TECHNIQUES
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    # async def run(self) -> None:
    #     """
    #     Run the complete optimization experiment sequentially.
    #     """
    #     try:
    #         # Load experiment
    #         await self._load_experiment()

    #         # Update status to running
    #         self.experiment.status = ExperimentStatus.RUNNING
    #         self.experiment.started_at = datetime.utcnow()
    #         self.db.commit()
    #         self.db.flush()

    #         # Load model
    #         await self._load_model()

    #         # Load dataset
    #         await self._load_dataset()

    #         # Get available techniques
    #         techniques_map = self._get_techniques_for_framework(self.experiment.framework)
    #         self.techniques = list(techniques_map.keys())

    #         # Run each technique sequentially
    #         for idx, technique_name in enumerate(self.techniques, start=1):
    #             logger.info(f"Running technique {idx}/{len(self.techniques)}: {technique_name}")

    #             technique_class = techniques_map[technique_name]
    #             await self._run_technique(
    #                 technique_name=technique_name,
    #                 technique_class=technique_class,
    #                 execution_order=idx,
    #             )

    #             # Update overall progress
    #             progress = int((idx / len(self.techniques)) * 100)
    #             self.experiment.progress_percent = progress
    #             self.db.commit()
    #             self.db.flush()

    #         # Mark experiment as completed
    #         self.experiment.status = ExperimentStatus.COMPLETED
    #         self.experiment.completed_at = datetime.utcnow()
    #         self.experiment.progress_percent = 100
    #         self.db.commit()
    #         self.db.flush()

    #         logger.info(f"Experiment {self.experiment_id} completed successfully")

    #     except Exception as e:
    #         logger.error(f"Experiment {self.experiment_id} failed: {e}", exc_info=True)

    #         # Mark experiment as failed
    #         if self.experiment:
    #             self.experiment.status = ExperimentStatus.FAILED
    #             self.experiment.error_message = str(e)
    #             self.experiment.completed_at = datetime.utcnow()
    #             self.db.commit()
    #             self.db.flush()

    #         raise

    #     finally:
    #         # CRITICAL: Always close DB session to release resources
    #         logger.info(f"Cleaning up resources for experiment {self.experiment_id}")
    #         try:
    #             # Ensure any pending changes are committed
    #             if self.db.is_active:
    #                 self.db.commit()
    #                 self.db.flush()
    #             self.db.close()
    #             logger.info(f"DB session closed for experiment {self.experiment_id}")
    #         except Exception as cleanup_error:
    #             logger.error(f"Error closing DB session: {cleanup_error}")

    async def run(self) -> None:
        """
        Run the complete optimization experiment sequentially.
        """
        try:
            # Load experiment
            await self._load_experiment()

            # Update status to running
            self.experiment.status = ExperimentStatus.RUNNING
            self.experiment.started_at = datetime.utcnow()
            self.db.commit()
            self.db.flush()

            # Load model
            await self._load_model()

            # Load dataset
            await self._load_dataset()

            # Get available techniques
            techniques_map = self._get_techniques_for_framework(self.experiment.framework)
            self.techniques = list(techniques_map.keys())

            # Run each technique sequentially
            for idx, technique_name in enumerate(self.techniques, start=1):
                logger.info(f"Running technique {idx}/{len(self.techniques)}: {technique_name}")

                technique_class = techniques_map[technique_name]
                await self._run_technique(
                    technique_name=technique_name,
                    technique_class=technique_class,
                    execution_order=idx,
                )

                # Update overall progress
                progress = int((idx / len(self.techniques)) * 100)
                self.experiment.progress_percent = progress
                self.db.commit()
                self.db.flush()

                # Small delay to allow CLI to catch state transition
                await asyncio.sleep(0.3)  # 300ms delay between techniques

            # Mark experiment as completed
            self.experiment.status = ExperimentStatus.COMPLETED
            self.experiment.completed_at = datetime.utcnow()
            self.experiment.progress_percent = 100
            self.db.commit()
            self.db.flush()

            logger.info(f"Experiment {self.experiment_id} completed successfully")

        except Exception as e:
            logger.error(f"Experiment {self.experiment_id} failed: {e}", exc_info=True)

            # Mark experiment as failed
            if self.experiment:
                self.experiment.status = ExperimentStatus.FAILED
                self.experiment.error_message = str(e)
                self.experiment.completed_at = datetime.utcnow()
                self.db.commit()
                self.db.flush()

            raise

        finally:
            # CRITICAL: Always close DB session to release resources
            logger.info(f"Cleaning up resources for experiment {self.experiment_id}")
            try:
                # Ensure any pending changes are committed
                if self.db.is_active:
                    self.db.commit()
                    self.db.flush()
                self.db.close()
                logger.info(f"DB session closed for experiment {self.experiment_id}")
            except Exception as cleanup_error:
                logger.error(f"Error closing DB session: {cleanup_error}")

    async def _load_experiment(self) -> None:
        """Load experiment from database."""
        self.experiment = (
            self.db.query(Experiment).filter(Experiment.id == self.experiment_id).first()
        )

        if not self.experiment:
            raise ValueError(f"Experiment {self.experiment_id} not found")

        logger.info(f"Loaded experiment: {self.experiment.name}")

    async def _load_model(self) -> None:
        """Load model file."""
        # Get the original model file
        model_file = (
            self.db.query(ModelFile)
            .filter(
                ModelFile.experiment_id == self.experiment_id, ModelFile.file_type == "original"
            )
            .first()
        )

        if not model_file:
            raise ValueError(f"No model file found for experiment {self.experiment_id}")

        model_path = Path(model_file.file_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model based on framework
        self.model = load_model(self.experiment.framework, model_path)
        logger.info(f"Loaded model from {model_path}")

    async def _reload_model(self) -> Any:
        """
        Reload a fresh copy of the model from disk.
        This ensures each technique gets an unmodified model.

        Returns:
            Fresh model instance
        """
        # Get the original model file
        model_file = (
            self.db.query(ModelFile)
            .filter(
                ModelFile.experiment_id == self.experiment_id, ModelFile.file_type == "original"
            )
            .first()
        )

        model_path = Path(model_file.file_path)

        # Reload model based on framework
        fresh_model = load_model(self.experiment.framework, model_path)
        logger.info("Reloaded fresh model for technique")

        return fresh_model

    async def _load_dataset(self) -> None:
        """Load dataset for evaluation."""
        if self.experiment.dataset_type == "preset":
            self.dataset = load_dataset(
                framework=self.experiment.framework,
                dataset_type="preset",
                dataset_name=self.experiment.dataset_name,
                batch_size=32,
            )
            logger.info(f"Loaded preset dataset: {self.experiment.dataset_name}")
        elif self.experiment.dataset_type == "custom":
            # Load custom dataset
            if not self.experiment.dataset_path:
                raise ValueError("Custom dataset path not specified")

            self.dataset = load_dataset(
                framework=self.experiment.framework,
                dataset_type="custom",
                dataset_name=self.experiment.dataset_name,
                dataset_path=self.experiment.dataset_path,
                batch_size=32,
            )
            logger.info(f"Loaded custom dataset from: {self.experiment.dataset_path}")
        else:
            logger.warning(f"Unknown dataset type: {self.experiment.dataset_type}")
            self.dataset = None

    # async def _run_technique(
    #     self, technique_name: str, technique_class: type, execution_order: int
    # ) -> None:
    #     """Run a single optimization technique."""

    #     # Create optimization run record
    #     opt_run = OptimizationRun(
    #         experiment_id=self.experiment_id,
    #         technique_name=technique_name,
    #         technique_config={},
    #         status=OptimizationStatus.RUNNING,
    #         execution_order=execution_order,
    #         started_at=datetime.utcnow(),
    #     )
    #     self.db.add(opt_run)
    #     self.db.commit()
    #     self.db.flush()
    #     self.db.refresh(opt_run)

    #     # Log that technique is starting
    #     logger.info(f"▶️  Starting technique: {technique_name}")

    #     # Give CLI time to see RUNNING status
    #     await asyncio.sleep(1.0)  # 1 second pause before starting

    #     try:
    #         # Initialize technique
    #         technique = technique_class(config={})

    #         # Run optimization
    #         fresh_model = await self._reload_model()
    #         result: OptimizationResult = technique.optimize(model=fresh_model, dataset=self.dataset)

    #         # Update optimization run with results
    #         opt_run.status = OptimizationStatus.COMPLETED
    #         opt_run.completed_at = datetime.utcnow()
    #         opt_run.original_accuracy = result.original_accuracy
    #         opt_run.optimized_accuracy = result.optimized_accuracy
    #         opt_run.original_size_mb = result.original_size_mb
    #         opt_run.optimized_size_mb = result.optimized_size_mb
    #         opt_run.original_params_count = result.original_params_count
    #         opt_run.optimized_params_count = result.optimized_params_count
    #         opt_run.execution_time_seconds = result.execution_time_seconds

    #         # Calculate derived metrics
    #         if result.original_accuracy > 0:
    #             opt_run.accuracy_drop_percent = (
    #                 (result.original_accuracy - result.optimized_accuracy)
    #                 / result.original_accuracy
    #             ) * 100

    #         if result.original_size_mb > 0:
    #             opt_run.size_reduction_percent = (
    #                 (result.original_size_mb - result.optimized_size_mb) / result.original_size_mb
    #             ) * 100

    #         self.db.commit()
    #         self.db.flush()

    #         # Save optimized model
    #         await self._save_optimized_model(opt_run.id, technique_name, result)

    #         # Log completion
    #         logger.info(
    #             f"✅ Technique {technique_name} completed: {result.optimized_accuracy*100:.1f}% accuracy, {opt_run.size_reduction_percent:.1f}% size reduction"
    #         )

    #         # Give CLI time to see COMPLETED status
    #         await asyncio.sleep(0.5)  # 500ms pause after completion

    #     except Exception as e:
    #         logger.error(f"❌ Technique {technique_name} failed: {e}", exc_info=True)

    #         opt_run.status = OptimizationStatus.FAILED
    #         opt_run.error_message = str(e)
    #         opt_run.completed_at = datetime.utcnow()
    #         self.db.commit()
    #         self.db.flush()

    async def _run_technique(
        self, technique_name: str, technique_class: type, execution_order: int
    ) -> None:
        """Run a single optimization technique."""

        # Create optimization run record
        opt_run = OptimizationRun(
            experiment_id=self.experiment_id,
            technique_name=technique_name,
            technique_config={},
            status=OptimizationStatus.RUNNING,
            execution_order=execution_order,
            started_at=datetime.utcnow(),
        )
        self.db.add(opt_run)
        self.db.commit()
        self.db.flush()
        self.db.refresh(opt_run)

        # Log that technique is starting
        logger.info(f"▶️  Starting technique: {technique_name}")

        # Give CLI time to see RUNNING status - INCREASED DELAY
        await asyncio.sleep(2.0)  # 2 second pause before starting (was 1.0)

        try:
            # Initialize technique
            technique = technique_class(config={})

            # Run optimization
            fresh_model = await self._reload_model()
            result: OptimizationResult = technique.optimize(model=fresh_model, dataset=self.dataset)

            # Update optimization run with results
            opt_run.status = OptimizationStatus.COMPLETED
            opt_run.completed_at = datetime.utcnow()
            opt_run.original_accuracy = result.original_accuracy
            opt_run.optimized_accuracy = result.optimized_accuracy
            opt_run.original_size_mb = result.original_size_mb
            opt_run.optimized_size_mb = result.optimized_size_mb
            opt_run.original_params_count = result.original_params_count
            opt_run.optimized_params_count = result.optimized_params_count
            opt_run.execution_time_seconds = result.execution_time_seconds

            # Calculate derived metrics
            if result.original_accuracy > 0:
                opt_run.accuracy_drop_percent = (
                    (result.original_accuracy - result.optimized_accuracy)
                    / result.original_accuracy
                ) * 100

            if result.original_size_mb > 0:
                opt_run.size_reduction_percent = (
                    (result.original_size_mb - result.optimized_size_mb) / result.original_size_mb
                ) * 100

            self.db.commit()
            self.db.flush()

            # Save optimized model
            await self._save_optimized_model(opt_run.id, technique_name, result)

            # Log completion
            logger.info(
                f"✅ Technique {technique_name} completed: {result.optimized_accuracy*100:.1f}% accuracy, {opt_run.size_reduction_percent:.1f}% size reduction"
            )

            # Give CLI time to see COMPLETED status - INCREASED DELAY
            await asyncio.sleep(1.0)  # 1 second pause after completion (was 0.5)

        except Exception as e:
            logger.error(f"❌ Technique {technique_name} failed: {e}", exc_info=True)

            opt_run.status = OptimizationStatus.FAILED
            opt_run.error_message = str(e)
            opt_run.completed_at = datetime.utcnow()
            self.db.commit()
            self.db.flush()

    async def _save_optimized_model(
        self, optimization_run_id: UUID, technique_name: str, result: OptimizationResult
    ) -> None:
        """
        Save optimized model to storage.

        Args:
            optimization_run_id: OptimizationRun UUID
            technique_name: Name of technique
            result: Optimization result
        """
        # Create directory for this experiment's optimized models
        optimized_dir = settings.optimized_models_path / str(self.experiment_id)
        optimized_dir.mkdir(parents=True, exist_ok=True)

        if "pretrained" in str(optimized_dir).lower():
            raise ValueError(
                f"Cannot save optimized model to pretrained directory: {optimized_dir}"
            )

        # Determine file extension
        if self.experiment.framework == "pytorch":
            extension = ".pth"
            file_format = "pytorch_pth"
        else:
            extension = ".tflite"
            file_format = "tensorflow_tflite"

        # Save model
        model_path = optimized_dir / f"{technique_name}{extension}"

        if self.experiment.framework == "pytorch":
            import torch

            torch.save(result.optimized_model.state_dict(), model_path)
        else:
            # TensorFlow TFLite is bytes
            model_path.write_bytes(result.optimized_model)

        # Create model file record
        model_file = ModelFile(
            experiment_id=self.experiment_id,
            optimization_run_id=optimization_run_id,
            file_type="optimized",
            file_format=file_format,
            file_path=str(model_path),
            file_size_mb=result.optimized_size_mb,
        )
        self.db.add(model_file)
        self.db.commit()
        self.db.flush()

        logger.info(f"Saved optimized model to {model_path}")

    async def _estimate_performance(
        self, opt_run: OptimizationRun, result: OptimizationResult
    ) -> None:
        """
        Estimate performance on target device.

        Args:
            opt_run: OptimizationRun record
            result: Optimization result
        """
        try:
            # Map target device to DeviceType enum
            device_mapping = {
                "raspberry_pi_3b": DeviceType.RASPBERRY_PI_3B,
                "raspberry_pi_4": DeviceType.RASPBERRY_PI_4,
                "raspberry_pi_5": DeviceType.RASPBERRY_PI_5,
                "jetson_nano": DeviceType.JETSON_NANO,
                "jetson_xavier_nx": DeviceType.JETSON_XAVIER_NX,
                "coral_dev_board": DeviceType.CORAL_DEV_BOARD,
            }

            device_type = device_mapping.get(
                self.experiment.target_device.lower().replace(" ", "_")
            )

            if not device_type:
                logger.warning(f"Unknown device type: {self.experiment.target_device}")
                return

            # Create estimator
            estimator = PerformanceEstimator(device_type)

            # Detect if model is quantized
            is_quantized = (
                "quantization" in opt_run.technique_name or "ptq" in opt_run.technique_name
            )
            quantization_bits = (
                8
                if "int8" in opt_run.technique_name
                else (4 if "int4" in opt_run.technique_name else 32)
            )

            # Estimate performance
            performance = estimator.estimate_performance(
                model_size_mb=result.optimized_size_mb,
                params_count=result.optimized_params_count,
                framework=self.experiment.framework,
                is_quantized=is_quantized,
                quantization_bits=quantization_bits,
            )

            # Update optimization run with estimates
            opt_run.estimated_latency_ms = performance.estimated_latency_ms
            opt_run.estimated_memory_mb = performance.estimated_memory_mb
            opt_run.estimated_power_watts = performance.estimated_power_watts

            logger.info(
                f"Performance estimate for {opt_run.technique_name}: "
                f"latency={performance.estimated_latency_ms:.1f}ms, "
                f"memory={performance.estimated_memory_mb:.1f}MB, "
                f"power={performance.estimated_power_watts:.1f}W"
            )

        except Exception as e:
            logger.error(f"Failed to estimate performance: {e}", exc_info=True)
            # Don't fail the optimization if estimation fails
