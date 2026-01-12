"""
TensorFlow optimization techniques.
"""

from src.core.tensorflow.distillation import KnowledgeDistillationTensorFlow
from src.core.tensorflow.hybrid import HybridPruneQuantizeTensorFlow
from src.core.tensorflow.pruning import (
    PruningMagnitudeStructuredTensorFlow,
    PruningMagnitudeUnstructuredTensorFlow)
from src.core.tensorflow.ptq_int4 import PTQInt4TensorFlow
from src.core.tensorflow.ptq_int8 import PTQInt8TensorFlow
from src.core.tensorflow.qat import QuantizationAwareTrainingTensorFlow

__all__ = [
    "PTQInt8TensorFlow",
    "PTQInt4TensorFlow",
    "PruningMagnitudeUnstructuredTensorFlow",
    "PruningMagnitudeStructuredTensorFlow",
    "KnowledgeDistillationTensorFlow",
    "QuantizationAwareTrainingTensorFlow",
    "HybridPruneQuantizeTensorFlow",
]
