"""
PyTorch optimization techniques.
"""

from src.core.pytorch.distillation import KnowledgeDistillationPyTorch
from src.core.pytorch.hybrid import HybridPruneQuantizePyTorch
from src.core.pytorch.pruning import (PruningMagnitudeStructuredPyTorch,
                                      PruningMagnitudeUnstructuredPyTorch)
from src.core.pytorch.ptq_int4 import PTQInt4PyTorch
from src.core.pytorch.ptq_int8 import PTQInt8PyTorch
from src.core.pytorch.qat import QuantizationAwareTrainingPyTorch

__all__ = [
    "PTQInt8PyTorch",
    "PTQInt4PyTorch",
    "PruningMagnitudeUnstructuredPyTorch",
    "PruningMagnitudeStructuredPyTorch",
    "KnowledgeDistillationPyTorch",
    "QuantizationAwareTrainingPyTorch",
    "HybridPruneQuantizePyTorch",
]
