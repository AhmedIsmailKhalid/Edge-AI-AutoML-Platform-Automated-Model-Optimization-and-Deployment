"""
Pydantic schemas for optimization results.
"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from src.schemas.optimization import OptimizationRunResponse


class MetricComparison(BaseModel):
    """Schema for comparing metrics between original and optimized models."""

    original_accuracy: float | None = None
    optimized_accuracy: float | None = None
    accuracy_drop_percent: float | None = None
    original_size_mb: float | None = None
    optimized_size_mb: float | None = None
    size_reduction_percent: float | None = None
    compression_ratio: float | None = None
    estimated_latency_ms: float | None = None


class ResultResponse(BaseModel):
    """Schema for single optimization result."""

    optimization_run: OptimizationRunResponse
    metrics: MetricComparison
    meets_constraints: bool = Field(..., description="Whether result meets user constraints")

    model_config = {"from_attributes": True}


class ResultsListResponse(BaseModel):
    """Schema for list of optimization results."""

    experiment_id: UUID
    results: list[ResultResponse]
    total_techniques: int
    completed_techniques: int
    failed_techniques: int

    model_config = {"from_attributes": True}


class TechniqueResultDetail(BaseModel):
    """Detailed result for a single technique."""

    optimization_run: OptimizationRunResponse  # Changed from OptimizationRun
    metrics: dict[str, Any]
    meets_constraints: bool

    model_config = {"from_attributes": True}


class ExperimentResultsResponse(BaseModel):
    """Response containing all experiment results."""

    experiment_id: UUID
    total_techniques: int
    completed_techniques: int
    failed_techniques: int
    results: list[TechniqueResultDetail]


class RecommendationResponse(BaseModel):
    """A single technique recommendation."""

    rank: int = Field(..., description="Ranking position (1 = best)")
    technique_name: str = Field(..., description="Name of the optimization technique")
    score: float = Field(..., ge=0, le=1, description="Overall score (0-1)")
    primary_reason: str = Field(..., description="Primary reason for recommendation")
    explanation: str = Field(..., description="Human-readable explanation")
    metrics: dict[str, Any] = Field(..., description="Key metrics")
    meets_constraints: bool = Field(..., description="Whether it meets all constraints")
    strengths: list[str] = Field(..., description="Key strengths")
    tradeoffs: list[str] = Field(..., description="Tradeoffs and limitations")

    model_config = {"from_attributes": True}
