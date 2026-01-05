"""
Pydantic schemas for OptimizationRun model.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class OptimizationConfig(BaseModel):
    """Schema for optimization technique configuration."""

    technique_name: str = Field(..., description="Optimization technique name")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Technique-specific parameters"
    )


class OptimizationRunBase(BaseModel):
    """Base schema for OptimizationRun."""

    technique_name: str = Field(..., max_length=100, description="Optimization technique name")
    technique_config: dict[str, Any] = Field(
        default_factory=dict, description="Technique configuration"
    )
    execution_order: int = Field(..., ge=1, description="Execution order in experiment")


class OptimizationRunCreate(OptimizationRunBase):
    """Schema for creating optimization run."""

    experiment_id: UUID


class OptimizationRunUpdate(BaseModel):
    """Schema for updating optimization run."""

    status: str | None = Field(None, pattern="^(pending|running|completed|failed)$")
    original_accuracy: float | None = Field(None, ge=0, le=1)
    original_size_mb: float | None = Field(None, gt=0)
    original_params_count: int | None = Field(None, gt=0)
    optimized_accuracy: float | None = Field(None, ge=0, le=1)
    optimized_size_mb: float | None = Field(None, gt=0)
    optimized_params_count: int | None = Field(None, gt=0)
    accuracy_drop_percent: float | None = None
    size_reduction_percent: float | None = None
    estimated_latency_ms: float | None = Field(None, gt=0)
    estimated_memory_mb: float | None = Field(None, gt=0)
    estimated_power_watts: float | None = Field(None, gt=0)
    execution_time_seconds: float | None = Field(None, gt=0)
    error_message: str | None = None


class OptimizationRunResponse(OptimizationRunBase):
    """Schema for optimization run response."""

    id: UUID
    experiment_id: UUID
    status: str
    original_accuracy: float | None = None
    original_size_mb: float | None = None
    original_params_count: int | None = None
    optimized_accuracy: float | None = None
    optimized_size_mb: float | None = None
    optimized_params_count: int | None = None
    accuracy_drop_percent: float | None = None
    size_reduction_percent: float | None = None
    estimated_latency_ms: float | None = None
    estimated_memory_mb: float | None = None
    estimated_power_watts: float | None = None
    execution_time_seconds: float | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
