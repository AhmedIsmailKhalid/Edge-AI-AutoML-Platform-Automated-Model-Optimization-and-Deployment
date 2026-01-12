"""
Pydantic schemas for Experiment model.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ExperimentBase(BaseModel):
    """Base schema for Experiment."""

    name: str = Field(..., min_length=1, max_length=255, description="Experiment name")
    description: str | None = Field(None, description="Experiment description")
    framework: str = Field(..., pattern="^(pytorch|tensorflow)$", description="ML framework")

    # Dataset information
    dataset_type: str = Field(..., pattern="^(preset|custom)$", description="Dataset type")
    dataset_name: str | None = Field(None, max_length=255, description="Dataset name")

    # Target device and optimization goal
    target_device: str | None = Field(None, max_length=100, description="Target edge device")
    optimization_goal: str | None = Field(
        None,
        pattern="^(maximize_accuracy|minimize_size|minimize_latency|balanced)$",
        description="Optimization goal",
    )

    # Custom constraints
    min_accuracy_percent: float | None = Field(None, ge=0, le=100)
    max_size_mb: float | None = Field(None, gt=0)
    max_latency_ms: float | None = Field(None, gt=0)
    max_accuracy_drop_percent: float | None = Field(None, ge=0, le=100)


class ExperimentCreate(ExperimentBase):
    """
    Schema for creating an experiment.

    Note: model_name is now OPTIONAL and will be inferred from uploaded file.
    """

    model_name: str | None = Field(
        None,
        max_length=255,
        description="Model name (auto-inferred from uploaded file if not provided)",
    )
    model_architecture: str | None = Field(None, description="Model architecture JSON")
    dataset_path: str | None = Field(None, description="Custom dataset path")


class ExperimentUpdate(BaseModel):
    """Schema for updating an experiment."""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    model_name: str | None = Field(None, max_length=255)
    dataset_name: str | None = None
    dataset_path: str | None = None
    target_device: str | None = None
    optimization_goal: str | None = None
    min_accuracy_percent: float | None = Field(None, ge=0, le=100)
    max_size_mb: float | None = Field(None, gt=0)
    max_latency_ms: float | None = Field(None, gt=0)
    max_accuracy_drop_percent: float | None = Field(None, ge=0, le=100)


class ExperimentResponse(BaseModel):
    """Schema for experiment response."""

    id: UUID
    name: str
    description: str | None = None
    model_name: str | None = None  # Can be None until model is uploaded
    framework: str
    model_architecture: str | None = None
    dataset_type: str
    dataset_name: str | None = None
    dataset_path: str | None = None
    target_device: str | None = None
    optimization_goal: str | None = None
    min_accuracy_percent: Optional[float] = None  # noqa: UP007
    max_size_mb: Optional[float] = None  # noqa: UP007
    max_latency_ms: Optional[float] = None  # noqa: UP007
    max_accuracy_drop_percent: float | None = None
    status: str
    progress_percent: int
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ExperimentListResponse(BaseModel):
    """Schema for list of experiments."""

    experiments: list[ExperimentResponse]
    total: int
