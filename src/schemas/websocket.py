"""
Pydantic schemas for WebSocket messages.
"""

from uuid import UUID

from pydantic import BaseModel, Field


class TechniqueProgress(BaseModel):
    """Schema for current technique progress."""

    name: str = Field(..., description="Technique name")
    progress_percent: int = Field(..., ge=0, le=100, description="Progress percentage")
    eta_seconds: float | None = Field(None, ge=0, description="Estimated time remaining in seconds")
    stage: str = Field(..., description="Current stage description")


class OverallProgress(BaseModel):
    """Schema for overall experiment progress."""

    completed_techniques: int = Field(..., ge=0, description="Number of completed techniques")
    total_techniques: int = Field(..., gt=0, description="Total number of techniques")
    progress_percent: int = Field(..., ge=0, le=100, description="Overall progress percentage")
    eta_seconds: float | None = Field(None, ge=0, description="Estimated time remaining in seconds")


class LiveMetrics(BaseModel):
    """Schema for live optimization metrics."""

    original_accuracy: float | None = Field(None, ge=0, le=1, description="Original model accuracy")
    current_accuracy: float | None = Field(
        None, ge=0, le=1, description="Current optimized accuracy"
    )
    original_size_mb: float | None = Field(None, gt=0, description="Original model size in MB")
    current_size_mb: float | None = Field(None, gt=0, description="Current optimized size in MB")
    compression_ratio: float | None = Field(None, gt=0, description="Compression ratio")


class ProgressUpdate(BaseModel):
    """Schema for WebSocket progress update message."""

    type: str = Field(default="progress_update", description="Message type")
    experiment_id: UUID = Field(..., description="Experiment ID")
    current_technique: TechniqueProgress | None = Field(
        None, description="Current technique progress"
    )
    overall_progress: OverallProgress = Field(..., description="Overall progress")
    metrics: LiveMetrics | None = Field(None, description="Live metrics")


class ErrorMessage(BaseModel):
    """Schema for WebSocket error message."""

    type: str = Field(default="error", description="Message type")
    experiment_id: UUID = Field(..., description="Experiment ID")
    error: str = Field(..., description="Error message")
    technique_name: str | None = Field(None, description="Technique that failed")


class CompletionMessage(BaseModel):
    """Schema for WebSocket completion message."""

    type: str = Field(default="completion", description="Message type")
    experiment_id: UUID = Field(..., description="Experiment ID")
    status: str = Field(..., pattern="^(completed|failed)$", description="Final status")
    message: str = Field(..., description="Completion message")
    total_time_seconds: float | None = Field(None, gt=0, description="Total execution time")
