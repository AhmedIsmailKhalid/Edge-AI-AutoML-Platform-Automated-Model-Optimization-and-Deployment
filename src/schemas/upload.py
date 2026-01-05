"""
Pydantic schemas for file upload operations.
"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ModelUploadResponse(BaseModel):
    """Schema for model upload response."""

    experiment_id: UUID
    model_file_id: UUID
    model_name: str
    framework: str
    file_size_mb: float
    file_path: str
    message: str = Field(default="Model uploaded successfully")


class DatasetUploadResponse(BaseModel):
    """Schema for dataset upload response."""

    experiment_id: UUID
    dataset_name: str
    dataset_type: Literal["preset", "custom"]
    dataset_path: str
    message: str = Field(default="Dataset uploaded successfully")


class UploadValidationError(BaseModel):
    """Schema for upload validation errors."""

    field: str
    message: str


class UploadErrorResponse(BaseModel):
    """Schema for upload error response."""

    error: str
    details: str | None = None
    validation_errors: list[UploadValidationError] | None = None
