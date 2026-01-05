"""
Pydantic schemas for Recommendation model.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class RecommendationBase(BaseModel):
    """Base schema for Recommendation."""

    rank: int = Field(..., ge=1, description="Recommendation rank (1 = best)")
    overall_score: float = Field(..., ge=0, le=100, description="Overall recommendation score")
    meets_constraints: bool = Field(..., description="Whether recommendation meets constraints")
    reasoning: str = Field(..., min_length=1, description="Reasoning for recommendation")
    pros: list[str] | None = Field(default_factory=list, description="List of pros")
    cons: list[str] | None = Field(default_factory=list, description="List of cons")
    warnings: list[str] | None = Field(default_factory=list, description="List of warnings")
    pareto_optimal: bool = Field(default=False, description="Whether this is Pareto optimal")


class RecommendationCreate(RecommendationBase):
    """Schema for creating recommendation."""

    experiment_id: UUID
    optimization_run_id: UUID | None = None


class RecommendationResponse(RecommendationBase):
    """Schema for recommendation response."""

    id: UUID
    experiment_id: UUID
    optimization_run_id: UUID | None = None
    technique_name: str | None = Field(None, description="Optimization technique name")
    created_at: datetime

    model_config = {"from_attributes": True}


class RecommendationListResponse(BaseModel):
    """Schema for list of recommendations."""

    experiment_id: UUID
    recommendations: list[RecommendationResponse]
    total: int
    best_recommendation: RecommendationResponse | None = None

    model_config = {"from_attributes": True}
