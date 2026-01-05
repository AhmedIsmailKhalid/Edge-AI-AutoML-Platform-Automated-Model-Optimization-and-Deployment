"""
Recommendation database model.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from src.database import Base


class Recommendation(Base):
    """
    Recommendation model representing optimization technique recommendations.
    """

    __tablename__ = "recommendations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False)
    optimization_run_id = Column(
        UUID(as_uuid=True), ForeignKey("optimization_runs.id"), nullable=True
    )

    # Recommendation details
    rank = Column(Integer, nullable=False)  # 1 = best recommendation
    overall_score = Column(Float, nullable=False)
    meets_constraints = Column(Boolean, nullable=False, default=False)

    # Reasoning
    reasoning = Column(Text, nullable=False)
    pros = Column(JSON, nullable=True)  # Array of pros (changed from JSONB to JSON)
    cons = Column(JSON, nullable=True)  # Array of cons (changed from JSONB to JSON)
    warnings = Column(JSON, nullable=True)  # Array of warnings (changed from JSONB to JSON)

    # Trade-off analysis
    pareto_optimal = Column(Boolean, default=False)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    experiment = relationship("Experiment", back_populates="recommendations")

    def __repr__(self) -> str:
        return f"<Recommendation(id={self.id}, rank={self.rank}, score={self.overall_score})>"
