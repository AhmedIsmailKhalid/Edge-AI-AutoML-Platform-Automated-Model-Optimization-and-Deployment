"""
ExperimentProgress database model for real-time progress tracking.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from src.database import Base


class ExperimentProgress(Base):
    """
    ExperimentProgress model for tracking real-time optimization progress.
    Used for WebSocket updates.
    """

    __tablename__ = "experiment_progress"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False)
    optimization_run_id = Column(
        UUID(as_uuid=True), ForeignKey("optimization_runs.id"), nullable=True
    )

    # Progress details
    stage = Column(String(100), nullable=False)  # 'loading_model', 'quantizing', etc.
    progress_percent = Column(Integer, nullable=False)
    message = Column(Text, nullable=True)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    experiment = relationship("Experiment", back_populates="progress_updates")

    def __repr__(self) -> str:
        return f"<ExperimentProgress(id={self.id}, stage='{self.stage}', progress={self.progress_percent}%)>"
