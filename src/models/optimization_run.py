"""
OptimizationRun database model.
"""

import enum
from datetime import datetime
from uuid import uuid4

from sqlalchemy import JSON, Column, DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from src.database import Base


class OptimizationStatus(str, enum.Enum):
    """Optimization run status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class OptimizationRun(Base):
    """
    OptimizationRun model representing a single optimization technique execution.
    """

    __tablename__ = "optimization_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False)

    # Technique information
    technique_name = Column(String(100), nullable=False)  # 'ptq_int8', 'pruning_magnitude', etc.
    technique_config = Column(
        JSON, nullable=False
    )  # Configuration parameters (changed from JSONB to JSON)

    # Execution information
    status = Column(SQLEnum(OptimizationStatus), nullable=False, default=OptimizationStatus.PENDING)
    execution_order = Column(Integer, nullable=False)  # For sequential execution tracking

    # Metrics - Original model
    original_accuracy = Column(Float, nullable=True)
    original_size_mb = Column(Float, nullable=True)
    original_params_count = Column(Integer, nullable=True)

    # Metrics - Optimized model
    optimized_accuracy = Column(Float, nullable=True)
    optimized_size_mb = Column(Float, nullable=True)
    optimized_params_count = Column(Integer, nullable=True)

    # Calculated metrics
    accuracy_drop_percent = Column(Float, nullable=True)
    size_reduction_percent = Column(Float, nullable=True)

    # Performance estimates
    estimated_latency_ms = Column(Float, nullable=True)
    estimated_memory_mb = Column(Float, nullable=True)
    estimated_power_watts = Column(Float, nullable=True)

    # Execution timing
    execution_time_seconds = Column(Float, nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)

    # Timestamps
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    experiment = relationship("Experiment", back_populates="optimization_runs")
    model_files = relationship(
        "ModelFile", back_populates="optimization_run", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<OptimizationRun(id={self.id}, technique='{self.technique_name}', status='{self.status}')>"
