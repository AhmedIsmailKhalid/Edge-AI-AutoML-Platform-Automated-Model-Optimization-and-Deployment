"""
Experiment database model.
"""

import enum
from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from src.database import Base


class ExperimentStatus(str, enum.Enum):
    """Experiment status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class OptimizationGoal(str, enum.Enum):
    """Optimization goal enumeration."""

    MAXIMIZE_ACCURACY = "maximize_accuracy"
    MINIMIZE_SIZE = "minimize_size"
    MINIMIZE_LATENCY = "minimize_latency"
    BALANCED = "balanced"


class Experiment(Base):
    """
    Experiment model representing a model optimization experiment.
    """

    __tablename__ = "experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Model information
    model_name = Column(String(255), nullable=True)  # CHANGED: Now nullable, set on upload
    framework = Column(String(50), nullable=False)  # 'pytorch' or 'tensorflow'
    model_architecture = Column(Text, nullable=True)  # JSON string

    # Dataset information
    dataset_type = Column(String(50), nullable=False)  # 'preset' or 'custom'
    dataset_name = Column(String(255), nullable=True)
    dataset_path = Column(Text, nullable=True)

    # Target device and constraints
    target_device = Column(String(100), nullable=True)  # 'raspberry_pi_4', etc.
    optimization_goal = Column(SQLEnum(OptimizationGoal), nullable=True)

    # Custom constraints (optional)
    min_accuracy_percent = Column(Float, nullable=True)
    max_size_mb = Column(Float, nullable=True)
    max_latency_ms = Column(Float, nullable=True)
    max_accuracy_drop_percent = Column(Float, nullable=True)

    # Execution tracking
    status = Column(SQLEnum(ExperimentStatus), nullable=False, default=ExperimentStatus.PENDING)
    progress_percent = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)

    # Timestamps
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    optimization_runs = relationship(
        "OptimizationRun", back_populates="experiment", cascade="all, delete-orphan"
    )
    model_files = relationship(
        "ModelFile", back_populates="experiment", cascade="all, delete-orphan"
    )
    recommendations = relationship(
        "Recommendation", back_populates="experiment", cascade="all, delete-orphan"
    )
    progress_updates = relationship(
        "ExperimentProgress", back_populates="experiment", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Experiment(id={self.id}, name='{self.name}', status='{self.status}')>"
