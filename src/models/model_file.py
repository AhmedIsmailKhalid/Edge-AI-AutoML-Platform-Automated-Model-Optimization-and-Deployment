"""
ModelFile database model.
"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from src.database import Base


class ModelFile(Base):
    """
    ModelFile model representing stored model files.
    """

    __tablename__ = "model_files"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False)
    optimization_run_id = Column(
        UUID(as_uuid=True), ForeignKey("optimization_runs.id"), nullable=True
    )

    # File information
    file_type = Column(String(50), nullable=False)  # 'original', 'optimized'
    file_format = Column(String(50), nullable=False)  # 'pytorch_pt', 'tensorflow_pb', etc.
    file_path = Column(Text, nullable=False)
    file_size_mb = Column(Float, nullable=False)

    # Metadata
    checksum = Column(String(64), nullable=True)  # SHA256 checksum

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    experiment = relationship("Experiment", back_populates="model_files")
    optimization_run = relationship("OptimizationRun", back_populates="model_files")

    def __repr__(self) -> str:
        return f"<ModelFile(id={self.id}, type='{self.file_type}', format='{self.file_format}')>"
