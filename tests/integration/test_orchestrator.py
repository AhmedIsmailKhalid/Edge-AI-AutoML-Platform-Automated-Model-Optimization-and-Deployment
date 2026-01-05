"""
Integration tests for the orchestrator.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.core.orchestrator import ExperimentOrchestrator
from src.database import Base
from src.models.experiment import Experiment, ExperimentStatus
from src.models.model_file import ModelFile

# Create in-memory test database
TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class CIFAR10Test(nn.Module):
    """Simple model for CIFAR-10 (32x32x3 images)."""

    def __init__(self):
        super().__init__()
        # CIFAR-10 images are 32x32x3 = 3072 pixels
        self.fc1 = nn.Linear(3072, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)  # 10 classes
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Flatten the image
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


@pytest.fixture
def db_session():
    """Create a test database session."""
    Base.metadata.create_all(bind=engine)
    session = TestSessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_model_path():
    """Create a temporary test model."""
    model = CIFAR10Test()
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        model_path = Path(tmp.name)
        # Save the entire model, not just state_dict
        torch.save(model, model_path)

    yield model_path

    # Cleanup
    if model_path.exists():
        model_path.unlink()


@pytest.mark.asyncio
async def test_orchestrator_complete_workflow(db_session, test_model_path):
    """Test complete orchestrator workflow."""
    print("\n  Testing complete orchestrator workflow...")

    # Create experiment
    experiment = Experiment(
        name="Test Orchestrator Experiment",
        description="Testing orchestrator",
        model_name="test_model",
        framework="pytorch",
        dataset_type="preset",
        dataset_name="cifar10",
        target_device="raspberry_pi_4",
        status=ExperimentStatus.PENDING,
    )
    db_session.add(experiment)
    db_session.commit()
    db_session.refresh(experiment)

    print(f"✅ Created experiment: {experiment.id}")

    # Create model file record
    model_file = ModelFile(
        experiment_id=experiment.id,
        file_type="original",
        file_format="pytorch_pth",
        file_path=str(test_model_path),
        file_size_mb=0.1,
    )
    db_session.add(model_file)
    db_session.commit()

    print("✅ Created model file record")

    # Create orchestrator
    orchestrator = ExperimentOrchestrator(experiment.id, db_session)

    # Run orchestrator
    await orchestrator.run()

    # Refresh experiment
    db_session.refresh(experiment)

    # Verify experiment completed
    assert experiment.status == ExperimentStatus.COMPLETED
    assert experiment.progress_percent == 100
    assert experiment.started_at is not None
    assert experiment.completed_at is not None

    print(f"✅ Experiment status: {experiment.status}")
    print(f"✅ Progress: {experiment.progress_percent}%")

    # Verify optimization runs were created
    from src.models.optimization_run import OptimizationRun

    opt_runs = (
        db_session.query(OptimizationRun)
        .filter(OptimizationRun.experiment_id == experiment.id)
        .all()
    )

    assert len(opt_runs) > 0, "No optimization runs created"
    print(f"✅ Created {len(opt_runs)} optimization run(s)")

    # Verify at least one technique completed successfully
    completed_runs = [r for r in opt_runs if r.status.value == "completed"]
    assert len(completed_runs) > 0, "No techniques completed successfully"
    print(f"✅ {len(completed_runs)} technique(s) completed successfully")

    # Verify metrics were recorded
    for run in completed_runs:
        print(f"\n  Technique: {run.technique_name}")
        print(f"   Original size: {run.original_size_mb:.2f} MB")
        print(f"   Optimized size: {run.optimized_size_mb:.2f} MB")
        print(f"   Size reduction: {run.size_reduction_percent:.2f}%")
        print(f"   Execution time: {run.execution_time_seconds:.2f}s")

        assert run.original_size_mb > 0
        assert run.optimized_size_mb > 0
        assert run.optimized_size_mb < run.original_size_mb

    print("\n✅ Orchestrator workflow test PASSED!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
