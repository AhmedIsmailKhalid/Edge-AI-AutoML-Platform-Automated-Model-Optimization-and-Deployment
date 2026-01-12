"""
Integration tests for performance estimation API.
"""

import pytest
from fastapi.testclient import TestClient

from src.database import SessionLocal
from src.main import app
from src.models.experiment import (Experiment, ExperimentStatus,
                                   OptimizationGoal)
from src.models.optimization_run import OptimizationRun, OptimizationStatus

client = TestClient(app)


@pytest.fixture
def test_experiment():
    """Create a test experiment with optimization runs."""
    db = SessionLocal()
    try:
        # Create experiment
        experiment = Experiment(
            name="Performance Test",
            model_name="test_model",
            framework="pytorch",
            dataset_type="preset",
            dataset_name="mnist",
            optimization_goal=OptimizationGoal.BALANCED,
            target_device="raspberry_pi_4",
            status=ExperimentStatus.COMPLETED,
            progress_percent=100,
        )
        db.add(experiment)
        db.commit()
        db.refresh(experiment)

        # Create optimization run
        opt_run = OptimizationRun(
            experiment_id=experiment.id,
            technique_name="ptq_int8",
            technique_config={},
            status=OptimizationStatus.COMPLETED,
            execution_order=1,
            original_accuracy=0.95,
            optimized_accuracy=0.93,
            original_size_mb=20.0,
            optimized_size_mb=5.0,
            original_params_count=5_000_000,
            optimized_params_count=5_000_000,
            execution_time_seconds=120.0,
        )
        db.add(opt_run)
        db.commit()

        yield experiment

    finally:
        # Cleanup
        db.query(OptimizationRun).filter(OptimizationRun.experiment_id == experiment.id).delete()
        db.query(Experiment).filter(Experiment.id == experiment.id).delete()
        db.commit()
        db.close()


def test_list_devices():
    """Test listing supported devices."""
    print("\n🧪 Testing device listing...")

    response = client.get("/api/performance/devices")

    assert response.status_code == 200
    data = response.json()

    print(f"   📱 Found {data['total']} devices:")
    for device in data["devices"]:
        print(f"      - {device['name']} ({device['device_type']})")

    assert data["total"] > 0
    assert len(data["devices"]) == data["total"]

    # Check first device has required fields
    first_device = data["devices"][0]
    assert "device_type" in first_device
    assert "name" in first_device
    assert "cpu_cores" in first_device
    assert "ram_gb" in first_device

    print("   ✅ Device listing works!")


def test_get_performance_estimate(test_experiment):
    """Test getting performance estimate for a technique."""
    print("\n🧪 Testing performance estimation...")

    experiment_id = str(test_experiment.id)

    response = client.get(f"/api/performance/experiments/{experiment_id}/performance/ptq_int8")

    assert response.status_code == 200
    data = response.json()

    print("   📊 Performance estimate for ptq_int8:")
    print(f"      Device: {data['device']['name']}")
    print(
        f"      Latency: {data['latency']['estimated_ms']:.1f}ms ({data['latency']['fps']:.1f} FPS)"
    )
    print(
        f"      Memory: {data['memory']['estimated_mb']:.1f}MB ({data['memory']['utilization_percent']:.1f}%)"
    )
    print(f"      Power: {data['power']['estimated_watts']:.2f}W")
    print(f"      Feasible: {data['feasibility']['is_feasible']}")

    # Check structure
    assert "experiment_id" in data
    assert "technique_name" in data
    assert "device" in data
    assert "latency" in data
    assert "memory" in data
    assert "power" in data
    assert "feasibility" in data

    # Check values are reasonable
    assert data["latency"]["estimated_ms"] > 0
    assert data["memory"]["estimated_mb"] > 0
    assert data["power"]["estimated_watts"] > 0

    print("   ✅ Performance estimation works!")


def test_compare_devices(test_experiment):
    """Test comparing performance across devices."""
    print("\n🧪 Testing device comparison...")

    experiment_id = str(test_experiment.id)

    response = client.get(f"/api/performance/experiments/{experiment_id}/compare-devices/ptq_int8")

    assert response.status_code == 200
    data = response.json()

    print(f"   📊 Comparison across {data['total_devices']} devices:")
    print(f"   🏆 Best device: {data['best_device']}")
    print("\n   Device Rankings:")

    for i, comp in enumerate(data["comparisons"][:5], 1):  # Top 5
        print(
            f"      {i}. {comp['device_name']}: {comp['latency_ms']:.1f}ms @ {comp['fps']:.1f} FPS"
        )

    # Check structure
    assert "comparisons" in data
    assert "best_device" in data
    assert "total_devices" in data
    assert len(data["comparisons"]) > 0

    # Check sorting (by latency, ascending)
    latencies = [c["latency_ms"] for c in data["comparisons"]]
    assert latencies == sorted(latencies)

    print("   ✅ Device comparison works!")


def test_invalid_experiment():
    """Test error handling for invalid experiment."""
    print("\n🧪 Testing error handling...")

    response = client.get(
        "/api/performance/experiments/00000000-0000-0000-0000-000000000000/performance/ptq_int8"
    )

    assert response.status_code == 404
    print("   ✅ Correctly handles invalid experiment!")


def test_invalid_technique(test_experiment):
    """Test error handling for invalid technique."""
    print("\n🧪 Testing invalid technique...")

    experiment_id = str(test_experiment.id)

    response = client.get(
        f"/api/performance/experiments/{experiment_id}/performance/invalid_technique"
    )

    assert response.status_code == 404
    print("   ✅ Correctly handles invalid technique!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
