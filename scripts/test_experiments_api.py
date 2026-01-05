"""
Test script for Experiments API endpoints.
"""

import json

import requests

BASE_URL = "http://localhost:8000"


def test_create_experiment():
    """Test creating a new experiment."""
    print("\n🧪 Testing: Create Experiment")

    payload = {
        "name": "Test Experiment 1",
        "description": "Testing experiment creation",
        "model_name": "resnet18",
        "framework": "pytorch",
        "dataset_type": "preset",
        "dataset_name": "cifar10",
        "target_device": "raspberry_pi_4",
        "optimization_goal": "balanced",
        "min_accuracy_percent": 90.0,
        "max_size_mb": 25.0,
    }

    response = requests.post(f"{BASE_URL}/api/experiments/create", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 201, "Failed to create experiment"
    data = response.json()
    assert data["name"] == payload["name"]
    assert data["status"] == "pending"
    print("✅ Create Experiment: PASSED")

    return data["id"]


def test_list_experiments():
    """Test listing all experiments."""
    print("\n🧪 Testing: List Experiments")

    response = requests.get(f"{BASE_URL}/api/experiments")
    print(f"Status Code: {response.status_code}")

    assert response.status_code == 200, "Failed to list experiments"
    data = response.json()
    print(f"Total Experiments: {data['total']}")
    print("✅ List Experiments: PASSED")


def test_get_experiment(experiment_id):
    """Test getting a specific experiment."""
    print(f"\n🧪 Testing: Get Experiment {experiment_id}")

    response = requests.get(f"{BASE_URL}/api/experiments/{experiment_id}")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200, "Failed to get experiment"
    print("✅ Get Experiment: PASSED")


def test_update_experiment(experiment_id):
    """Test updating an experiment."""
    print(f"\n🧪 Testing: Update Experiment {experiment_id}")

    payload = {"name": "Updated Test Experiment", "status": "running", "progress_percent": 50}

    response = requests.patch(f"{BASE_URL}/api/experiments/{experiment_id}", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200, "Failed to update experiment"
    data = response.json()
    assert data["name"] == payload["name"]
    assert data["status"] == payload["status"]
    print("✅ Update Experiment: PASSED")


def test_delete_experiment(experiment_id):
    """Test deleting an experiment."""
    print(f"\n🧪 Testing: Delete Experiment {experiment_id}")

    response = requests.delete(f"{BASE_URL}/api/experiments/{experiment_id}")
    print(f"Status Code: {response.status_code}")

    assert response.status_code == 204, "Failed to delete experiment"
    print("✅ Delete Experiment: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENTS API TEST SUITE")
    print("=" * 60)

    try:
        # Test workflow
        experiment_id = test_create_experiment()
        test_list_experiments()
        test_get_experiment(experiment_id)
        test_update_experiment(experiment_id)
        test_delete_experiment(experiment_id)

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
