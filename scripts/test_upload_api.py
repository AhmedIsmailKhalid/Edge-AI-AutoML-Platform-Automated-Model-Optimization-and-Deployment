"""
Test script for Upload API endpoints.
"""

import io

import requests

BASE_URL = "http://localhost:8000"


def create_test_experiment():
    """Create a test experiment."""
    print("\n🧪 Creating test experiment...")

    payload = {
        "name": "Upload Test Experiment",
        "description": "Testing file upload",
        "model_name": "test_model",
        "framework": "pytorch",
        "dataset_type": "preset",
        "dataset_name": "cifar10",
        "target_device": "raspberry_pi_4",
    }

    response = requests.post(f"{BASE_URL}/api/experiments/create", json=payload)
    assert response.status_code == 201

    experiment_id = response.json()["id"]
    print(f"✅ Created experiment: {experiment_id}")
    return experiment_id


def test_upload_model(experiment_id):
    """Test uploading a model file."""
    print(f"\n🧪 Testing: Upload Model to {experiment_id}")

    # Create a dummy model file
    dummy_model = b"dummy pytorch model content"
    files = {"file": ("test_model.pth", io.BytesIO(dummy_model), "application/octet-stream")}
    data = {"model_source": "custom", "custom_name": "my_test_model"}

    response = requests.post(f"{BASE_URL}/api/upload/{experiment_id}/model", files=files, data=data)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    assert response.status_code == 201, "Failed to upload model"
    data = response.json()
    assert data["model_name"] == "my_test_model"
    print("✅ Upload Model: PASSED")


def test_upload_invalid_model(experiment_id):
    """Test uploading invalid model file."""
    print("\n🧪 Testing: Upload Invalid Model")

    # Create a file with wrong extension
    dummy_file = b"invalid file content"
    files = {"file": ("test_model.txt", io.BytesIO(dummy_file), "text/plain")}
    data = {
        "model_source": "custom",
    }

    response = requests.post(f"{BASE_URL}/api/upload/{experiment_id}/model", files=files, data=data)

    print(f"Status Code: {response.status_code}")
    assert response.status_code == 400, "Should reject invalid file"
    print("✅ Invalid Model Rejection: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("UPLOAD API TEST SUITE")
    print("=" * 60)

    try:
        experiment_id = create_test_experiment()
        test_upload_model(experiment_id)
        test_upload_invalid_model(experiment_id)

        print("\n" + "=" * 60)
        print("✅ ALL UPLOAD TESTS PASSED!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
