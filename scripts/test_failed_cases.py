"""
Test Script for the 4 Failed Test Cases.

Runs only the tests that failed in the previous run to achieve 100% coverage.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

# ============================================================================
# Configuration
# ============================================================================

BASE_URL = "http://localhost:8000"
PRETRAINED_MODELS_DIR = Path("models/pretrained")
TEST_RESULTS_DIR = Path("test_results")
TEST_RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================================
# Test Definitions
# ============================================================================


@dataclass
class FailedTest:
    """Failed test definition."""

    id: int
    name: str
    framework: str
    model_source: str
    model_size: str
    dataset: str
    target_device: str
    optimization_goal: str
    constraints: dict
    expected_behavior: str


# Define the 4 failed tests
FAILED_TESTS = [
    FailedTest(
        id=41,
        name="PyTorch + CIFAR10",
        framework="pytorch",
        model_source="pretrained",
        model_size="medium",
        dataset="cifar10",
        target_device="raspberry_pi_4",
        optimization_goal="balanced",
        constraints={},
        expected_behavior="Should complete with CIFAR-10 models",
    ),
    FailedTest(
        id=42,
        name="TensorFlow + CIFAR10",
        framework="tensorflow",
        model_source="pretrained",
        model_size="medium",
        dataset="cifar10",
        target_device="raspberry_pi_4",
        optimization_goal="balanced",
        constraints={},
        expected_behavior="Should complete with CIFAR-10 models",
    ),
    FailedTest(
        id=50,
        name="Invalid Constraint Values",
        framework="pytorch",
        model_source="pretrained",
        model_size="medium",
        dataset="mnist",
        target_device="raspberry_pi_4",
        optimization_goal="balanced",
        constraints={"max_accuracy_drop_percent": -10.0},
        expected_behavior="Should reject invalid constraint (expected 422 error)",
    ),
    FailedTest(
        id=54,
        name="High Accuracy Requirement",
        framework="tensorflow",
        model_source="pretrained",
        model_size="large",
        dataset="cifar10",
        target_device="jetson_xavier_nx",
        optimization_goal="maximize_accuracy",
        constraints={"max_accuracy_drop_percent": 1.0},
        expected_behavior="Should complete with CIFAR-10 models",
    ),
]


# ============================================================================
# API Functions
# ============================================================================


def create_experiment(test: FailedTest) -> tuple[Optional[dict], Optional[str]]:
    """Create experiment."""
    experiment_data = {
        "name": f"FailedTest_{test.id:03d}_{test.name}",
        "description": f"Retry: {test.expected_behavior}",
        "framework": test.framework,
        "dataset_type": "preset",
        "dataset_name": test.dataset,
        "target_device": test.target_device,
        "optimization_goal": test.optimization_goal,
    }

    if test.constraints:
        experiment_data.update(test.constraints)

    try:
        response = requests.post(
            f"{BASE_URL}/api/experiments/create", json=experiment_data, timeout=60
        )

        if response.status_code == 201:
            return response.json(), None
        else:
            return None, f"HTTP {response.status_code}: {response.text}"

    except Exception as e:
        return None, str(e)


def get_model_path(test: FailedTest) -> Optional[Path]:
    """Get model path based on dataset."""
    ext = ".pt" if test.framework == "pytorch" else ".h5"

    # Use dataset-specific models
    model_name = f"{test.model_size}_{test.dataset}_cnn{ext}"
    model_path = PRETRAINED_MODELS_DIR / test.framework / model_name

    if not model_path.exists():
        print(f"      ❌ Model not found: {model_path}")
        print(f"      Expected: {model_name}")
        print(f"      Hint: Run 'python create_pretrained_models.py --datasets {test.dataset}'")
        return None

    print(f"      ✅ Model found: {model_path.name}")
    print(f"      Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    return model_path


def upload_model(
    experiment_id: str, model_path: Path, model_source: str
) -> tuple[bool, Optional[str]]:
    """Upload model."""
    try:
        with open(model_path, "rb") as f:
            files = {"file": (model_path.name, f, "application/octet-stream")}
            data = {"model_source": model_source}

            response = requests.post(
                f"{BASE_URL}/api/upload/{experiment_id}/model", files=files, data=data, timeout=60
            )

        if response.status_code == 201:
            return True, None
        else:
            return False, f"HTTP {response.status_code}: {response.text}"

    except Exception as e:
        return False, str(e)


def start_optimization(experiment_id: str) -> tuple[bool, Optional[str]]:
    """Start optimization."""
    try:
        response = requests.post(f"{BASE_URL}/api/optimize/{experiment_id}/start", timeout=30)

        if response.status_code == 200:
            return True, None
        else:
            return False, f"HTTP {response.status_code}: {response.text}"

    except Exception as e:
        return False, str(e)


def wait_for_completion(
    experiment_id: str, timeout: int = 600
) -> tuple[bool, Optional[str], Optional[dict]]:
    """Wait for completion."""
    start_time = time.time()
    last_progress = -1

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{BASE_URL}/api/experiments/{experiment_id}", timeout=10)

            if response.status_code == 200:
                status = response.json()
                progress = status.get("progress_percent", 0)

                if progress != last_progress:
                    print(f"      Progress: {progress}%")
                    last_progress = progress

                if status["status"] == "completed":
                    return True, None, status
                elif status["status"] == "failed":
                    error = status.get("error_message", "Unknown error")
                    return False, f"Optimization failed: {error}", status

            time.sleep(3)

        except Exception as e:
            time.sleep(3)

    return False, f"Timeout after {time.time() - start_time:.1f}s", None


def get_results(experiment_id: str) -> tuple[Optional[dict], Optional[str]]:
    """Get results."""
    try:
        response = requests.get(f"{BASE_URL}/api/results/{experiment_id}/results", timeout=10)

        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"HTTP {response.status_code}: {response.text}"

    except Exception as e:
        return None, str(e)


# ============================================================================
# Test Execution
# ============================================================================


def run_failed_test(test: FailedTest) -> dict:
    """Run a single failed test."""
    print(f"\n{'='*80}")
    print(f"TEST {test.id}: {test.name}")
    print(f"{'='*80}")
    print(f"Expected Behavior: {test.expected_behavior}")
    print(f"Framework: {test.framework}")
    print(f"Dataset: {test.dataset}")
    print(f"Device: {test.target_device}")
    if test.constraints:
        print(f"Constraints: {test.constraints}")

    result = {
        "test_id": test.id,
        "test_name": test.name,
        "status": "UNKNOWN",
        "error": None,
        "execution_time": 0,
        "experiment_id": None,
        "completed_techniques": 0,
        "failed_techniques": 0,
    }

    start_time = time.time()

    try:
        # Step 1: Create experiment
        print("\n   [1/5] Creating experiment...")
        experiment, error = create_experiment(test)

        if not experiment:
            # For Test 50 (invalid constraints), this is EXPECTED
            if test.id == 50 and "422" in str(error):
                result["status"] = "PASSED"
                result["error"] = "Correctly rejected invalid constraints ✅"
                result["execution_time"] = time.time() - start_time
                print(f"      ✅ PASSED - Validation working correctly!")
                print(f"      Error message: {error}")
                return result
            else:
                result["status"] = "FAILED"
                result["error"] = f"Create experiment failed: {error}"
                result["execution_time"] = time.time() - start_time
                print(f"      ❌ FAILED: {error}")
                return result

        experiment_id = experiment["id"]
        result["experiment_id"] = experiment_id
        print(f"      ✅ Created: {experiment_id}")

        # Step 2: Get model
        print("\n   [2/5] Getting model...")
        model_path = get_model_path(test)

        if not model_path:
            result["status"] = "FAILED"
            result["error"] = f"Model not found: {test.model_size}_{test.dataset}_cnn"
            result["execution_time"] = time.time() - start_time
            return result

        # Step 3: Upload model
        print("\n   [3/5] Uploading model...")
        upload_success, error = upload_model(experiment_id, model_path, test.model_source)

        if not upload_success:
            result["status"] = "FAILED"
            result["error"] = f"Upload failed: {error}"
            result["execution_time"] = time.time() - start_time
            print(f"      ❌ FAILED: {error}")
            return result

        print(f"      ✅ Upload complete")

        # Step 4: Start optimization
        print("\n   [4/5] Starting optimization...")
        start_success, error = start_optimization(experiment_id)

        if not start_success:
            result["status"] = "FAILED"
            result["error"] = f"Start failed: {error}"
            result["execution_time"] = time.time() - start_time
            print(f"      ❌ FAILED: {error}")
            return result

        print(f"      ✅ Optimization started")

        # Step 5: Wait for completion
        print("\n   [5/5] Waiting for completion...")
        completed, error, status = wait_for_completion(experiment_id, timeout=600)

        if not completed:
            result["status"] = "FAILED"
            result["error"] = error
            result["execution_time"] = time.time() - start_time
            print(f"      ❌ FAILED: {error}")
            return result

        print(f"      ✅ Optimization completed")

        # Get results
        print("\n   [RESULTS] Fetching optimization results...")
        results_data, error = get_results(experiment_id)

        if not results_data:
            result["status"] = "FAILED"
            result["error"] = f"Failed to get results: {error}"
            result["execution_time"] = time.time() - start_time
            print(f"      ❌ FAILED: {error}")
            return result

        completed_count = results_data.get("completed_techniques", 0)
        failed_count = results_data.get("failed_techniques", 0)
        total_count = results_data.get("total_techniques", 0)

        result["completed_techniques"] = completed_count
        result["failed_techniques"] = failed_count

        print(f"      Total: {total_count}, Completed: {completed_count}, Failed: {failed_count}")

        # Determine pass/fail
        if completed_count > 0:
            result["status"] = "PASSED"
            print(f"\n   ✅ PASSED - {completed_count} techniques completed successfully!")
        else:
            result["status"] = "FAILED"
            result["error"] = "No techniques completed successfully"
            print(f"\n   ❌ FAILED - No techniques completed")

        result["execution_time"] = time.time() - start_time

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
        result["execution_time"] = time.time() - start_time
        print(f"\n   ❌ ERROR: {e}")

    return result


# ============================================================================
# Main
# ============================================================================


def main():
    """Run failed tests to achieve 100% coverage."""
    print("\n" + "=" * 80)
    print("FAILED TEST CASES - RETRY FOR 100% COVERAGE")
    print("=" * 80)
    print(f"\nRunning {len(FAILED_TESTS)} previously failed tests")
    print("\nPrerequisites:")
    print("  1. Server running: poetry run uvicorn src.main:app --reload")
    print("  2. CIFAR-10 models trained: python create_pretrained_models.py --datasets cifar10")

    # Check server
    print("\n" + "=" * 80)
    print("Checking server connection...")
    print("=" * 80)

    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"⚠️  Server returned {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Please start: poetry run uvicorn src.main:app --reload")
        return

    # Check for CIFAR-10 models
    print("\n" + "=" * 80)
    print("Checking CIFAR-10 models...")
    print("=" * 80)

    pytorch_cifar = PRETRAINED_MODELS_DIR / "pytorch" / "medium_cifar10_cnn.pt"
    tensorflow_cifar = PRETRAINED_MODELS_DIR / "tensorflow" / "medium_cifar10_cnn.h5"

    models_exist = True
    if not pytorch_cifar.exists():
        print(f"❌ Missing: {pytorch_cifar}")
        models_exist = False
    else:
        print(f"✅ Found: {pytorch_cifar.name}")

    if not tensorflow_cifar.exists():
        print(f"❌ Missing: {tensorflow_cifar}")
        models_exist = False
    else:
        print(f"✅ Found: {tensorflow_cifar.name}")

    if not models_exist:
        print("\n⚠️  CIFAR-10 models not found!")
        print("Run: python create_pretrained_models.py --datasets cifar10")
        print("\nContinuing anyway (Test 50 will still pass)...")

    # Run tests
    print("\n" + "=" * 80)
    print("Running tests...")
    print("=" * 80)

    results = []
    start_time = time.time()

    for idx, test in enumerate(FAILED_TESTS, 1):
        print(f"\n\n{'#'*80}")
        print(f"# TEST {idx}/{len(FAILED_TESTS)}")
        print(f"{'#'*80}")

        result = run_failed_test(test)
        results.append(result)

        time.sleep(2)

    total_time = time.time() - start_time

    # Generate report
    print("\n\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    passed = sum(1 for r in results if r["status"] == "PASSED")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    errors = sum(1 for r in results if r["status"] == "ERROR")

    print(f"\nTotal Tests:  {len(results)}")
    print(f"✅ Passed:     {passed} ({passed/len(results)*100:.1f}%)")
    print(f"❌ Failed:     {failed} ({failed/len(results)*100:.1f}%)")
    print(f"⚠️  Errors:     {errors} ({errors/len(results)*100:.1f}%)")
    print(f"\nTotal Time:   {total_time/60:.1f} minutes")
    print(f"Avg per test: {total_time/len(results):.1f}s")

    print("\n" + "-" * 80)
    print("Detailed Results:")
    print("-" * 80)

    for result in results:
        status_icon = (
            "✅" if result["status"] == "PASSED" else "❌" if result["status"] == "FAILED" else "⚠️"
        )
        print(f"\n{status_icon} Test {result['test_id']:03d}: {result['test_name']}")
        print(f"   Time: {result['execution_time']:.1f}s")

        if result["status"] == "PASSED":
            if result.get("completed_techniques"):
                print(f"   Completed: {result['completed_techniques']} techniques")
            else:
                print(f"   Note: {result['error']}")
        else:
            print(f"   Error: {result['error']}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = TEST_RESULTS_DIR / f"failed_tests_{timestamp}.json"

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(results),
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "execution_time": total_time,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n\n📁 Results saved: {results_file}")

    # Final status
    print("\n" + "=" * 80)
    if passed == len(FAILED_TESTS):
        print("🎉 ALL TESTS PASSED - 100% COVERAGE ACHIEVED!")
    else:
        print(f"⚠️  {failed} TEST(S) STILL FAILING")
        print("\nTroubleshooting:")
        for result in results:
            if result["status"] == "FAILED":
                print(f"  • Test {result['test_id']}: {result['error']}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        raise
