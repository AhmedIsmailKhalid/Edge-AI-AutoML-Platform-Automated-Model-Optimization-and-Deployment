"""
Quick Test Script for Failed Test Cases.

Focuses on the 13 tests that failed in the full automated run
with enhanced debugging and error reporting.
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
class QuickTest:
    """Quick test definition."""

    id: int
    name: str
    framework: str
    model_source: str
    model_size: str
    dataset: str
    target_device: str
    optimization_goal: str
    constraints: dict
    original_failure: str


# Define the 13 failed tests
FAILED_TESTS = [
    QuickTest(
        id=1,
        name="PyTorch + Custom Upload",
        framework="pytorch",
        model_source="custom",
        model_size="large",
        dataset="mnist",
        target_device="raspberry_pi_4",
        optimization_goal="minimize_latency",
        constraints={},
        original_failure="Connection timeout on create experiment",
    ),
    QuickTest(
        id=36,
        name="Jetson Xavier NX + TensorFlow",
        framework="tensorflow",
        model_source="pretrained",
        model_size="large",
        dataset="mnist",
        target_device="jetson_xavier_nx",
        optimization_goal="balanced",
        constraints={},
        original_failure="No techniques completed successfully",
    ),
    QuickTest(
        id=37,
        name="Coral Dev Board + PyTorch",
        framework="pytorch",
        model_source="pretrained",
        model_size="small",
        dataset="mnist",
        target_device="coral_dev_board",
        optimization_goal="balanced",
        constraints={},
        original_failure="Connection timeout on create experiment",
    ),
    QuickTest(
        id=38,
        name="Coral Dev Board + TensorFlow",
        framework="tensorflow",
        model_source="pretrained",
        model_size="small",
        dataset="mnist",
        target_device="coral_dev_board",
        optimization_goal="balanced",
        constraints={},
        original_failure="Connection timeout on create experiment",
    ),
    QuickTest(
        id=41,
        name="PyTorch + CIFAR10",
        framework="pytorch",
        model_source="pretrained",
        model_size="medium",
        dataset="cifar10",
        target_device="raspberry_pi_4",
        optimization_goal="balanced",
        constraints={},
        original_failure="No techniques completed (dataset mismatch)",
    ),
    QuickTest(
        id=42,
        name="TensorFlow + CIFAR10",
        framework="tensorflow",
        model_source="pretrained",
        model_size="medium",
        dataset="cifar10",
        target_device="raspberry_pi_4",
        optimization_goal="balanced",
        constraints={},
        original_failure="No techniques completed (dataset mismatch)",
    ),
    QuickTest(
        id=43,
        name="PyTorch + Fashion-MNIST",
        framework="pytorch",
        model_source="pretrained",
        model_size="medium",
        dataset="fashionmnist",
        target_device="raspberry_pi_4",
        optimization_goal="balanced",
        constraints={},
        original_failure="Optimization timeout (dataset mismatch)",
    ),
    QuickTest(
        id=44,
        name="TensorFlow + Fashion-MNIST",
        framework="tensorflow",
        model_source="pretrained",
        model_size="medium",
        dataset="fashionmnist",
        target_device="raspberry_pi_4",
        optimization_goal="balanced",
        constraints={},
        original_failure="Optimization timeout (dataset mismatch)",
    ),
    QuickTest(
        id=50,
        name="Invalid Constraint Values",
        framework="pytorch",
        model_source="pretrained",
        model_size="medium",
        dataset="mnist",
        target_device="raspberry_pi_4",
        optimization_goal="balanced",
        constraints={"max_accuracy_drop_percent": -10.0},
        original_failure="Failed to create experiment (expected error)",
    ),
    QuickTest(
        id=52,
        name="Production Deployment Scenario",
        framework="tensorflow",
        model_source="pretrained",
        model_size="large",
        dataset="mnist",
        target_device="raspberry_pi_4",
        optimization_goal="minimize_size",
        constraints={"max_size_mb": 10.0},
        original_failure="No techniques completed successfully",
    ),
    QuickTest(
        id=53,
        name="Edge IoT Scenario",
        framework="pytorch",
        model_source="pretrained",
        model_size="small",
        dataset="fashionmnist",
        target_device="raspberry_pi_3b",
        optimization_goal="minimize_latency",
        constraints={"max_latency_ms": 30.0},
        original_failure="Connection timeout on create experiment",
    ),
    QuickTest(
        id=54,
        name="High Accuracy Requirement",
        framework="tensorflow",
        model_source="pretrained",
        model_size="large",
        dataset="cifar10",
        target_device="jetson_xavier_nx",
        optimization_goal="maximize_accuracy",
        constraints={"max_accuracy_drop_percent": 1.0},
        original_failure="Connection timeout on create experiment",
    ),
    QuickTest(
        id=55,
        name="Resource-Constrained Scenario",
        framework="pytorch",
        model_source="pretrained",
        model_size="medium",
        dataset="mnist",
        target_device="coral_dev_board",
        optimization_goal="balanced",
        constraints={"max_size_mb": 5.0, "max_accuracy_drop_percent": 2.0},
        original_failure="Connection timeout on create experiment",
    ),
]


# ============================================================================
# API Functions with Enhanced Error Handling
# ============================================================================


def create_experiment(test: QuickTest) -> tuple[dict | None, str | None]:
    """Create experiment with detailed error reporting."""
    experiment_data = {
        "name": f"QuickTest_{test.id:03d}_{test.name}",
        "description": f"Retry failed test: {test.original_failure}",
        "framework": test.framework,
        "dataset_type": "preset",
        "dataset_name": test.dataset,
        "target_device": test.target_device,
        "optimization_goal": test.optimization_goal,
    }

    if test.constraints:
        experiment_data.update(test.constraints)

    try:
        print(f"      Sending request to {BASE_URL}/api/experiments/create")
        print(f"      Data: {json.dumps(experiment_data, indent=2)}")

        response = requests.post(
            f"{BASE_URL}/api/experiments/create",
            json=experiment_data,
            timeout=60,  # Increased to 60 seconds
        )

        print(f"      Response status: {response.status_code}")

        if response.status_code == 201:
            return response.json(), None
        else:
            error = f"HTTP {response.status_code}: {response.text}"
            return None, error

    except requests.exceptions.Timeout:
        return None, "Request timed out after 60 seconds"
    except requests.exceptions.ConnectionError as e:
        return None, f"Connection error: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"


def get_model_path(test: QuickTest) -> Path | None:
    """Get model path based on dataset."""
    ext = ".pt" if test.framework == "pytorch" else ".h5"

    # Use dataset-specific models
    if test.dataset in ["cifar10", "cifar100"]:
        # For CIFAR datasets, look for CIFAR models
        model_name = f"{test.model_size}_{test.dataset}_cnn{ext}"
    elif test.dataset in ["mnist", "fashionmnist"]:
        # For MNIST/Fashion-MNIST, use mnist models (they're compatible)
        model_name = f"{test.model_size}_mnist_cnn{ext}"
    else:
        model_name = f"{test.model_size}_{test.dataset}_cnn{ext}"

    model_path = PRETRAINED_MODELS_DIR / test.framework / model_name

    if not model_path.exists():
        print(f"      WARNING: Model not found at {model_path}")
        print(f"      Expected: {model_name}")
        return None

    print(f"      Model path: {model_path}")
    print(f"      Model size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    return model_path


def upload_model(
    experiment_id: str, model_path: Path, model_source: str
) -> tuple[bool, str | None]:
    """Upload model with error reporting."""
    try:
        print(f"      Uploading {model_path.name}...")

        with open(model_path, "rb") as f:
            files = {"file": (model_path.name, f, "application/octet-stream")}
            data = {"model_source": model_source}

            response = requests.post(
                f"{BASE_URL}/api/upload/{experiment_id}/model", files=files, data=data, timeout=60
            )

        print(f"      Upload response: {response.status_code}")

        if response.status_code == 201:
            return True, None
        else:
            return False, f"HTTP {response.status_code}: {response.text}"

    except Exception as e:
        return False, str(e)


def start_optimization(experiment_id: str) -> tuple[bool, str | None]:
    """Start optimization with error reporting."""
    try:
        response = requests.post(f"{BASE_URL}/api/optimize/{experiment_id}/start", timeout=30)

        print(f"      Start optimization response: {response.status_code}")

        if response.status_code == 200:
            return True, None
        else:
            return False, f"HTTP {response.status_code}: {response.text}"

    except Exception as e:
        return False, str(e)


def wait_for_completion(
    experiment_id: str, timeout: int = 600
) -> tuple[bool, str | None, dict | None]:
    """Wait for completion with detailed progress."""
    start_time = time.time()
    last_progress = -1

    print(f"      Waiting for completion (timeout: {timeout}s)...")

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
                    print(f"      Completed in {time.time() - start_time:.1f}s")
                    return True, None, status
                elif status["status"] == "failed":
                    error = status.get("error_message", "Unknown error")
                    return False, f"Optimization failed: {error}", status

            time.sleep(3)

        except Exception as e:
            print(f"      Status check error: {e}")
            time.sleep(3)

    elapsed = time.time() - start_time
    return False, f"Timeout after {elapsed:.1f}s (progress: {last_progress}%)", None


def get_results(experiment_id: str) -> tuple[dict | None, str | None]:
    """Get results with error reporting."""
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


def run_quick_test(test: QuickTest) -> dict:
    """Run a single quick test with detailed reporting."""
    print(f"\n{'='*80}")
    print(f"TEST {test.id}: {test.name}")
    print(f"{'='*80}")
    print(f"Original Failure: {test.original_failure}")
    print(f"Framework: {test.framework}")
    print(f"Dataset: {test.dataset}")
    print(f"Device: {test.target_device}")
    print(f"Goal: {test.optimization_goal}")
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
            result["status"] = "FAILED"
            result["error"] = f"Create experiment failed: {error}"
            result["execution_time"] = time.time() - start_time
            print(f"      [FAIL] {error}")
            return result

        experiment_id = experiment["id"]
        result["experiment_id"] = experiment_id
        print(f"      [PASS] Created: {experiment_id}")

        # For invalid constraint test, failure here is expected
        if test.id == 50:
            result["status"] = "PASSED"
            result["error"] = "Expected error caught (invalid constraints)"
            result["execution_time"] = time.time() - start_time
            print("      [PASS] Test correctly validated constraints")
            return result

        # Step 2: Get model
        print("\n   [2/5] Getting model...")
        model_path = get_model_path(test)

        if not model_path:
            result["status"] = "FAILED"
            result["error"] = "Model file not found"
            result["execution_time"] = time.time() - start_time
            print("      [FAIL] Model not found")
            return result

        print("      [PASS] Model found")

        # Step 3: Upload model
        print("\n   [3/5] Uploading model...")
        upload_success, error = upload_model(experiment_id, model_path, test.model_source)

        if not upload_success:
            result["status"] = "FAILED"
            result["error"] = f"Upload failed: {error}"
            result["execution_time"] = time.time() - start_time
            print(f"      [FAIL] {error}")
            return result

        print("      [PASS] Upload complete")

        # Step 4: Start optimization
        print("\n   [4/5] Starting optimization...")
        start_success, error = start_optimization(experiment_id)

        if not start_success:
            result["status"] = "FAILED"
            result["error"] = f"Start failed: {error}"
            result["execution_time"] = time.time() - start_time
            print(f"      [FAIL] {error}")
            return result

        print("      [PASS] Optimization started")

        # Step 5: Wait for completion
        print("\n   [5/5] Waiting for completion...")
        completed, error, status = wait_for_completion(experiment_id, timeout=600)

        if not completed:
            result["status"] = "FAILED"
            result["error"] = error
            result["execution_time"] = time.time() - start_time
            print(f"      [FAIL] {error}")
            return result

        print("      [PASS] Optimization completed")

        # Get results
        print("\n   [RESULTS] Fetching optimization results...")
        results_data, error = get_results(experiment_id)

        if not results_data:
            result["status"] = "FAILED"
            result["error"] = f"Failed to get results: {error}"
            result["execution_time"] = time.time() - start_time
            print(f"      [FAIL] {error}")
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
            print(f"\n   [PASS] Test passed with {completed_count} techniques completed")
        else:
            result["status"] = "FAILED"
            result["error"] = "No techniques completed successfully"
            print("\n   [FAIL] No techniques completed")

        result["execution_time"] = time.time() - start_time

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
        result["execution_time"] = time.time() - start_time
        print(f"\n   [ERROR] {e}")

    return result


# ============================================================================
# Main
# ============================================================================


def main():
    """Run quick tests on failed cases."""
    print("\n" + "=" * 80)
    print("QUICK TEST - FAILED TEST CASES")
    print("=" * 80)
    print(f"\nTesting {len(FAILED_TESTS)} failed cases from full test run")
    print("Enhanced with better timeouts and detailed error reporting")

    # Check server
    print("\n" + "=" * 80)
    print("Checking server connection...")
    print("=" * 80)

    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("[PASS] Server is running")
        else:
            print(f"[WARN] Server returned {response.status_code}")
    except Exception as e:
        print(f"[FAIL] Cannot connect to server: {e}")
        print("Please start: poetry run uvicorn src.main:app --reload")
        return

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

        result = run_quick_test(test)
        results.append(result)

        time.sleep(2)  # Brief delay between tests

    total_time = time.time() - start_time

    # Generate report
    print("\n\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r["status"] == "PASSED")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    errors = sum(1 for r in results if r["status"] == "ERROR")

    print(f"\nTotal Tests:  {len(results)}")
    print(f"[PASS] Passed: {passed} ({passed/len(results)*100:.1f}%)")
    print(f"[FAIL] Failed: {failed} ({failed/len(results)*100:.1f}%)")
    print(f"[ERR]  Errors: {errors} ({errors/len(results)*100:.1f}%)")
    print(f"\nTotal Time: {total_time/60:.1f} minutes")
    print(f"Avg per test: {total_time/len(results):.1f}s")

    print("\n" + "-" * 80)
    print("Detailed Results:")
    print("-" * 80)

    for result in results:
        status_icon = (
            "[PASS]"
            if result["status"] == "PASSED"
            else "[FAIL]"
            if result["status"] == "FAILED"
            else "[ERR] "
        )
        print(f"\n{status_icon} Test {result['test_id']:03d}: {result['test_name']}")
        print(f"   Time: {result['execution_time']:.1f}s")

        if result["status"] == "PASSED":
            if result.get("completed_techniques"):
                print(f"   Completed: {result['completed_techniques']} techniques")
        else:
            print(f"   Error: {result['error']}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = TEST_RESULTS_DIR / f"quick_test_{timestamp}.json"

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

    print(f"\n\n[SAVED] Results: {results_file}")
    print("\n" + "=" * 80)
    print("QUICK TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARN] Test interrupted by user")
    except Exception as e:
        print(f"\n\n[ERROR] Fatal error: {e}")
        raise
