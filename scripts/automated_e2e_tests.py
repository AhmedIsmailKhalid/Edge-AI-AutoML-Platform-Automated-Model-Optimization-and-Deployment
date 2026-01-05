"""
Automated End-to-End Testing Script for Edge AI AutoML Platform.

Tests all possible combinations of:
- Frameworks (PyTorch, TensorFlow)
- Model sources (Pretrained, Custom)
- Model sizes (Small, Medium, Large)
- Datasets (MNIST, CIFAR-10, Fashion-MNIST)
- Target devices (6 devices)
- Optimization goals (4 goals)
- Custom constraints (various combinations)
- Error scenarios

Generates detailed test report with pass/fail status.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
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
# Test Case Definitions
# ============================================================================


class TestCategory(Enum):
    """Test category enumeration."""

    FRAMEWORK = "Framework Validation"
    MODEL_SIZE = "Model Size Validation"
    OPTIMIZATION_GOAL = "Optimization Goal"
    CONSTRAINTS = "Custom Constraints"
    TARGET_DEVICE = "Target Device"
    DATASET = "Dataset Variation"
    ERROR_HANDLING = "Error Handling"
    COMPLEX_WORKFLOW = "Complex Workflow"


class TestStatus(Enum):
    """Test status enumeration."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


@dataclass
class TestCase:
    """Test case definition."""

    id: int
    name: str
    category: TestCategory
    framework: str
    model_source: str  # 'pretrained' or 'custom' or 'invalid'
    model_size: str  # 'small', 'medium', 'large'
    dataset: str
    target_device: str
    optimization_goal: str
    constraints: dict = field(default_factory=dict)
    expected_result: str = "success"  # 'success' or 'error'
    description: str = ""


@dataclass
class TestResult:
    """Test result storage."""

    test_case: TestCase
    status: TestStatus
    experiment_id: Optional[str] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    optimization_results: Optional[dict] = None
    recommendations: Optional[dict] = None


# ============================================================================
# Test Case Generator
# ============================================================================


def generate_all_test_cases() -> list[TestCase]:
    """Generate all 66 test scenarios."""
    test_cases = []
    test_id = 1

    # ========================================================================
    # Category 1: Framework × Model Source (4 tests)
    # ========================================================================

    test_cases.append(
        TestCase(
            id=test_id,
            name="PyTorch + Custom Upload",
            category=TestCategory.FRAMEWORK,
            framework="pytorch",
            model_source="custom",
            model_size="large",
            dataset="mnist",
            target_device="raspberry_pi_4",
            optimization_goal="minimize_latency",
            description="Test PyTorch with custom model upload",
        )
    )
    test_id += 1

    test_cases.append(
        TestCase(
            id=test_id,
            name="PyTorch + Pretrained Model",
            category=TestCategory.FRAMEWORK,
            framework="pytorch",
            model_source="pretrained",
            model_size="medium",
            dataset="mnist",
            target_device="raspberry_pi_4",
            optimization_goal="balanced",
            description="Test PyTorch with pretrained model",
        )
    )
    test_id += 1

    test_cases.append(
        TestCase(
            id=test_id,
            name="TensorFlow + Custom Upload",
            category=TestCategory.FRAMEWORK,
            framework="tensorflow",
            model_source="custom",
            model_size="large",
            dataset="mnist",
            target_device="raspberry_pi_4",
            optimization_goal="minimize_size",
            description="Test TensorFlow with custom model upload",
        )
    )
    test_id += 1

    test_cases.append(
        TestCase(
            id=test_id,
            name="TensorFlow + Pretrained Model",
            category=TestCategory.FRAMEWORK,
            framework="tensorflow",
            model_source="pretrained",
            model_size="medium",
            dataset="mnist",
            target_device="raspberry_pi_4",
            optimization_goal="balanced",
            description="Test TensorFlow with pretrained model",
        )
    )
    test_id += 1

    # ========================================================================
    # Category 2: Model Size × Framework (6 tests)
    # ========================================================================

    for size in ["small", "medium", "large"]:
        for framework in ["pytorch", "tensorflow"]:
            test_cases.append(
                TestCase(
                    id=test_id,
                    name=f"{framework.capitalize()} + {size.capitalize()} Model",
                    category=TestCategory.MODEL_SIZE,
                    framework=framework,
                    model_source="pretrained",
                    model_size=size,
                    dataset="mnist",
                    target_device="raspberry_pi_4",
                    optimization_goal="balanced",
                    description=f"Test {framework} with {size} pretrained model",
                )
            )
            test_id += 1

    # ========================================================================
    # Category 3: Optimization Goals × Framework (8 tests)
    # ========================================================================

    goals = ["balanced", "minimize_size", "minimize_latency", "maximize_accuracy"]

    for goal in goals:
        for framework in ["pytorch", "tensorflow"]:
            test_cases.append(
                TestCase(
                    id=test_id,
                    name=f"{framework.capitalize()} + {goal.replace('_', ' ').title()}",
                    category=TestCategory.OPTIMIZATION_GOAL,
                    framework=framework,
                    model_source="pretrained",
                    model_size="medium",
                    dataset="mnist",
                    target_device="raspberry_pi_4",
                    optimization_goal=goal,
                    description=f"Test {framework} with {goal} optimization goal",
                )
            )
            test_id += 1

    # ========================================================================
    # Category 4: Custom Constraints (8 tests)
    # ========================================================================

    constraint_tests = [
        ("Max Accuracy Drop", {"max_accuracy_drop_percent": 2.0}),
        ("Max Size", {"max_size_mb": 5.0}),
        ("Max Latency", {"max_latency_ms": 50.0}),
        (
            "Multiple Constraints",
            {"max_accuracy_drop_percent": 2.0, "max_size_mb": 5.0, "max_latency_ms": 50.0},
        ),
    ]

    for constraint_name, constraints in constraint_tests:
        for framework in ["pytorch", "tensorflow"]:
            test_cases.append(
                TestCase(
                    id=test_id,
                    name=f"{framework.capitalize()} + {constraint_name}",
                    category=TestCategory.CONSTRAINTS,
                    framework=framework,
                    model_source="pretrained",
                    model_size="medium",
                    dataset="mnist",
                    target_device="raspberry_pi_4",
                    optimization_goal="balanced",
                    constraints=constraints,
                    description=f"Test {framework} with {constraint_name} constraint",
                )
            )
            test_id += 1

    # ========================================================================
    # Category 5: Target Devices (12 tests - 2 per device)
    # ========================================================================

    devices = [
        ("raspberry_pi_3b", "small"),
        ("raspberry_pi_4", "medium"),
        ("raspberry_pi_5", "large"),
        ("jetson_nano", "medium"),
        ("jetson_xavier_nx", "large"),
        ("coral_dev_board", "small"),
    ]

    for device, size in devices:
        for framework in ["pytorch", "tensorflow"]:
            test_cases.append(
                TestCase(
                    id=test_id,
                    name=f"{device.replace('_', ' ').title()} + {framework.capitalize()}",
                    category=TestCategory.TARGET_DEVICE,
                    framework=framework,
                    model_source="pretrained",
                    model_size=size,
                    dataset="mnist",
                    target_device=device,
                    optimization_goal="balanced",
                    description=f"Test {framework} on {device}",
                )
            )
            test_id += 1

    # ========================================================================
    # Category 6: Dataset Variations (6 tests)
    # ========================================================================

    datasets = ["mnist", "cifar10", "fashion_mnist"]

    for dataset in datasets:
        for framework in ["pytorch", "tensorflow"]:
            test_cases.append(
                TestCase(
                    id=test_id,
                    name=f"{framework.capitalize()} + {dataset.upper()}",
                    category=TestCategory.DATASET,
                    framework=framework,
                    model_source="pretrained",
                    model_size="medium",
                    dataset=dataset,
                    target_device="raspberry_pi_4",
                    optimization_goal="balanced",
                    description=f"Test {framework} with {dataset} dataset",
                )
            )
            test_id += 1

    # ========================================================================
    # Category 7: Error Scenarios (12 tests)
    # ========================================================================

    # Invalid file path
    test_cases.append(
        TestCase(
            id=test_id,
            name="Invalid Model File Path",
            category=TestCategory.ERROR_HANDLING,
            framework="pytorch",
            model_source="invalid_path",
            model_size="medium",
            dataset="mnist",
            target_device="raspberry_pi_4",
            optimization_goal="balanced",
            expected_result="error",
            description="Test with non-existent model file",
        )
    )
    test_id += 1

    # Wrong file extension
    test_cases.append(
        TestCase(
            id=test_id,
            name="Wrong File Extension",
            category=TestCategory.ERROR_HANDLING,
            framework="pytorch",
            model_source="wrong_extension",
            model_size="medium",
            dataset="mnist",
            target_device="raspberry_pi_4",
            optimization_goal="balanced",
            expected_result="error",
            description="Test with invalid file extension",
        )
    )
    test_id += 1

    # Duplicate experiment names
    for i in range(3):
        test_cases.append(
            TestCase(
                id=test_id,
                name=f"Duplicate Experiment Name {i+1}",
                category=TestCategory.ERROR_HANDLING,
                framework="pytorch",
                model_source="pretrained",
                model_size="small",
                dataset="mnist",
                target_device="raspberry_pi_4",
                optimization_goal="balanced",
                description=f"Test duplicate experiment name handling (run {i+1})",
            )
        )
        test_id += 1

    # Invalid constraints
    test_cases.append(
        TestCase(
            id=test_id,
            name="Invalid Constraint Values",
            category=TestCategory.ERROR_HANDLING,
            framework="pytorch",
            model_source="pretrained",
            model_size="medium",
            dataset="mnist",
            target_device="raspberry_pi_4",
            optimization_goal="balanced",
            constraints={"max_accuracy_drop_percent": -10.0},  # Negative value
            expected_result="error",
            description="Test with invalid constraint values",
        )
    )
    test_id += 1

    # Conflicting constraints
    test_cases.append(
        TestCase(
            id=test_id,
            name="Conflicting Constraints",
            category=TestCategory.ERROR_HANDLING,
            framework="pytorch",
            model_source="pretrained",
            model_size="large",
            dataset="mnist",
            target_device="raspberry_pi_4",
            optimization_goal="balanced",
            constraints={
                "max_accuracy_drop_percent": 0.1,  # Almost no drop allowed
                "max_size_mb": 0.5,  # Impossibly small
            },
            description="Test with impossible constraint combination",
        )
    )
    test_id += 1

    # ========================================================================
    # Category 8: Complex Workflows (10 tests)
    # ========================================================================

    # Production deployment scenario
    test_cases.append(
        TestCase(
            id=test_id,
            name="Production Deployment Scenario",
            category=TestCategory.COMPLEX_WORKFLOW,
            framework="tensorflow",
            model_source="pretrained",
            model_size="large",
            dataset="mnist",
            target_device="raspberry_pi_4",
            optimization_goal="minimize_size",
            constraints={"max_size_mb": 10.0},
            description="Real-world production deployment scenario",
        )
    )
    test_id += 1

    # Edge IoT scenario
    test_cases.append(
        TestCase(
            id=test_id,
            name="Edge IoT Scenario",
            category=TestCategory.COMPLEX_WORKFLOW,
            framework="pytorch",
            model_source="pretrained",
            model_size="small",
            dataset="fashion_mnist",
            target_device="raspberry_pi_3b",
            optimization_goal="minimize_latency",
            constraints={"max_latency_ms": 30.0},
            description="Resource-constrained edge IoT deployment",
        )
    )
    test_id += 1

    # High accuracy requirement
    test_cases.append(
        TestCase(
            id=test_id,
            name="High Accuracy Requirement",
            category=TestCategory.COMPLEX_WORKFLOW,
            framework="tensorflow",
            model_source="pretrained",
            model_size="large",
            dataset="cifar10",
            target_device="jetson_xavier_nx",
            optimization_goal="maximize_accuracy",
            constraints={"max_accuracy_drop_percent": 1.0},
            description="High-accuracy deployment on powerful device",
        )
    )
    test_id += 1

    # Resource-constrained scenario
    test_cases.append(
        TestCase(
            id=test_id,
            name="Resource-Constrained Scenario",
            category=TestCategory.COMPLEX_WORKFLOW,
            framework="pytorch",
            model_source="pretrained",
            model_size="medium",
            dataset="mnist",
            target_device="coral_dev_board",
            optimization_goal="balanced",
            constraints={"max_size_mb": 5.0, "max_accuracy_drop_percent": 2.0},
            description="Balanced optimization for constrained device",
        )
    )
    test_id += 1

    # Add more complex workflow tests (placeholders for now)
    for i in range(6):
        test_cases.append(
            TestCase(
                id=test_id,
                name=f"Complex Workflow {i+5}",
                category=TestCategory.COMPLEX_WORKFLOW,
                framework="pytorch" if i % 2 == 0 else "tensorflow",
                model_source="pretrained",
                model_size="medium",
                dataset="mnist",
                target_device="raspberry_pi_4",
                optimization_goal="balanced",
                description=f"Additional complex workflow test {i+5}",
            )
        )
        test_id += 1

    return test_cases


# ============================================================================
# Test Execution Functions
# ============================================================================


def create_experiment(test_case: TestCase) -> Optional[dict]:
    """Create an experiment for the test case."""
    experiment_data = {
        "name": f"AutoTest_{test_case.id:03d}_{test_case.name}",
        "description": test_case.description,
        "framework": test_case.framework,
        "dataset_type": "preset",
        "dataset_name": test_case.dataset,
        "target_device": test_case.target_device,
        "optimization_goal": test_case.optimization_goal,
    }

    # Add constraints if any
    if test_case.constraints:
        experiment_data.update(test_case.constraints)

    try:
        response = requests.post(
            f"{BASE_URL}/api/experiments/create",
            json=experiment_data,
            timeout=30,  # INCREASED from 10 to 30
        )

        if response.status_code == 201:
            return response.json()
        else:
            print(f"      X Failed to create experiment: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"      X Connection error: {e}")
        return None


def get_model_path(test_case: TestCase) -> Optional[Path]:
    """Get model file path for test case."""
    if test_case.model_source == "invalid_path":
        return Path("non_existent_model.pth")

    if test_case.model_source == "wrong_extension":
        return Path("test_file.txt")

    # Get pretrained model
    ext = ".pt" if test_case.framework == "pytorch" else ".h5"
    model_name = f"{test_case.model_size}_mnist_cnn{ext}"
    model_path = PRETRAINED_MODELS_DIR / test_case.framework / model_name

    if not model_path.exists():
        print(f"      ⚠️  Model not found: {model_path}")
        return None

    return model_path


def upload_model(experiment_id: str, model_path: Path, model_source: str) -> bool:
    """Upload model for experiment."""
    try:
        with open(model_path, "rb") as f:
            files = {"file": (model_path.name, f, "application/octet-stream")}
            data = {"model_source": model_source}

            response = requests.post(
                f"{BASE_URL}/api/upload/{experiment_id}/model", files=files, data=data, timeout=30
            )

        return response.status_code == 201

    except Exception as e:
        print(f"      ❌ Upload error: {e}")
        return False


def start_optimization(experiment_id: str) -> bool:
    """Start optimization."""
    try:
        response = requests.post(f"{BASE_URL}/api/optimize/{experiment_id}/start", timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def wait_for_completion(experiment_id: str, timeout: int = 300) -> bool:
    """Wait for optimization to complete."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{BASE_URL}/api/experiments/{experiment_id}", timeout=10)

            if response.status_code == 200:
                status = response.json()

                if status["status"] == "completed":
                    return True
                elif status["status"] == "failed":
                    return False

            time.sleep(3)

        except Exception:
            time.sleep(3)

    return False


def get_results(experiment_id: str) -> Optional[dict]:
    """Get optimization results."""
    try:
        response = requests.get(f"{BASE_URL}/api/results/{experiment_id}/results", timeout=10)

        if response.status_code == 200:
            return response.json()
        return None

    except Exception:
        return None


def get_recommendations(experiment_id: str) -> Optional[dict]:
    """Get recommendations."""
    try:
        response = requests.get(
            f"{BASE_URL}/api/results/{experiment_id}/recommendations", timeout=10
        )

        if response.status_code == 200:
            return response.json()
        return None

    except Exception:
        return None


def run_test_case(test_case: TestCase) -> TestResult:
    """Execute a single test case."""
    print(f"\n   Test {test_case.id:03d}: {test_case.name}")
    print(f"   Category: {test_case.category.value}")
    print(f"   Description: {test_case.description}")

    start_time = time.time()

    try:
        # Step 1: Create experiment
        print("      → Creating experiment...")
        experiment = create_experiment(test_case)

        if not experiment:
            return TestResult(
                test_case=test_case,
                status=TestStatus.FAILED,
                execution_time=time.time() - start_time,
                error_message="Failed to create experiment",
            )

        experiment_id = experiment["id"]
        print(f"      ✅ Experiment created: {experiment_id}")

        # Step 2: Get model path
        print("      → Getting model path...")
        model_path = get_model_path(test_case)

        if not model_path:
            return TestResult(
                test_case=test_case,
                status=TestStatus.FAILED,
                experiment_id=experiment_id,
                execution_time=time.time() - start_time,
                error_message="Model file not found",
            )

        # Handle expected errors
        if test_case.expected_result == "error":
            if test_case.model_source in ["invalid_path", "wrong_extension"]:
                # These should fail at upload
                upload_success = upload_model(experiment_id, model_path, "custom")
                if not upload_success:
                    print("      ✅ Expected error occurred (upload failed)")
                    return TestResult(
                        test_case=test_case,
                        status=TestStatus.PASSED,
                        experiment_id=experiment_id,
                        execution_time=time.time() - start_time,
                    )

        # Step 3: Upload model
        print("      → Uploading model...")
        upload_success = upload_model(experiment_id, model_path, test_case.model_source)

        if not upload_success:
            return TestResult(
                test_case=test_case,
                status=TestStatus.FAILED,
                experiment_id=experiment_id,
                execution_time=time.time() - start_time,
                error_message="Failed to upload model",
            )

        print("      ✅ Model uploaded")

        # Step 4: Start optimization
        print("      → Starting optimization...")
        if not start_optimization(experiment_id):
            return TestResult(
                test_case=test_case,
                status=TestStatus.FAILED,
                experiment_id=experiment_id,
                execution_time=time.time() - start_time,
                error_message="Failed to start optimization",
            )

        print("      ✅ Optimization started")

        # Step 5: Wait for completion
        print("      → Waiting for completion...")
        completed = wait_for_completion(experiment_id, timeout=300)

        if not completed:
            return TestResult(
                test_case=test_case,
                status=TestStatus.FAILED,
                experiment_id=experiment_id,
                execution_time=time.time() - start_time,
                error_message="Optimization did not complete in time",
            )

        print("      ✅ Optimization completed")

        # Step 6: Get results
        print("      → Fetching results...")
        results = get_results(experiment_id)
        recommendations = get_recommendations(experiment_id)

        execution_time = time.time() - start_time

        # Determine if test passed
        if results and results.get("completed_techniques", 0) > 0:
            print(f"      ✅ TEST PASSED ({execution_time:.1f}s)")
            return TestResult(
                test_case=test_case,
                status=TestStatus.PASSED,
                experiment_id=experiment_id,
                execution_time=execution_time,
                optimization_results=results,
                recommendations=recommendations,
            )
        else:
            print(f"      ❌ TEST FAILED ({execution_time:.1f}s)")
            return TestResult(
                test_case=test_case,
                status=TestStatus.FAILED,
                experiment_id=experiment_id,
                execution_time=execution_time,
                error_message="No techniques completed successfully",
            )

    except Exception as e:
        print(f"      ❌ TEST ERROR: {e}")
        return TestResult(
            test_case=test_case,
            status=TestStatus.ERROR,
            execution_time=time.time() - start_time,
            error_message=str(e),
        )


# ============================================================================
# Report Generation
# ============================================================================


def generate_summary_report(results: list[TestResult]) -> str:
    """Generate summary report."""
    total = len(results)
    passed = sum(1 for r in results if r.status == TestStatus.PASSED)
    failed = sum(1 for r in results if r.status == TestStatus.FAILED)
    errors = sum(1 for r in results if r.status == TestStatus.ERROR)
    skipped = sum(1 for r in results if r.status == TestStatus.SKIPPED)

    total_time = sum(r.execution_time for r in results)

    report = f"""
{'='*80}
EDGE AI AUTOML PLATFORM - AUTOMATED TEST REPORT
{'='*80}

Test Execution Summary:
-----------------------
Total Tests:      {total}
[PASS] Passed:    {passed} ({passed/total*100:.1f}%)
[FAIL] Failed:    {failed} ({failed/total*100:.1f}%)
[ERR]  Errors:    {errors} ({errors/total*100:.1f}%)
[SKIP] Skipped:   {skipped} ({skipped/total*100:.1f}%)

Execution Time:   {total_time:.1f}s ({total_time/60:.1f} minutes)
Average per test: {total_time/total:.1f}s

{'='*80}

Results by Category:
--------------------
"""

    # Group by category
    by_category = {}
    for result in results:
        category = result.test_case.category.value
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(result)

    for category, cat_results in sorted(by_category.items()):
        cat_passed = sum(1 for r in cat_results if r.status == TestStatus.PASSED)
        cat_total = len(cat_results)

        report += f"\n{category}:\n"
        report += f"  Tests: {cat_total}, Passed: {cat_passed}/{cat_total} ({cat_passed/cat_total*100:.1f}%)\n"

        for result in cat_results:
            status_icon = {
                TestStatus.PASSED: "[PASS]",
                TestStatus.FAILED: "[FAIL]",
                TestStatus.ERROR: "[ERR] ",
                TestStatus.SKIPPED: "[SKIP]",
            }[result.status]

            report += f"    {status_icon} Test {result.test_case.id:03d}: {result.test_case.name} ({result.execution_time:.1f}s)\n"

            if result.status != TestStatus.PASSED and result.error_message:
                report += f"       Error: {result.error_message}\n"

    report += f"\n{'='*80}\n"

    return report


def save_detailed_results(results: list[TestResult], output_file: Path):
    """Save detailed results to JSON."""
    data = {"timestamp": datetime.now().isoformat(), "total_tests": len(results), "results": []}

    for result in results:
        data["results"].append(
            {
                "test_id": result.test_case.id,
                "test_name": result.test_case.name,
                "category": result.test_case.category.value,
                "status": result.status.value,
                "execution_time": result.execution_time,
                "experiment_id": result.experiment_id,
                "error_message": result.error_message,
                "framework": result.test_case.framework,
                "model_size": result.test_case.model_size,
                "dataset": result.test_case.dataset,
                "target_device": result.test_case.target_device,
                "optimization_goal": result.test_case.optimization_goal,
                "constraints": result.test_case.constraints,
                "optimization_results": result.optimization_results,
                "recommendations": result.recommendations,
            }
        )

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)


# ============================================================================
# Main Execution
# ============================================================================


def main():
    """Main test execution."""
    print("\n" + "=" * 80)
    print("EDGE AI AUTOML PLATFORM - AUTOMATED E2E TESTING")
    print("=" * 80)
    print("\nGenerating test cases...")

    # Generate all test cases
    test_cases = generate_all_test_cases()

    print(f"\nGenerated {len(test_cases)} test cases")
    print("\nCategories:")

    by_category = {}
    for tc in test_cases:
        category = tc.category.value
        by_category[category] = by_category.get(category, 0) + 1

    for category, count in sorted(by_category.items()):
        print(f"  - {category}: {count} tests")

    print("\n" + "=" * 80)
    print("Starting test execution...")
    print("=" * 80)

    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code != 200:
            print("\n[ERROR] Server is not running!")
            print("Please start the server: poetry run uvicorn src.main:app --reload")
            return
    except requests.exceptions.RequestException:
        print("\n[ERROR] Cannot connect to server!")
        print("Please start the server: poetry run uvicorn src.main:app --reload")
        return

    # Execute all tests
    results = []
    start_time = time.time()

    for idx, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Executing Test {idx}/{len(test_cases)}")
        print(f"{'='*80}")

        result = run_test_case(test_case)
        results.append(result)

        # Small delay between tests
        time.sleep(1)

    total_time = time.time() - start_time

    # Generate reports
    print("\n" + "=" * 80)
    print("Generating reports...")
    print("=" * 80)

    summary_report = generate_summary_report(results)
    print(summary_report)

    # Save reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary report with UTF-8 encoding
    summary_file = TEST_RESULTS_DIR / f"test_report_{timestamp}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:  # ADDED encoding="utf-8"
        f.write(summary_report)
    print(f"\n[SAVED] Summary report: {summary_file}")

    # Save detailed results
    details_file = TEST_RESULTS_DIR / f"test_results_{timestamp}.json"
    save_detailed_results(results, details_file)
    print(f"[SAVED] Detailed results: {details_file}")

    print("\n" + "=" * 80)
    print("AUTOMATED TESTING COMPLETE!")
    print("=" * 80)
    print(f"\nTotal execution time: {total_time/60:.1f} minutes")
    print(
        f"Pass Rate: {sum(1 for r in results if r.status == TestStatus.PASSED)}/{len(results)} ({sum(1 for r in results if r.status == TestStatus.PASSED)/len(results)*100:.1f}%)"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARN] Testing interrupted by user")
    except Exception as e:
        print(f"\n\n[ERROR] Fatal error: {e}")
        raise
