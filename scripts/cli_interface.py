"""
CLI Interface for Edge AI AutoML Platform.

Provides an interactive command-line interface for users to:
- Create experiments
- Upload or select pretrained models
- Configure optimization settings
- Monitor progress
- View results
"""

import json

# import sys
import time
from pathlib import Path

import requests

# Configuration
BASE_URL = "http://localhost:8000"
PRETRAINED_MODELS_DIR = Path("models/pretrained")


# ============================================================================
# Helper Functions
# ============================================================================


def print_header():
    """Print application header."""
    print("\n" + "=" * 80)
    print("🚀 EDGE AI AUTOML PLATFORM")
    print("   Intelligent Model Optimization for Edge Devices")
    print("=" * 80 + "\n")


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80 + "\n")


def print_success(message: str):
    """Print success message."""
    print(f"✅ {message}")


def print_error(message: str):
    """Print error message."""
    print(f"❌ {message}")


def print_info(message: str):
    """Print info message."""
    print(f"ℹ️  {message}")


def get_input(prompt: str, default: str = None) -> str:
    """Get user input with optional default."""
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "

    value = input(full_prompt).strip()
    return value if value else default


def get_choice(prompt: str, choices: list[str]) -> str:
    """Get user choice from a list."""
    while True:
        value = input(f"{prompt}: ").strip()
        if value in choices:
            return value
        print(f"❌ Invalid choice. Please select from: {', '.join(choices)}")


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no confirmation."""
    default_str = "Y/n" if default else "y/N"
    value = input(f"{prompt} [{default_str}]: ").strip().lower()

    if not value:
        return default

    return value in ["y", "yes"]


# ============================================================================
# API Interaction Functions
# ============================================================================


def create_experiment(experiment_data: dict) -> dict | None:
    """Create a new experiment via API."""
    try:
        response = requests.post(
            f"{BASE_URL}/api/experiments/create", json=experiment_data, timeout=10
        )
        if response.status_code == 201:
            return response.json()
        else:
            print_error(f"Failed to create experiment: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print_error(f"Connection error: {e}")
        print_info("Make sure the server is running: poetry run uvicorn src.main:app --reload")
        return None


def upload_model(
    experiment_id: str,
    file_path: Path = None,
    model_source: str = "custom",
    pretrained_model_name: str = None,
) -> dict | None:
    """Upload a model file or load pretrained model."""
    try:
        if model_source == "pretrained":
            # For pretrained models
            if not pretrained_model_name:
                print_error("pretrained_model_name is required for pretrained models")
                return None

            data = {"model_source": "pretrained", "pretrained_model_name": pretrained_model_name}

            response = requests.post(
                f"{BASE_URL}/api/upload/{experiment_id}/model", data=data, timeout=30
            )
        else:
            # For custom models
            if not file_path or not file_path.exists():
                print_error(f"File not found: {file_path}")
                return None

            with open(file_path, "rb") as f:
                files = {"file": (file_path.name, f, "application/octet-stream")}
                data = {"model_source": model_source}

                response = requests.post(
                    f"{BASE_URL}/api/upload/{experiment_id}/model",
                    files=files,
                    data=data,
                    timeout=30,
                )

        if response.status_code == 201:
            return response.json()
        else:
            print_error(f"Failed to upload model: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print_error(f"Connection error: {e}")
        return None
    except FileNotFoundError:
        print_error(f"File not found: {file_path}")
        return None


def start_optimization(experiment_id: str) -> bool:
    """Start optimization for an experiment."""
    try:
        response = requests.post(f"{BASE_URL}/api/optimize/{experiment_id}/start", timeout=10)
        if response.status_code == 200:
            return True
        else:
            print_error(f"Failed to start optimization: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print_error(f"Connection error: {e}")
        return False


def get_experiment_status(experiment_id: str) -> dict | None:
    """Get experiment status."""
    try:
        response = requests.get(f"{BASE_URL}/api/experiments/{experiment_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def get_results(experiment_id: str, max_retries: int = 3) -> dict | None:
    """Get optimization results with retry logic."""
    url = f"{BASE_URL}/api/results/{experiment_id}/results"

    for attempt in range(max_retries):
        try:
            # Progressive timeout: 15s, 30s, 45s
            timeout = 15 * (attempt + 1)

            if attempt > 0:
                print(f"⏳ Retry {attempt}/{max_retries} (timeout: {timeout}s)...")

            response = requests.get(url, timeout=timeout)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️  API returned status {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                continue

        except requests.exceptions.Timeout:
            print(f"⚠️  Request timed out after {timeout}s")
            if attempt < max_retries - 1:
                print(f"⏳ Waiting {2 ** attempt}s before retry...")
                time.sleep(2**attempt)
            continue

        except requests.exceptions.ConnectionError:
            print(f"⚠️  Cannot connect to {BASE_URL}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            continue

        except Exception as e:
            print(f"⚠️  Error: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            continue

    print(f"❌ Failed to get results after {max_retries} attempts")
    print(f"ℹ️  Manual check: http://localhost:8000/api/results/{experiment_id}/results")
    return None


def get_recommendations(experiment_id: str, max_retries: int = 3) -> dict | None:
    """Get recommendations with retry logic."""
    url = f"{BASE_URL}/api/results/{experiment_id}/recommendations"

    for attempt in range(max_retries):
        try:
            timeout = 15 * (attempt + 1)
            response = requests.get(url, timeout=timeout)

            if response.status_code == 200:
                return response.json()
            else:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                continue

        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            continue

    return None


def download_model(experiment_id: str, technique_name: str, output_path: str) -> bool:
    """Download optimized model."""
    try:
        response = requests.get(
            f"{BASE_URL}/api/results/{experiment_id}/download/{technique_name}",
            timeout=30,
            stream=True,
        )

        if response.status_code == 200:
            # Save model to file
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True
        else:
            print_error(f"Failed to download model: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print_error(f"Connection error: {e}")
        return False


# ============================================================================
# Menu Functions
# ============================================================================


def get_experiment_name() -> str:
    """Prompt for experiment name."""
    return get_input("Please enter Experiment Name")


def select_target_device() -> str:
    """Prompt for target device selection."""
    print_section("SELECT TARGET DEVICE")

    devices = [
        ("raspberry_pi_3b", "Raspberry Pi 3B", "1.2GHz Quad-core, 1GB RAM"),
        ("raspberry_pi_4", "Raspberry Pi 4", "1.5GHz Quad-core, 4GB RAM"),
        ("raspberry_pi_5", "Raspberry Pi 5", "2.4GHz Quad-core, 8GB RAM"),
        ("jetson_nano", "NVIDIA Jetson Nano", "Maxwell GPU, 4GB RAM"),
        ("jetson_xavier_nx", "NVIDIA Jetson Xavier NX", "Volta GPU, 8GB RAM"),
        ("coral_dev_board", "Google Coral Dev Board", "Edge TPU, 1GB RAM"),
    ]

    for idx, (_key, name, specs) in enumerate(devices, 1):
        print(f"{idx}. {name} ({specs})")

    choice = get_choice(
        f"\nSelect device [1-{len(devices)}]", [str(i) for i in range(1, len(devices) + 1)]
    )

    return devices[int(choice) - 1][0]


def select_optimization_goal() -> str:
    """Prompt for optimization goal."""
    print_section("SELECT OPTIMIZATION GOAL")

    goals = [
        ("balanced", "Balanced", "Best trade-off between accuracy, size, and speed"),
        ("minimize_size", "Minimize Size", "Smallest possible model"),
        ("minimize_latency", "Maximize Speed", "Fastest inference time"),
        ("maximize_accuracy", "Maximize Accuracy", "Preserve accuracy at all costs"),
    ]

    for idx, (_key, name, desc) in enumerate(goals, 1):
        print(f"  {idx}. {name} ({desc})")

    choice = get_choice(
        f"\nSelect optimization goal [1-{len(goals)}]", [str(i) for i in range(1, len(goals) + 1)]
    )

    return goals[int(choice) - 1][0]


def select_model_choice() -> str:
    """Prompt for model selection method."""
    print_section("SELECT MODEL CHOICE")

    print("  1. Upload Custom Model")
    print("  2. Use Pretrained Model")

    choice = get_choice("\nSelect option [1-2]", ["1", "2"])

    return "upload" if choice == "1" else "pretrained"


def select_framework() -> str:
    """Prompt for framework selection."""
    print_section("SELECT FRAMEWORK")

    print("  1. PyTorch")
    print("  2. TensorFlow")

    choice = get_choice("\nSelect framework [1-2]", ["1", "2"])

    return "pytorch" if choice == "1" else "tensorflow"


def select_pretrained_model(framework: str) -> Path | None:
    """Select a pretrained model - FILTERED to only show actual pretrained models."""
    print_section(f"SELECT PRETRAINED {framework.upper()} MODEL")

    models_dir = PRETRAINED_MODELS_DIR / framework

    if not models_dir.exists():
        print_error(f"Pretrained models directory not found: {models_dir}")
        print_info("Please run: poetry run python scripts/create_pretrained_models.py")
        return None

    # Load metadata
    metadata_path = models_dir / "models_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata_list = json.load(f)
    else:
        metadata_list = []

    # List available models - FILTER to only pretrained models
    if framework == "pytorch":
        all_files = list(models_dir.glob("*.pt"))
    else:
        all_files = list(models_dir.glob("*.h5"))

    # Filter to only valid pretrained model names
    valid_sizes = ["small", "medium", "large"]
    valid_datasets = ["mnist", "fashionmnist", "cifar10", "cifar100"]

    model_files = []
    for file in all_files:
        # Expected format: {size}_{dataset}_cnn.pt or {size}_{dataset}_cnn.h5
        parts = file.stem.split("_")

        # Check if it matches the pattern
        if len(parts) >= 3:
            size = parts[0]
            # Dataset might have underscores (e.g., fashion_mnist)
            dataset = "_".join(parts[1:-1])  # Everything between size and "cnn"
            model_type = parts[-1]

            # Normalize dataset name (remove underscores)
            dataset_normalized = dataset.replace("_", "").lower()

            if size in valid_sizes and dataset_normalized in valid_datasets and model_type == "cnn":
                model_files.append(file)
            else:
                # This is not a valid pretrained model - skip it
                print(f"  ℹ️  Skipping non-pretrained file: {file.name}")

    # Sort by name for consistent display
    model_files = sorted(model_files, key=lambda f: f.name)

    if not model_files:
        print_error(f"No pretrained models found in {models_dir}")
        print_info("Run: poetry run python scripts/create_pretrained_models.py")
        return None

    # Display models with metadata
    print("\nAvailable pretrained models:")
    for idx, model_file in enumerate(model_files, 1):
        meta = next((m for m in metadata_list if m["model_name"] == model_file.stem), None)
        if meta:
            print(
                f"  {idx}. {model_file.name} ({meta['parameters']:,} params, {meta['file_size_mb']} MB)"
            )
        else:
            print(f"  {idx}. {model_file.name}")

    choice = get_choice(
        f"\nSelect model [1-{len(model_files)}]", [str(i) for i in range(1, len(model_files) + 1)]
    )

    selected_file = model_files[int(choice) - 1]
    print_success(f"Selected: {selected_file.name}")

    return selected_file


def upload_custom_model(framework: str) -> Path | None:
    """Prompt for custom model file path."""
    print_section("UPLOAD CUSTOM MODEL")

    file_ext = ".pt or .pth" if framework == "pytorch" else ".h5"
    print(f"Expected file format: {file_ext}\n")

    file_path_str = get_input("Enter full path to model file")
    file_path = Path(file_path_str)

    if not file_path.exists():
        print_error(f"File not found: {file_path}")
        return None

    if framework == "pytorch":
        valid_exts = [".pt", ".pth", ".pkl"]
        if file_path.suffix.lower() not in valid_exts:
            print_error(f"Invalid file format. Expected {valid_exts}, got {file_path.suffix}")
            return None
    else:
        if file_path.suffix.lower() != ".h5":
            print_error(f"Invalid file format. Expected .h5, got {file_path.suffix}")
            return None

    return file_path


def select_dataset_type() -> str:
    """Prompt for dataset type."""
    print_section("SELECT DATASET")

    print("  1. Preset Dataset")
    print("  2. Custom Dataset")

    choice = get_choice("\nSelect dataset type [1-2]", ["1", "2"])

    return "preset" if choice == "1" else "custom"


def select_preset_dataset() -> str:
    """Prompt for preset dataset."""
    print_section("SELECT PRESET DATASET")

    datasets = [
        ("mnist", "MNIST", "Handwritten digits (28x28 grayscale)"),
        ("cifar10", "CIFAR-10", "10-class color images (32x32)"),
        ("fashionmnist", "Fashion-MNIST", "Fashion items (28x28 grayscale)"),
    ]

    for idx, (_key, name, desc) in enumerate(datasets, 1):
        print(f"  {idx}. {name} ({desc})")

    choice = get_choice(
        f"\nSelect dataset [1-{len(datasets)}]", [str(i) for i in range(1, len(datasets) + 1)]
    )

    return datasets[int(choice) - 1][0]


def upload_dataset(experiment_id: str) -> dict | None:
    """Upload a custom dataset zip file."""
    try:
        # Get file path
        print_info("Enter the full path to your dataset zip file")
        print_info("Example: dataset/custom/my_dataset/my_dataset.zip")
        print()

        file_path_str = input("Dataset zip path: ").strip()
        file_path = Path(file_path_str.strip('"').strip("'"))  # Remove quotes if present

        if not file_path.exists():
            print_error(f"File not found: {file_path}")
            return None

        if not file_path.suffix.lower() == ".zip":
            print_error("File must be a .zip file")
            return None

        # Get dataset name from the zip file name (without .zip)
        dataset_name = file_path.stem

        print()
        print_info(f"Dataset name: {dataset_name}")
        print_info(f"Uploading from: {file_path}")
        print_info("This may take a few minutes depending on dataset size...")
        print()

        # Upload - the backend will extract it to dataset/custom/{dataset_name}/
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/zip")}
            data = {"dataset_name": dataset_name}

            response = requests.post(
                f"{BASE_URL}/api/upload/{experiment_id}/dataset",
                files=files,
                data=data,
                timeout=300,  # 5 minutes for large files
            )

        if response.status_code == 200:
            return response.json()
        else:
            print_error(f"Upload failed: {response.text}")
            return None

    except requests.exceptions.Timeout:
        print_error("Upload timed out. Dataset might be too large.")
        return None
    except Exception as e:
        print_error(f"Error uploading dataset: {e}")
        return None


def configure_custom_constraints() -> dict | None:
    """Configure custom constraints if user wants."""
    print_section("CUSTOM CONSTRAINTS")

    use_constraints = get_yes_no("Do you want to add custom constraints?", default=False)

    if not use_constraints:
        return None

    constraints = {}

    print("\nLeave blank to skip a constraint\n")

    # Max accuracy drop
    max_drop = get_input("Maximum accuracy drop (%)", default="")
    if max_drop:
        constraints["max_accuracy_drop_percent"] = float(max_drop)

    # Max size
    max_size = get_input("Maximum model size (MB)", default="")
    if max_size:
        constraints["max_size_mb"] = float(max_size)

    # Max latency
    max_latency = get_input("Maximum latency (ms)", default="")
    if max_latency:
        constraints["max_latency_ms"] = float(max_latency)

    return constraints if constraints else None


# ============================================================================
# Progress Monitoring
# ============================================================================


def monitor_optimization(experiment_id: str):
    """Monitor optimization progress with real-time technique tracking."""

    print_section("OPTIMIZATION IN PROGRESS")
    print()

    # Small delay to let backend start first technique
    time.sleep(2.5)

    last_progress = -1
    seen_techniques = {}  # Track {technique_name: last_status}
    printed_completed = set()  # Track which completed techniques we've already printed
    last_results_check = 0  # Timestamp of last results check
    RESULTS_CHECK_INTERVAL = 2.0  # Only check results every 2 seconds
    current_technique_printed = None  # Track which "Current Technique" line we've printed
    # progress_bar_printed = False  # Track if we've printed the initial progress bar

    # Technique display names
    technique_display_names = {
        "ptq_int8": "Post-Training Quantization (INT8)",
        "ptq_int4": "Post-Training Quantization (INT4)",
        "pruning_magnitude_unstructured": "Magnitude Pruning (Unstructured)",
        "pruning_magnitude_structured": "Magnitude Pruning (Structured)",
        "quantization_aware_training": "Quantization-Aware Training",
        "hybrid_prune_quantize": "Hybrid (Pruning + Quantization)",
        "distillation": "Knowledge Distillation",
    }

    while True:
        try:
            current_time = time.time()

            # Get experiment status (lightweight check)
            status = get_experiment_status(experiment_id)
            if not status:
                time.sleep(1.0)
                continue

            current_progress = status.get("progress_percent", 0)
            exp_status = status.get("status", "")

            # Only check results periodically to reduce DB load
            if current_time - last_results_check >= RESULTS_CHECK_INTERVAL:
                last_results_check = current_time

                try:
                    results_response = requests.get(
                        f"{BASE_URL}/api/results/{experiment_id}/results", timeout=5
                    )

                    if results_response.ok:
                        # API returns a LIST of results directly
                        results_data = results_response.json()

                        # Handle both list and dict responses for compatibility
                        if isinstance(results_data, list):
                            sorted_results = sorted(
                                results_data, key=lambda x: x.get("execution_order", 999)
                            )
                        else:
                            sorted_results = sorted(
                                results_data.get("results", []),
                                key=lambda x: x.get("optimization_run", {}).get(
                                    "execution_order", 999
                                ),
                            )

                        for result in sorted_results:
                            # Handle both direct result and nested optimization_run format
                            if "optimization_run" in result:
                                opt_run = result.get("optimization_run", {})
                            else:
                                opt_run = result

                            tech_name = opt_run.get("technique_name")
                            tech_status = opt_run.get("status", "").upper()

                            if not tech_name:
                                continue

                            # Get previous status
                            prev_status = seen_techniques.get(tech_name)
                            display_name = technique_display_names.get(tech_name, tech_name)

                            # State transition: None/PENDING → RUNNING
                            if tech_status == "RUNNING" and prev_status != "RUNNING":
                                # Only print if we haven't already printed this technique
                                if current_technique_printed != tech_name:
                                    print(f"🔧 Current Technique: {display_name}")
                                    current_technique_printed = tech_name
                                seen_techniques[tech_name] = "RUNNING"

                            # State transition: RUNNING → COMPLETED
                            elif tech_status == "COMPLETED" and prev_status == "RUNNING":
                                accuracy = opt_run.get("optimized_accuracy")
                                size_reduction = opt_run.get("size_reduction_percent")

                                if accuracy is not None and size_reduction is not None:
                                    print(
                                        f"✅ {display_name}: {accuracy*100:.1f}% accuracy, {size_reduction:.1f}% size reduction"
                                    )
                                elif accuracy is not None:
                                    print(f"✅ {display_name}: {accuracy*100:.1f}% accuracy")
                                else:
                                    print(f"✅ {display_name} completed")

                                printed_completed.add(tech_name)
                                seen_techniques[tech_name] = "COMPLETED"
                                current_technique_printed = None  # Reset for next technique

                            # Handle already-completed techniques (completed before monitoring started)
                            elif tech_status == "COMPLETED" and tech_name not in printed_completed:
                                accuracy = opt_run.get("optimized_accuracy")
                                size_reduction = opt_run.get("size_reduction_percent")

                                # Print "Current Technique" line first ONLY if not already printed
                                if current_technique_printed != tech_name:
                                    print(f"🔧 Current Technique: {display_name}")
                                    current_technique_printed = tech_name

                                # Then print completion
                                if accuracy is not None and size_reduction is not None:
                                    print(
                                        f"✅ {display_name}: {accuracy*100:.1f}% accuracy, {size_reduction:.1f}% size reduction"
                                    )
                                elif accuracy is not None:
                                    print(f"✅ {display_name}: {accuracy*100:.1f}% accuracy")
                                else:
                                    print(f"✅ {display_name} completed")

                                printed_completed.add(tech_name)
                                seen_techniques[tech_name] = "COMPLETED"
                                current_technique_printed = None  # Reset for next technique

                            # State transition: RUNNING → FAILED
                            elif tech_status == "FAILED" and prev_status == "RUNNING":
                                print(f"❌ {display_name} failed!")
                                seen_techniques[tech_name] = "FAILED"
                                current_technique_printed = None  # Reset for next technique

                            # Handle already-failed techniques
                            elif tech_status == "FAILED" and tech_name not in seen_techniques:
                                if current_technique_printed != tech_name:
                                    print(f"🔧 Current Technique: {display_name}")
                                    current_technique_printed = tech_name
                                print(f"❌ {display_name} failed!")
                                seen_techniques[tech_name] = "FAILED"
                                current_technique_printed = None  # Reset for next technique

                except requests.exceptions.Timeout:
                    # Skip this results check on timeout
                    pass
                except Exception:
                    # Skip this results check on error
                    pass

            # Update progress bar if changed
            if current_progress != last_progress:
                bar_length = 30
                filled = int(bar_length * current_progress / 100)
                bar = "█" * filled + "░" * (bar_length - filled)

                # Print progress bar on new line each time it updates
                print(f"\nOverall Progress: {current_progress}% |{bar}| {current_progress}/100")

                last_progress = current_progress
                # progress_bar_printed = True

            # Check if completed
            if exp_status == "completed":
                print()
                print_success("Optimization completed successfully!")
                break
            elif exp_status == "failed":
                print()
                print_error("Optimization failed!")
                return False

            time.sleep(0.5)  # Poll status every 0.5 seconds, but only check results every 2 seconds

        except requests.exceptions.Timeout:
            # Continue on timeout
            time.sleep(1.0)
            continue
        except Exception:
            # Log but continue on other errors
            time.sleep(1.0)
            continue

    return True


# ============================================================================
# Results Display
# ============================================================================


def display_results(experiment_id: str):
    """Display optimization results."""
    print_section("OPTIMIZATION RESULTS")

    # Get results with shorter timeout since optimization is complete
    try:
        response = requests.get(f"{BASE_URL}/api/results/{experiment_id}/results", timeout=10)
        if not response.ok:
            print_error(f"Failed to get results: {response.status_code}")
            return

        results_data = response.json()

        # Handle both list and dict responses
        if isinstance(results_data, list):
            results_list = results_data
        else:
            results_list = results_data.get("results", [])

    except Exception as e:
        print_error(f"Error fetching results: {e}")
        print_info(f"Check manually: http://localhost:8000/api/results/{experiment_id}/results")
        return

    if not results_list:
        print_info("No results available yet.")
        return

    # Display results table
    print(f"{'Technique':<40} {'Accuracy':<12} {'Size Reduction':<15} {'Status':<10}")
    print("=" * 80)

    for result in results_list:
        # Handle both formats
        if "optimization_run" in result:
            opt_run = result["optimization_run"]
        else:
            opt_run = result

        tech_name = opt_run.get("technique_name", "Unknown")
        status = opt_run.get("status", "unknown").upper()
        accuracy = opt_run.get("optimized_accuracy")
        size_reduction = opt_run.get("size_reduction_percent")

        # Format technique name
        display_name = tech_name.replace("_", " ").title()

        # Format accuracy
        acc_str = f"{accuracy*100:.1f}%" if accuracy is not None else "N/A"

        # Format size reduction
        size_str = f"{size_reduction:.1f}%" if size_reduction is not None else "N/A"

        # Format status
        status_symbol = "✅" if status == "COMPLETED" else "❌" if status == "FAILED" else "⏳"
        status_str = f"{status_symbol} {status}"

        print(f"{display_name:<40} {acc_str:<12} {size_str:<15} {status_str:<10}")

    print("=" * 80)
    print()

    # Get recommendations
    try:
        rec_response = requests.get(
            f"{BASE_URL}/api/results/{experiment_id}/recommendations", timeout=10
        )
        if rec_response.ok:
            recommendations = rec_response.json()

            if recommendations:
                print_section("TOP RECOMMENDATION")
                top_rec = recommendations[0]
                print(
                    f"✨ Recommended Technique: {top_rec.get('technique_name', 'N/A').replace('_', ' ').title()}"
                )
                print(f"   Accuracy: {top_rec.get('accuracy', 0)*100:.1f}%")
                print(f"   Reasoning: {top_rec.get('reasoning', 'N/A')}")
                print()
    except Exception:
        pass  # Skip recommendations if error


# ============================================================================
# Main Workflow
# ============================================================================


def main():
    """Main CLI workflow."""
    print_header()

    # Step 1: Experiment Configuration
    print_section("STEP 1: EXPERIMENT CONFIGURATION")

    experiment_name = get_experiment_name()
    target_device = select_target_device()
    optimization_goal = select_optimization_goal()

    # Step 2: Model Selection
    model_choice = select_model_choice()
    framework = select_framework()

    if model_choice == "pretrained":
        model_path = select_pretrained_model(framework)
        model_source = "pretrained"
    else:
        model_path = upload_custom_model(framework)
        model_source = "custom"

    if not model_path:
        print_error("Model selection failed. Exiting.")
        return

    # Step 3: Dataset Selection
    dataset_type = select_dataset_type()
    custom_dataset_path = None  # Store path for later upload

    if dataset_type == "preset":
        dataset_name = select_preset_dataset()
    else:
        # Get custom dataset path (will upload after experiment creation)
        print_section("CUSTOM DATASET PATH")
        print_info("You will upload the dataset after creating the experiment")
        print_info("Dataset structure: zip file with class folders containing images")
        print_info("Supported formats: .jpg, .jpeg, .png")
        print_info("Example: dataset/custom/my_dataset/my_dataset.zip")
        print()

        file_path_str = input("Enter path to dataset zip file: ").strip()
        custom_dataset_path = Path(file_path_str.strip('"').strip("'"))

        if not custom_dataset_path.exists():
            print_error(f"File not found: {custom_dataset_path}")
            return

        if not custom_dataset_path.suffix.lower() == ".zip":
            print_error("File must be a .zip file")
            return

        # Use stem as dataset name
        dataset_name = custom_dataset_path.stem
        print_success(f"Dataset will be named: {dataset_name}")

    # Step 4: Custom Constraints
    constraints = configure_custom_constraints()

    # Summary
    print_section("EXPERIMENT SUMMARY")
    print(f"Experiment Name: {experiment_name}")
    print(f"Framework: {framework}")
    print(f"Model: {model_path.name}")
    print(f"Dataset: {dataset_name}")
    print(f"Target Device: {target_device}")
    print(f"Optimization Goal: {optimization_goal}")
    if constraints:
        print(f"Constraints: {constraints}")

    print()
    proceed = get_yes_no("Start optimization?", default=True)

    if not proceed:
        print_info("Experiment cancelled.")
        return

    # Step 5: Create Experiment
    print_section("CREATING EXPERIMENT")

    experiment_data = {
        "name": experiment_name,
        "description": f"CLI experiment - {framework} - {dataset_name}",
        "framework": framework,
        "dataset_type": dataset_type,
        "dataset_name": dataset_name,
        "target_device": target_device,
        "optimization_goal": optimization_goal,
    }

    if constraints:
        experiment_data.update(constraints)

    experiment = create_experiment(experiment_data)

    if not experiment:
        print_error("Failed to create experiment. Exiting.")
        return

    experiment_id = experiment["id"]
    print_success(f"Experiment created: {experiment_id}")
    if dataset_type == "custom":
        print_section("UPLOADING CUSTOM DATASET")

        print_info(f"Dataset name: {dataset_name}")
        print_info(f"Uploading from: {custom_dataset_path}")
        print_info("This may take a few minutes...")
        print()

        # Upload the dataset
        with open(custom_dataset_path, "rb") as f:
            files = {"file": (custom_dataset_path.name, f, "application/zip")}
            data = {"dataset_name": dataset_name}

            response = requests.post(
                f"{BASE_URL}/api/upload/{experiment_id}/dataset",
                files=files,
                data=data,
                timeout=300,
            )

        if response.status_code != 200:
            print_error(f"Upload failed: {response.text}")
            return

        dataset_info = response.json()

        print_success(f"Dataset uploaded: {dataset_info['dataset_name']}")
        print_info(f"Classes: {len(dataset_info['classes'])}")
        print_info(f"Total images: {dataset_info['total_images']}")
        print()

        # Update dataset_name with actual uploaded name
        dataset_name = dataset_info["dataset_name"]

    # Step 6: Upload Model
    print_section("UPLOADING MODEL")

    if model_source == "pretrained":
        # Extract pretrained model name from path (without extension)
        pretrained_name = model_path.stem  # e.g., 'large_cifar10_cnn'
        upload_result = upload_model(
            experiment_id, model_source="pretrained", pretrained_model_name=pretrained_name
        )
    else:
        upload_result = upload_model(experiment_id, file_path=model_path, model_source="custom")

    if not upload_result:
        print_error("Failed to upload model. Exiting.")
        return

    print_success(f"Model uploaded: {upload_result['model_name']}")
    print_info(f"File size: {upload_result['file_size_mb']:.2f} MB")

    # Step 7: Start Optimization
    print_section("STARTING OPTIMIZATION")

    if not start_optimization(experiment_id):
        print_error("Failed to start optimization. Exiting.")
        return

    print_success("Optimization started!")

    # Step 8: Monitor Progress
    success = monitor_optimization(experiment_id)

    if not success:
        return

    # Step 9: Display Results
    display_results(experiment_id)

    # Done
    print("\n" + "=" * 80)
    print("✅ EXPERIMENT COMPLETE!")
    print(f"Experiment ID: {experiment_id}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        raise
