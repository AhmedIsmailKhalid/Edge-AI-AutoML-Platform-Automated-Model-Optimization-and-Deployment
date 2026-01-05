"""
Input validation helper functions.
"""


def validate_framework(framework: str) -> bool:
    """Validate framework name."""
    return framework.lower() in ["pytorch", "tensorflow"]


def validate_dataset_type(dataset_type: str) -> bool:
    """Validate dataset type."""
    return dataset_type.lower() in ["preset", "custom"]


def validate_preset_dataset(dataset_name: str) -> bool:
    """Validate preset dataset name."""
    valid_datasets = ["cifar10", "cifar100", "mnist", "fashionmnist"]
    return dataset_name.lower() in valid_datasets


def validate_target_device(device: str) -> bool:
    """Validate target device name."""
    valid_devices = [
        "raspberry_pi_4",
        "raspberry_pi_5",
        "raspberry_pi_zero_2w",
    ]
    return device.lower() in valid_devices


def validate_optimization_goal(goal: str) -> bool:
    """Validate optimization goal."""
    valid_goals = [
        "maximize_accuracy",
        "minimize_size",
        "minimize_latency",
        "balanced",
    ]
    return goal.lower() in valid_goals
