"""
Performance estimation for edge devices.

Estimates latency, memory usage, and power consumption for optimized models
on various edge devices without requiring physical hardware.
"""

from dataclasses import dataclass
from enum import Enum


class DeviceType(str, Enum):
    """Supported edge device types."""

    RASPBERRY_PI_3B = "raspberry_pi_3b"
    RASPBERRY_PI_4 = "raspberry_pi_4"
    RASPBERRY_PI_5 = "raspberry_pi_5"
    JETSON_NANO = "jetson_nano"
    JETSON_XAVIER_NX = "jetson_xavier_nx"
    CORAL_DEV_BOARD = "coral_dev_board"
    GENERIC_ARM_CORTEX_A53 = "generic_arm_cortex_a53"
    GENERIC_ARM_CORTEX_A72 = "generic_arm_cortex_a72"


@dataclass
class DeviceSpecs:
    """Hardware specifications for an edge device."""

    name: str
    cpu_cores: int
    cpu_freq_ghz: float
    ram_gb: float
    has_gpu: bool
    gpu_tflops: float | None = None
    has_npu: bool = False
    npu_tops: float | None = None
    power_watts: float = 5.0
    int8_ops_per_sec: float = 0
    fp32_ops_per_sec: float = 0


# Device specifications database
DEVICE_SPECS: dict[DeviceType, DeviceSpecs] = {
    DeviceType.RASPBERRY_PI_3B: DeviceSpecs(
        name="Raspberry Pi 3 Model B",
        cpu_cores=4,
        cpu_freq_ghz=1.2,
        ram_gb=1.0,
        has_gpu=False,
        power_watts=2.5,
        int8_ops_per_sec=1.2e9,
        fp32_ops_per_sec=0.3e9,
    ),
    DeviceType.RASPBERRY_PI_4: DeviceSpecs(
        name="Raspberry Pi 4",
        cpu_cores=4,
        cpu_freq_ghz=1.5,
        ram_gb=4.0,
        has_gpu=True,
        gpu_tflops=0.032,
        power_watts=3.0,
        int8_ops_per_sec=2.4e9,
        fp32_ops_per_sec=0.6e9,
    ),
    DeviceType.RASPBERRY_PI_5: DeviceSpecs(
        name="Raspberry Pi 5",
        cpu_cores=4,
        cpu_freq_ghz=2.4,
        ram_gb=8.0,
        has_gpu=True,
        gpu_tflops=0.1,
        power_watts=5.0,
        int8_ops_per_sec=4.8e9,
        fp32_ops_per_sec=1.2e9,
    ),
    DeviceType.JETSON_NANO: DeviceSpecs(
        name="NVIDIA Jetson Nano",
        cpu_cores=4,
        cpu_freq_ghz=1.43,
        ram_gb=4.0,
        has_gpu=True,
        gpu_tflops=0.472,
        power_watts=10.0,
        int8_ops_per_sec=5.0e9,
        fp32_ops_per_sec=1.5e9,
    ),
    DeviceType.JETSON_XAVIER_NX: DeviceSpecs(
        name="NVIDIA Jetson Xavier NX",
        cpu_cores=6,
        cpu_freq_ghz=1.9,
        ram_gb=8.0,
        has_gpu=True,
        gpu_tflops=1.4,
        power_watts=15.0,
        int8_ops_per_sec=20e9,
        fp32_ops_per_sec=6e9,
    ),
    DeviceType.CORAL_DEV_BOARD: DeviceSpecs(
        name="Google Coral Dev Board",
        cpu_cores=4,
        cpu_freq_ghz=1.5,
        ram_gb=1.0,
        has_gpu=False,
        has_npu=True,
        npu_tops=4.0,
        power_watts=3.0,
        int8_ops_per_sec=4e12,
        fp32_ops_per_sec=0.5e9,
    ),
    DeviceType.GENERIC_ARM_CORTEX_A53: DeviceSpecs(
        name="Generic ARM Cortex-A53",
        cpu_cores=4,
        cpu_freq_ghz=1.4,
        ram_gb=2.0,
        has_gpu=False,
        power_watts=2.0,
        int8_ops_per_sec=1.5e9,
        fp32_ops_per_sec=0.4e9,
    ),
    DeviceType.GENERIC_ARM_CORTEX_A72: DeviceSpecs(
        name="Generic ARM Cortex-A72",
        cpu_cores=4,
        cpu_freq_ghz=2.0,
        ram_gb=4.0,
        has_gpu=False,
        power_watts=4.0,
        int8_ops_per_sec=3.0e9,
        fp32_ops_per_sec=0.8e9,
    ),
}


@dataclass
class PerformanceEstimate:
    """Estimated performance metrics for a model on a device."""

    device_type: DeviceType
    device_name: str

    # Latency estimates
    estimated_latency_ms: float
    estimated_latency_min_ms: float
    estimated_latency_max_ms: float

    # Memory estimates
    estimated_memory_mb: float
    memory_utilization_percent: float

    # Power estimates
    estimated_power_watts: float
    estimated_energy_per_inference_mj: float

    # Throughput
    estimated_fps: float

    # Feasibility
    is_feasible: bool
    warnings: list[str]
    recommendations: list[str]


class PerformanceEstimator:
    """
    Estimates model performance on edge devices.

    Uses empirical models and device specifications to predict latency,
    memory usage, and power consumption without physical hardware.
    """

    def __init__(self, device_type: DeviceType):
        """
        Initialize performance estimator.

        Args:
            device_type: Target edge device type
        """
        self.device_type = device_type
        self.device_specs = DEVICE_SPECS[device_type]

    def estimate_performance(
        self,
        model_size_mb: float,
        params_count: int,
        framework: str,
        is_quantized: bool = False,
        quantization_bits: int = 32,
    ) -> PerformanceEstimate:
        """
        Estimate model performance on the target device.

        Args:
            model_size_mb: Model size in megabytes
            params_count: Number of model parameters
            framework: Model framework (pytorch/tensorflow)
            is_quantized: Whether model is quantized
            quantization_bits: Quantization bit width (8, 16, 32)

        Returns:
            PerformanceEstimate with predicted metrics
        """
        # Estimate compute operations
        compute_ops = self._estimate_operations(params_count)

        # Estimate latency
        latency_ms = self._estimate_latency(
            compute_ops=compute_ops, is_quantized=is_quantized, quantization_bits=quantization_bits
        )

        # Add variance for min/max estimates
        latency_variance = 0.2  # ±20% variance
        latency_min = latency_ms * (1 - latency_variance)
        latency_max = latency_ms * (1 + latency_variance)

        # Estimate memory usage
        memory_mb = self._estimate_memory(
            model_size_mb=model_size_mb, params_count=params_count, framework=framework
        )

        memory_utilization = (memory_mb / (self.device_specs.ram_gb * 1024)) * 100

        # Estimate power consumption
        power_watts = self._estimate_power(latency_ms=latency_ms, is_quantized=is_quantized)

        # Calculate energy per inference (millijoules)
        energy_per_inference = (power_watts * latency_ms) / 1000.0

        # Calculate throughput (FPS)
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0

        # Check feasibility
        is_feasible, warnings, recommendations = self._check_feasibility(
            memory_mb=memory_mb, memory_utilization=memory_utilization, latency_ms=latency_ms
        )

        return PerformanceEstimate(
            device_type=self.device_type,
            device_name=self.device_specs.name,
            estimated_latency_ms=latency_ms,
            estimated_latency_min_ms=latency_min,
            estimated_latency_max_ms=latency_max,
            estimated_memory_mb=memory_mb,
            memory_utilization_percent=memory_utilization,
            estimated_power_watts=power_watts,
            estimated_energy_per_inference_mj=energy_per_inference,
            estimated_fps=fps,
            is_feasible=is_feasible,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _estimate_operations(self, params_count: int) -> float:
        """
        Estimate number of compute operations.

        Args:
            params_count: Number of model parameters

        Returns:
            Estimated number of operations
        """
        # Rough estimate: ~2 operations per parameter (multiply-add)
        # Plus overhead for activations, batch norm, etc.
        return params_count * 2.5

    def _estimate_latency(
        self, compute_ops: float, is_quantized: bool, quantization_bits: int
    ) -> float:
        """
        Estimate inference latency.

        Args:
            compute_ops: Number of compute operations
            is_quantized: Whether model is quantized
            quantization_bits: Bit width of quantization

        Returns:
            Estimated latency in milliseconds
        """
        # Select appropriate operations per second based on quantization
        if is_quantized and quantization_bits == 8:
            ops_per_sec = self.device_specs.int8_ops_per_sec
        else:
            ops_per_sec = self.device_specs.fp32_ops_per_sec

        # Base latency calculation
        if ops_per_sec == 0:
            # Fallback calculation based on CPU specs
            ops_per_sec = (
                self.device_specs.cpu_cores
                * self.device_specs.cpu_freq_ghz
                * 1e9
                * 0.5  # Utilization factor
            )

        # Calculate compute time
        compute_time_seconds = compute_ops / ops_per_sec

        # Add overhead for memory access, data movement, etc.
        overhead_factor = 1.5

        # Convert to milliseconds
        latency_ms = compute_time_seconds * 1000 * overhead_factor

        # Hardware-specific adjustments
        if self.device_specs.has_npu:
            # NPU/TPU provides significant speedup for INT8
            if is_quantized and quantization_bits == 8:
                latency_ms *= 0.1  # 10x faster with TPU
        elif self.device_specs.has_gpu:
            # GPU provides moderate speedup
            latency_ms *= 0.6  # 1.67x faster with GPU

        return latency_ms

    def _estimate_memory(self, model_size_mb: float, params_count: int, framework: str) -> float:
        """
        Estimate memory usage during inference.

        Args:
            model_size_mb: Model file size
            params_count: Number of parameters
            framework: Model framework

        Returns:
            Estimated memory usage in MB
        """
        # Model weights
        model_memory = model_size_mb

        # Activation memory (rough estimate based on params)
        # Typical activation memory is 10-20% of parameter count
        activation_memory = (params_count * 4) / (1024 * 1024) * 0.15  # 15% estimate

        # Framework overhead
        if framework == "tensorflow":
            framework_overhead = 100  # TensorFlow has higher overhead
        else:  # pytorch
            framework_overhead = 50

        # Total memory
        total_memory = model_memory + activation_memory + framework_overhead

        return total_memory

    def _estimate_power(self, latency_ms: float, is_quantized: bool) -> float:
        """
        Estimate power consumption during inference.

        Args:
            latency_ms: Estimated latency
            is_quantized: Whether model is quantized

        Returns:
            Estimated power consumption in watts
        """
        # Base power consumption
        base_power = self.device_specs.power_watts

        # Compute power scales with utilization
        # Longer inference = more power
        utilization_factor = min(latency_ms / 100, 1.5)  # Cap at 1.5x

        compute_power = base_power * utilization_factor

        # Quantized models use less power
        if is_quantized:
            compute_power *= 0.7  # 30% power reduction

        return compute_power

    def _check_feasibility(
        self, memory_mb: float, memory_utilization: float, latency_ms: float
    ) -> tuple[bool, list[str], list[str]]:
        """
        Check if deployment is feasible and generate warnings/recommendations.

        Args:
            memory_mb: Estimated memory usage
            memory_utilization: Memory utilization percentage
            latency_ms: Estimated latency

        Returns:
            (is_feasible, warnings, recommendations)
        """
        warnings = []
        recommendations = []
        is_feasible = True

        # Check memory constraints
        if memory_utilization > 90:
            is_feasible = False
            warnings.append(f"Memory usage ({memory_utilization:.1f}%) exceeds safe limit")
            recommendations.append("Consider using a more aggressive quantization (INT8)")
            recommendations.append("Consider pruning to reduce model size")
        elif memory_utilization > 70:
            warnings.append(f"High memory usage ({memory_utilization:.1f}%)")
            recommendations.append("Monitor memory usage in production")

        # Check latency for real-time requirements
        if latency_ms > 1000:
            warnings.append(f"High latency ({latency_ms:.0f}ms) - not suitable for real-time")
            recommendations.append("Consider using INT8 quantization for speedup")
            if not self.device_specs.has_gpu and self.device_type != DeviceType.CORAL_DEV_BOARD:
                recommendations.append("Consider a device with GPU acceleration")
        elif latency_ms > 500:
            warnings.append(f"Moderate latency ({latency_ms:.0f}ms)")
            recommendations.append("May not be suitable for real-time video processing")

        # Device-specific recommendations
        if self.device_type == DeviceType.RASPBERRY_PI_3B:
            if memory_mb > 512:
                recommendations.append("Consider Raspberry Pi 4 for better performance")

        if not recommendations:
            recommendations.append("Model is well-suited for this device")

        return is_feasible, warnings, recommendations
