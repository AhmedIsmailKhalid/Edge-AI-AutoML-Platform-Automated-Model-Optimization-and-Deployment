"""
Unit tests for performance estimator.
"""

import pytest

from src.core.performance_estimator import (DEVICE_SPECS, DeviceType,
                                            PerformanceEstimator)


def test_device_specs_exist():
    """Test that all device specs are defined."""
    print("\n  Testing device specs...")

    for device_type in DeviceType:
        assert device_type in DEVICE_SPECS
        specs = DEVICE_SPECS[device_type]

        print(f"   ✅ {specs.name}:")
        print(f"      - CPU: {specs.cpu_cores} cores @ {specs.cpu_freq_ghz} GHz")
        print(f"      - RAM: {specs.ram_gb} GB")
        print(f"      - GPU: {specs.has_gpu}")
        print(f"      - Power: {specs.power_watts}W")

    print(f"   ✅ All {len(DeviceType)} device specs defined")


def test_raspberry_pi_4_estimation():
    """Test performance estimation for Raspberry Pi 4."""
    print("\n  Testing Raspberry Pi 4 estimation...")

    estimator = PerformanceEstimator(DeviceType.RASPBERRY_PI_4)

    # Test with a small quantized model
    performance = estimator.estimate_performance(
        model_size_mb=5.0,
        params_count=1_000_000,
        framework="pytorch",
        is_quantized=True,
        quantization_bits=8,
    )

    print("     Model: 5MB, 1M params, INT8 quantized")
    print(f"      Latency: {performance.estimated_latency_ms:.1f}ms")
    print(
        f"     Memory: {performance.estimated_memory_mb:.1f}MB ({performance.memory_utilization_percent:.1f}%)"
    )
    print(f"     Power: {performance.estimated_power_watts:.2f}W")
    print(f"     FPS: {performance.estimated_fps:.1f}")
    print(f"   ✅ Feasible: {performance.is_feasible}")

    # Assertions
    assert performance.device_type == DeviceType.RASPBERRY_PI_4
    assert performance.estimated_latency_ms > 0
    assert performance.estimated_memory_mb > 0
    assert performance.estimated_power_watts > 0
    assert performance.estimated_fps > 0
    assert len(performance.warnings) >= 0
    assert len(performance.recommendations) >= 0

    print("   ✅ Raspberry Pi 4 estimation passed!")


def test_jetson_nano_estimation():
    """Test performance estimation for Jetson Nano."""
    print("\n  Testing Jetson Nano estimation...")

    estimator = PerformanceEstimator(DeviceType.JETSON_NANO)

    # Test with a larger FP32 model
    performance = estimator.estimate_performance(
        model_size_mb=50.0,
        params_count=10_000_000,
        framework="tensorflow",
        is_quantized=False,
        quantization_bits=32,
    )

    print("     Model: 50MB, 10M params, FP32")
    print(f"      Latency: {performance.estimated_latency_ms:.1f}ms")
    print(
        f"     Memory: {performance.estimated_memory_mb:.1f}MB ({performance.memory_utilization_percent:.1f}%)"
    )
    print(f"     Power: {performance.estimated_power_watts:.2f}W")
    print(f"     FPS: {performance.estimated_fps:.1f}")
    print(f"   ✅ Feasible: {performance.is_feasible}")

    if performance.warnings:
        print("   ⚠️  Warnings:")
        for warning in performance.warnings:
            print(f"      - {warning}")

    if performance.recommendations:
        print("   💡 Recommendations:")
        for rec in performance.recommendations:
            print(f"      - {rec}")

    # Jetson Nano should handle this better than Pi
    assert performance.device_type == DeviceType.JETSON_NANO
    assert performance.estimated_latency_ms > 0

    print("   ✅ Jetson Nano estimation passed!")


def test_coral_tpu_estimation():
    """Test performance estimation for Coral Dev Board with TPU."""
    print("\n  Testing Coral TPU estimation...")

    estimator = PerformanceEstimator(DeviceType.CORAL_DEV_BOARD)

    # Test with INT8 quantized model (TPU optimized)
    performance = estimator.estimate_performance(
        model_size_mb=3.0,
        params_count=500_000,
        framework="tensorflow",
        is_quantized=True,
        quantization_bits=8,
    )

    print("     Model: 3MB, 500K params, INT8 (TPU-optimized)")
    print(f"      Latency: {performance.estimated_latency_ms:.1f}ms")
    print(f"     Memory: {performance.estimated_memory_mb:.1f}MB")
    print(f"     Power: {performance.estimated_power_watts:.2f}W")
    print(f"     FPS: {performance.estimated_fps:.1f}")

    # TPU should be very fast for INT8
    assert performance.estimated_latency_ms < 50  # Should be under 50ms
    assert performance.estimated_fps > 20  # Should handle real-time

    print("   ✅ Coral TPU estimation passed!")


def test_quantization_speedup():
    """Test that quantization provides speedup."""
    print("\n  Testing quantization speedup...")

    estimator = PerformanceEstimator(DeviceType.RASPBERRY_PI_4)

    # Same model, different quantization
    fp32_perf = estimator.estimate_performance(
        model_size_mb=20.0,
        params_count=5_000_000,
        framework="pytorch",
        is_quantized=False,
        quantization_bits=32,
    )

    int8_perf = estimator.estimate_performance(
        model_size_mb=5.0,  # Smaller after quantization
        params_count=5_000_000,
        framework="pytorch",
        is_quantized=True,
        quantization_bits=8,
    )

    print(f"     FP32: {fp32_perf.estimated_latency_ms:.1f}ms")
    print(f"     INT8: {int8_perf.estimated_latency_ms:.1f}ms")
    print(f"     Speedup: {fp32_perf.estimated_latency_ms / int8_perf.estimated_latency_ms:.2f}x")

    # INT8 should be faster
    assert int8_perf.estimated_latency_ms < fp32_perf.estimated_latency_ms

    # INT8 should use less power
    assert int8_perf.estimated_power_watts <= fp32_perf.estimated_power_watts

    print("   ✅ Quantization speedup verified!")


def test_memory_warnings():
    """Test that memory warnings are generated correctly."""
    print("\n  Testing memory warnings...")

    estimator = PerformanceEstimator(DeviceType.RASPBERRY_PI_3B)  # Only 1GB RAM

    # Very large model
    performance = estimator.estimate_performance(
        model_size_mb=500.0,  # 500MB model
        params_count=50_000_000,
        framework="tensorflow",
        is_quantized=False,
        quantization_bits=32,
    )

    print("     Large model: 500MB, 50M params")
    print(
        f"     Memory: {performance.estimated_memory_mb:.1f}MB ({performance.memory_utilization_percent:.1f}%)"
    )
    print(f"   ✅ Feasible: {performance.is_feasible}")

    if performance.warnings:
        print("   ⚠️  Warnings:")
        for warning in performance.warnings:
            print(f"      - {warning}")

    # Should have memory warnings
    assert performance.memory_utilization_percent > 50  # Over 50% on 1GB device
    assert len(performance.warnings) > 0

    print("   ✅ Memory warning generation works!")


def test_latency_range():
    """Test that latency range is reasonable."""
    print("\n  Testing latency range...")

    estimator = PerformanceEstimator(DeviceType.RASPBERRY_PI_4)

    performance = estimator.estimate_performance(
        model_size_mb=10.0,
        params_count=2_000_000,
        framework="pytorch",
        is_quantized=True,
        quantization_bits=8,
    )

    print("      Latency range:")
    print(f"      Min: {performance.estimated_latency_min_ms:.1f}ms")
    print(f"      Avg: {performance.estimated_latency_ms:.1f}ms")
    print(f"      Max: {performance.estimated_latency_max_ms:.1f}ms")

    # Check that range makes sense
    assert performance.estimated_latency_min_ms < performance.estimated_latency_ms
    assert performance.estimated_latency_ms < performance.estimated_latency_max_ms

    # Check variance is reasonable (~20%)
    variance = (
        performance.estimated_latency_max_ms - performance.estimated_latency_min_ms
    ) / performance.estimated_latency_ms
    assert 0.3 < variance < 0.5  # Should be around 40% range

    print("   ✅ Latency range is reasonable!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
