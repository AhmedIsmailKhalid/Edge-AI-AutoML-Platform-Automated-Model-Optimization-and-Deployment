"""
Performance estimation API endpoints.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.core.performance_estimator import (DEVICE_SPECS, DeviceType,
                                            PerformanceEstimator)
from src.database import get_db
from src.models.experiment import Experiment
from src.models.optimization_run import OptimizationRun

router = APIRouter()


@router.get("/devices")
async def list_devices():
    """
    List all supported edge devices with specifications.

    Returns:
        List of supported devices and their specs
    """
    devices = []
    for device_type, specs in DEVICE_SPECS.items():
        devices.append(
            {
                "device_type": device_type.value,
                "name": specs.name,
                "cpu_cores": specs.cpu_cores,
                "cpu_freq_ghz": specs.cpu_freq_ghz,
                "ram_gb": specs.ram_gb,
                "has_gpu": specs.has_gpu,
                "gpu_tflops": specs.gpu_tflops,
                "has_npu": specs.has_npu,
                "npu_tops": specs.npu_tops,
                "power_watts": specs.power_watts,
            }
        )

    return {"devices": devices, "total": len(devices)}


@router.get("/experiments/{experiment_id}/performance/{technique_name}")
async def get_performance_estimate(
    experiment_id: UUID, technique_name: str, device_type: str = None, db: Session = Depends(get_db)
):
    """
    Get performance estimate for a specific optimization technique.

    Args:
        experiment_id: Experiment UUID
        technique_name: Optimization technique name
        device_type: Optional device type override
        db: Database session

    Returns:
        Performance estimate with latency, memory, and power predictions

    Raises:
        HTTPException: If experiment or technique not found
    """
    # Get experiment
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Experiment {experiment_id} not found"
        )

    # Get optimization run
    opt_run = (
        db.query(OptimizationRun)
        .filter(
            OptimizationRun.experiment_id == experiment_id,
            OptimizationRun.technique_name == technique_name,
        )
        .first()
    )

    if not opt_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Technique {technique_name} not found"
        )

    # Determine device type
    target_device = device_type or experiment.target_device
    if not target_device:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No target device specified"
        )

    # Map device string to enum
    device_mapping = {
        "raspberry_pi_3b": DeviceType.RASPBERRY_PI_3B,
        "raspberry_pi_4": DeviceType.RASPBERRY_PI_4,
        "raspberry_pi_5": DeviceType.RASPBERRY_PI_5,
        "jetson_nano": DeviceType.JETSON_NANO,
        "jetson_xavier_nx": DeviceType.JETSON_XAVIER_NX,
        "coral_dev_board": DeviceType.CORAL_DEV_BOARD,
    }

    device_enum = device_mapping.get(target_device.lower().replace(" ", "_"))
    if not device_enum:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported device: {target_device}"
        )

    # Create estimator
    estimator = PerformanceEstimator(device_enum)

    # Detect quantization
    is_quantized = "quantization" in technique_name or "ptq" in technique_name
    quantization_bits = 8 if "int8" in technique_name else (4 if "int4" in technique_name else 32)

    # Estimate performance
    performance = estimator.estimate_performance(
        model_size_mb=opt_run.optimized_size_mb or opt_run.original_size_mb,
        params_count=opt_run.optimized_params_count or opt_run.original_params_count,
        framework=experiment.framework,
        is_quantized=is_quantized,
        quantization_bits=quantization_bits,
    )

    return {
        "experiment_id": str(experiment_id),
        "technique_name": technique_name,
        "device": {"type": performance.device_type.value, "name": performance.device_name},
        "latency": {
            "estimated_ms": performance.estimated_latency_ms,
            "min_ms": performance.estimated_latency_min_ms,
            "max_ms": performance.estimated_latency_max_ms,
            "fps": performance.estimated_fps,
        },
        "memory": {
            "estimated_mb": performance.estimated_memory_mb,
            "utilization_percent": performance.memory_utilization_percent,
        },
        "power": {
            "estimated_watts": performance.estimated_power_watts,
            "energy_per_inference_mj": performance.estimated_energy_per_inference_mj,
        },
        "feasibility": {
            "is_feasible": performance.is_feasible,
            "warnings": performance.warnings,
            "recommendations": performance.recommendations,
        },
    }


@router.get("/experiments/{experiment_id}/compare-devices/{technique_name}")
async def compare_devices(experiment_id: UUID, technique_name: str, db: Session = Depends(get_db)):
    """
    Compare performance estimates across all supported devices.

    Args:
        experiment_id: Experiment UUID
        technique_name: Optimization technique name
        db: Database session

    Returns:
        Performance comparison across all devices

    Raises:
        HTTPException: If experiment or technique not found
    """
    # Get experiment
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Experiment {experiment_id} not found"
        )

    # Get optimization run
    opt_run = (
        db.query(OptimizationRun)
        .filter(
            OptimizationRun.experiment_id == experiment_id,
            OptimizationRun.technique_name == technique_name,
        )
        .first()
    )

    if not opt_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Technique {technique_name} not found"
        )

    # Detect quantization
    is_quantized = "quantization" in technique_name or "ptq" in technique_name
    quantization_bits = 8 if "int8" in technique_name else (4 if "int4" in technique_name else 32)

    # Estimate for all devices
    comparisons = []
    for device_type in DeviceType:
        estimator = PerformanceEstimator(device_type)

        performance = estimator.estimate_performance(
            model_size_mb=opt_run.optimized_size_mb or opt_run.original_size_mb,
            params_count=opt_run.optimized_params_count or opt_run.original_params_count,
            framework=experiment.framework,
            is_quantized=is_quantized,
            quantization_bits=quantization_bits,
        )

        comparisons.append(
            {
                "device_type": device_type.value,
                "device_name": performance.device_name,
                "latency_ms": performance.estimated_latency_ms,
                "memory_mb": performance.estimated_memory_mb,
                "power_watts": performance.estimated_power_watts,
                "fps": performance.estimated_fps,
                "is_feasible": performance.is_feasible,
            }
        )

    # Sort by latency (fastest first)
    comparisons.sort(key=lambda x: x["latency_ms"])

    return {
        "experiment_id": str(experiment_id),
        "technique_name": technique_name,
        "comparisons": comparisons,
        "best_device": comparisons[0]["device_name"],
        "total_devices": len(comparisons),
    }
