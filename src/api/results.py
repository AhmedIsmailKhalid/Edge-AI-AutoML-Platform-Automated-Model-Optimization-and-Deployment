"""
Results and recommendations API endpoints.
"""

import json
import zipfile
from io import BytesIO
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from src.database import get_db
from src.models.experiment import Experiment
from src.models.model_file import ModelFile
from src.models.optimization_run import OptimizationRun, OptimizationStatus

router = APIRouter()


@router.get("/{experiment_id}/results")
async def get_optimization_results(experiment_id: UUID, db: Session = Depends(get_db)):
    """Get all optimization results for an experiment."""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    runs = (
        db.query(OptimizationRun)
        .filter(OptimizationRun.experiment_id == experiment_id)
        .order_by(OptimizationRun.execution_order)
        .all()
    )

    return [
        {
            "id": str(run.id),
            "technique_name": run.technique_name,
            "status": run.status.value if hasattr(run.status, "value") else str(run.status),
            "original_accuracy": float(run.original_accuracy) if run.original_accuracy else None,
            "optimized_accuracy": float(run.optimized_accuracy) if run.optimized_accuracy else None,
            "accuracy_drop_percent": float(run.accuracy_drop_percent)
            if run.accuracy_drop_percent
            else None,
            "original_size_mb": float(run.original_size_mb) if run.original_size_mb else None,
            "optimized_size_mb": float(run.optimized_size_mb) if run.optimized_size_mb else None,
            "size_reduction_percent": float(run.size_reduction_percent)
            if run.size_reduction_percent
            else None,
            "inference_latency_ms": float(getattr(run, "inference_latency_ms", None))
            if getattr(run, "inference_latency_ms", None)
            else None,
            "memory_usage_mb": float(getattr(run, "memory_usage_mb", None))
            if getattr(run, "memory_usage_mb", None)
            else None,
            "execution_order": run.execution_order,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        }
        for run in runs
    ]


@router.get("/{experiment_id}/recommendations")
async def get_recommendations(experiment_id: UUID, db: Session = Depends(get_db)):
    """Get simple ranked recommendations based on results."""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Experiment {experiment_id} not found"
        )

    runs = (
        db.query(OptimizationRun)
        .filter(
            OptimizationRun.experiment_id == experiment_id,
            OptimizationRun.status == OptimizationStatus.COMPLETED,
        )
        .all()
    )

    if not runs:
        return []

    goal = experiment.optimization_goal
    scored_runs = []

    for run in runs:
        score = 0
        reasoning = ""

        meets_constraints = True
        constraint_issues = []

        if experiment.max_accuracy_drop_percent is not None:
            if (
                run.accuracy_drop_percent
                and run.accuracy_drop_percent > experiment.max_accuracy_drop_percent
            ):
                meets_constraints = False
                constraint_issues.append(
                    f"accuracy drop {run.accuracy_drop_percent:.1f}% exceeds limit {experiment.max_accuracy_drop_percent:.1f}%"
                )

        if experiment.max_size_mb is not None:
            if run.optimized_size_mb and run.optimized_size_mb > experiment.max_size_mb:
                meets_constraints = False
                constraint_issues.append(
                    f"size {run.optimized_size_mb:.1f}MB exceeds limit {experiment.max_size_mb:.1f}MB"
                )

        inference_latency = getattr(run, "inference_latency_ms", None)
        if experiment.max_latency_ms is not None and inference_latency is not None:
            if inference_latency > experiment.max_latency_ms:
                meets_constraints = False
                constraint_issues.append(
                    f"latency {inference_latency:.1f}ms exceeds limit {experiment.max_latency_ms:.1f}ms"
                )

        if not meets_constraints:
            continue

        if goal == "balanced":
            if run.optimized_accuracy:
                score += run.optimized_accuracy * 40
            if run.size_reduction_percent:
                score += run.size_reduction_percent / 2
            if inference_latency:
                score += (100 - min(inference_latency, 100)) / 10
            reasoning = f"Best balance: {run.optimized_accuracy*100:.1f}% accuracy, {run.size_reduction_percent:.1f}% smaller"

        elif goal == "maximize_accuracy":
            if run.optimized_accuracy:
                score += run.optimized_accuracy * 100
            reasoning = f"Highest accuracy: {run.optimized_accuracy*100:.1f}%"

        elif goal == "minimize_size":
            if run.size_reduction_percent:
                score += run.size_reduction_percent
            reasoning = f"Maximum size reduction: {run.size_reduction_percent:.1f}% smaller"

        elif goal == "minimize_latency":
            if inference_latency:
                score += (1000 - min(inference_latency, 1000)) / 10
            reasoning = f"Fastest inference: {inference_latency:.1f}ms"

        scored_runs.append(
            {
                "result_id": str(run.id),
                "technique_name": run.technique_name,
                "score": score,
                "confidence_score": min(score / 100, 1.0),
                "reasoning": reasoning,
                "accuracy": run.optimized_accuracy,
                "size_reduction": run.size_reduction_percent,
                "latency_ms": inference_latency,
            }
        )

    scored_runs.sort(key=lambda x: x["score"], reverse=True)

    return scored_runs


@router.get("/{experiment_id}/download/{result_id}")
async def download_optimized_model(
    experiment_id: UUID, result_id: UUID, db: Session = Depends(get_db)
):
    """Download the optimized model as a ZIP package with metadata and deployment script."""
    run = (
        db.query(OptimizationRun)
        .filter(
            OptimizationRun.experiment_id == experiment_id,
            OptimizationRun.id == result_id,
        )
        .first()
    )

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Optimization run not found",
        )

    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

    model_file = (
        db.query(ModelFile)
        .filter(ModelFile.optimization_run_id == run.id, ModelFile.file_type == "optimized")
        .first()
    )

    if not model_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Optimized model not found",
        )

    model_path = Path(model_file.file_path)

    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Model file not found on disk"
        )

    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(model_path, model_path.name)

        metadata = {
            "experiment_id": str(experiment_id),
            "experiment_name": experiment.name,
            "technique": run.technique_name,
            "framework": experiment.framework,
            "dataset": experiment.dataset_name,
            "target_device": experiment.target_device,
            "optimization_goal": experiment.optimization_goal,
            "original_accuracy": run.original_accuracy,
            "optimized_accuracy": run.optimized_accuracy,
            "accuracy_drop_percent": run.accuracy_drop_percent,
            "original_size_mb": run.original_size_mb,
            "optimized_size_mb": run.optimized_size_mb,
            "size_reduction_percent": run.size_reduction_percent,
            "model_filename": model_path.name,
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }

        zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))

        if experiment.framework == "pytorch":
            deploy_script = f"""#!/usr/bin/env python3
import torch

model_path = "{model_path.name}"
model = torch.load(model_path)
model.eval()

print("Model loaded successfully!")
print(f"Technique: {run.technique_name}")
print(f"Accuracy: {run.optimized_accuracy*100:.2f}%")
print(f"Size: {run.optimized_size_mb:.2f} MB")
"""
        else:
            deploy_script = f"""#!/usr/bin/env python3
import tensorflow as tf

model_path = "{model_path.name}"
model = tf.keras.models.load_model(model_path)

print("Model loaded successfully!")
print(f"Technique: {run.technique_name}")
print(f"Accuracy: {run.optimized_accuracy*100:.2f}%")
print(f"Size: {run.optimized_size_mb:.2f} MB")
"""

        zip_file.writestr("deploy.py", deploy_script)

        readme = f"""# Optimized Model Package

## Experiment Details
- **Experiment**: {experiment.name}
- **Technique**: {run.technique_name}
- **Framework**: {experiment.framework}
- **Dataset**: {experiment.dataset_name}
- **Target Device**: {experiment.target_device}

## Performance Metrics
- **Original Accuracy**: {run.original_accuracy*100:.2f}%
- **Optimized Accuracy**: {run.optimized_accuracy*100:.2f}%
- **Accuracy Drop**: {run.accuracy_drop_percent:.2f}%
- **Original Size**: {run.original_size_mb:.2f} MB
- **Optimized Size**: {run.optimized_size_mb:.2f} MB
- **Size Reduction**: {run.size_reduction_percent:.2f}%

## Files Included
- `{model_path.name}` - Optimized model file
- `metadata.json` - Complete experiment metadata
- `deploy.py` - Sample deployment script
- `README.md` - This file

## Quick Start
```bash
python deploy.py
```

## Loading the Model

### PyTorch
```python
import torch
model = torch.load("{model_path.name}")
model.eval()
```

### TensorFlow
```python
import tensorflow as tf
model = tf.keras.models.load_model("{model_path.name}")
```
"""

        zip_file.writestr("README.md", readme)

    zip_buffer.seek(0)

    filename = f"{run.technique_name}_optimized.zip"

    return Response(
        content=zip_buffer.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
