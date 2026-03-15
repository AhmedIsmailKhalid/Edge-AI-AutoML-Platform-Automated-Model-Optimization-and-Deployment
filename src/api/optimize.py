"""
Optimization start/stop API endpoints with robust session management.
"""

import asyncio
import json
import logging
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

# from src.core.orchestrator import ExperimentOrchestrator
from src.database import SessionLocal, get_db
from src.models.experiment import Experiment, ExperimentStatus
from src.models.optimization_run import OptimizationRun

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/{experiment_id}/start")
async def start_optimization(
    experiment_id: UUID, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
) -> dict:
    """
    Start optimization for an experiment.

    Args:
        experiment_id: Experiment UUID
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        Confirmation message

    Raises:
        HTTPException: If experiment not found or already running
    """
    # Get experiment
    from src.core.orchestrator import ExperimentOrchestrator

    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Experiment {experiment_id} not found"
        )

    # Check if already running
    if experiment.status == ExperimentStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Experiment is already running"
        )

    # Background task with robust session management
    async def run_optimization():
        """Background task with its own database session and proper cleanup."""
        task_db = None
        try:
            logger.info(f"🔵 Starting background optimization for experiment {experiment_id}")

            # Create new session for background task
            task_db = SessionLocal()
            logger.info("🔵 Created new database session")

            # Create and run orchestrator
            orchestrator = ExperimentOrchestrator(experiment_id, task_db)
            await orchestrator.run()

            logger.info("🟢 Optimization completed successfully!")

        except Exception as e:
            logger.error(f"🔴 Optimization failed: {type(e).__name__}: {e}", exc_info=True)

            # Update experiment status to failed
            if task_db:
                try:
                    exp = task_db.query(Experiment).filter(Experiment.id == experiment_id).first()
                    if exp:
                        exp.status = ExperimentStatus.FAILED
                        exp.error_message = str(e)
                        task_db.commit()
                        task_db.flush()
                        logger.info("🔴 Updated experiment status to FAILED")
                except Exception as update_error:
                    logger.error(f"🔴 Failed to update experiment status: {update_error}")
                    if task_db:
                        task_db.rollback()
        finally:
            # CRITICAL: Always close the session
            if task_db:
                try:
                    task_db.close()
                    logger.info("🔵 Database session closed")
                except Exception as close_error:
                    logger.error(f"🔴 Error closing database session: {close_error}")

    # Add to background tasks
    background_tasks.add_task(run_optimization)

    return {
        "experiment_id": str(experiment_id),
        "status": "started",
        "message": "Optimization started in background",
    }


@router.post("/{experiment_id}/stop")
async def stop_optimization(experiment_id: UUID, db: Session = Depends(get_db)) -> dict:
    """
    Stop a running optimization.

    Args:
        experiment_id: Experiment UUID
        db: Database session

    Returns:
        Confirmation message

    Raises:
        HTTPException: If experiment not found or not running
    """
    # Get experiment
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Experiment {experiment_id} not found"
        )

    # Check if running
    if experiment.status != ExperimentStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Experiment is not running"
        )

    # Update status to stopped
    experiment.status = ExperimentStatus.STOPPED
    db.commit()

    return {
        "experiment_id": str(experiment_id),
        "status": "stopped",
        "message": "Optimization stopped",
    }


@router.get("/{experiment_id}/progress-stream")
async def stream_progress(
    experiment_id: UUID,
    db: Session = Depends(get_db),
):
    """Stream real-time progress updates using Server-Sent Events."""

    async def event_generator():
        """Generate SSE events with progress updates."""
        last_progress = -1
        techniques_seen = set()

        while True:
            # Get current status
            experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

            if not experiment:
                yield f"data: {json.dumps({'error': 'Experiment not found'})}\n\n"
                break

            # Get optimization runs
            runs = (
                db.query(OptimizationRun)
                .filter(OptimizationRun.experiment_id == experiment_id)
                .all()
            )

            # Send updates for new/changed techniques
            for run in runs:
                tech_key = f"{run.technique_name}_{run.status}"

                if tech_key not in techniques_seen:
                    techniques_seen.add(tech_key)

                    event_data = {
                        "type": "technique_update",
                        "technique": run.technique_name,
                        "status": (
                            run.status.value if hasattr(run.status, "value") else str(run.status)
                        ),
                        "accuracy": (
                            float(run.optimized_accuracy) if run.optimized_accuracy else None
                        ),
                        "size_reduction": (
                            float(run.size_reduction_percent)
                            if run.size_reduction_percent
                            else None
                        ),
                    }

                    yield f"data: {json.dumps(event_data)}\n\n"

            # Send progress update if changed
            progress = experiment.progress_percent or 0
            if progress != last_progress:
                last_progress = progress

                event_data = {
                    "type": "progress",
                    "progress": progress,
                    "status": (
                        experiment.status.value
                        if hasattr(experiment.status, "value")
                        else str(experiment.status)
                    ),
                }

                yield f"data: {json.dumps(event_data)}\n\n"

            # Check if completed
            if experiment.status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]:
                event_data = {
                    "type": "complete",
                    "status": (
                        experiment.status.value
                        if hasattr(experiment.status, "value")
                        else str(experiment.status)
                    ),
                }
                yield f"data: {json.dumps(event_data)}\n\n"
                break

            await asyncio.sleep(0.3)  # Poll every 300ms

            # Refresh DB session
            db.expire_all()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
