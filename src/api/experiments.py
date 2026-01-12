"""
Experiment CRUD API endpoints.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from src.database import get_db
from src.models.experiment import Experiment, ExperimentStatus
from src.schemas.experiment import (
    ExperimentCreate,
    ExperimentListResponse,
    ExperimentResponse,
    ExperimentUpdate,
)


router = APIRouter()


def _generate_unique_experiment_name(base_name: str, db: Session) -> str:
    """
    Generate a unique experiment name by appending a number if the name already exists.

    Args:
        base_name: Base experiment name
        db: Database session

    Returns:
        Unique experiment name

    Example:
        If 'My Experiment' exists, returns 'My Experiment 2'
    """
    existing = db.query(Experiment).filter(Experiment.name == base_name).first()

    if not existing:
        return base_name

    counter = 2
    while True:
        new_name = f"{base_name} {counter}"
        existing = db.query(Experiment).filter(Experiment.name == new_name).first()
        if not existing:
            return new_name
        counter += 1


@router.post("/create", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    experiment_data: ExperimentCreate, db: Session = Depends(get_db)
) -> ExperimentResponse:
    """
    Create a new experiment.

    If an experiment with the same name exists, automatically appends a number suffix.
    model_name is now optional and will be set when model is uploaded.

    Args:
        experiment_data: Experiment configuration data
        db: Database session

    Returns:
        Created experiment details
    """
    unique_name = _generate_unique_experiment_name(experiment_data.name, db)

    experiment = Experiment(
        name=unique_name,
        description=experiment_data.description,
        model_name=experiment_data.model_name or "pending_upload",
        framework=experiment_data.framework,
        dataset_type=experiment_data.dataset_type,
        dataset_name=experiment_data.dataset_name,
        target_device=experiment_data.target_device,
        optimization_goal=experiment_data.optimization_goal,
        min_accuracy_percent=experiment_data.min_accuracy_percent,
        max_size_mb=experiment_data.max_size_mb,
        max_latency_ms=experiment_data.max_latency_ms,
        max_accuracy_drop_percent=experiment_data.max_accuracy_drop_percent,
        status=ExperimentStatus.PENDING,
    )

    db.add(experiment)
    db.commit()
    db.refresh(experiment)

    return ExperimentResponse.model_validate(experiment)


@router.post("/create-with-pretrained")
async def create_experiment_with_pretrained(
    request: ExperimentCreate, db: Session = Depends(get_db)
) -> dict:
    """
    Create experiment and automatically attach pretrained model.
    For demo/testing purposes.
    """

    from src.config import settings
    from src.models.model_file import ModelFile

    experiment = Experiment(
        name=request.name,
        description=request.description,
        model_name=request.model_name,
        framework=request.framework,
        dataset_type=request.dataset_type,
        dataset_name=request.dataset_name,
        target_device=request.target_device,
        optimization_goal=request.optimization_goal,
        status=ExperimentStatus.PENDING,
        progress_percent=0,
    )

    db.add(experiment)
    db.flush()

    size = "medium"
    ext = ".pt" if request.framework == "pytorch" else ".h5"
    model_filename = f"{size}_{request.dataset_name}_cnn{ext}"
    model_path = settings.pretrained_models_path / request.framework / model_filename

    if not model_path.exists():
        db.rollback()
        raise HTTPException(status_code=404, detail=f"Pretrained model not found: {model_path}")

    file_format = "pytorch_pt" if request.framework == "pytorch" else "tensorflow_h5"
    model_file = ModelFile(
        experiment_id=experiment.id,
        file_type="original",
        file_format=file_format,
        file_path=str(model_path),
        file_size_mb=model_path.stat().st_size / (1024 * 1024),
    )

    db.add(model_file)
    db.commit()
    db.refresh(experiment)

    return {
        "id": str(experiment.id),
        "name": experiment.name,
        "framework": experiment.framework,
        "status": experiment.status.value,
        "message": "Experiment created with pretrained model",
    }


@router.get("", response_model=ExperimentListResponse)
async def list_experiments(
    skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
) -> ExperimentListResponse:
    """
    List all experiments with pagination.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session

    Returns:
        List of experiments
    """
    experiments = db.query(Experiment).offset(skip).limit(limit).all()
    total = db.query(Experiment).count()

    return ExperimentListResponse(
        experiments=[ExperimentResponse.model_validate(exp) for exp in experiments], total=total
    )


@router.get("/recent")
async def get_recent_experiments(
    limit: int = Query(default=20, ge=1, le=100), db: Session = Depends(get_db)
):
    """Get recent experiments ordered by creation date."""
    experiments = db.query(Experiment).order_by(Experiment.created_at.desc()).limit(limit).all()

    return [
        {
            "id": str(exp.id),
            "name": exp.name,
            "description": exp.description,
            "model_name": exp.model_name,
            "framework": exp.framework,
            "dataset_name": exp.dataset_name,
            "target_device": exp.target_device,
            "optimization_goal": exp.optimization_goal,
            "status": exp.status.value if hasattr(exp.status, "value") else str(exp.status),
            "progress_percent": exp.progress_percent,
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
            "started_at": exp.started_at.isoformat() if exp.started_at else None,
            "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
        }
        for exp in experiments
    ]


@router.get("/search")
async def search_experiments(
    query: str = "",
    framework: str = None,
    status: str = None,
    target_device: str = None,
    optimization_goal: str = None,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """Search and filter experiments."""
    experiments_query = db.query(Experiment)

    if query:
        experiments_query = experiments_query.filter(
            (Experiment.name.ilike(f"%{query}%"))
            | (Experiment.description.ilike(f"%{query}%"))
            | (Experiment.model_name.ilike(f"%{query}%"))
        )

    if framework:
        experiments_query = experiments_query.filter(Experiment.framework == framework)

    if status:
        experiments_query = experiments_query.filter(Experiment.status == status)

    if target_device:
        experiments_query = experiments_query.filter(Experiment.target_device == target_device)

    if optimization_goal:
        experiments_query = experiments_query.filter(
            Experiment.optimization_goal == optimization_goal
        )

    experiments = experiments_query.order_by(Experiment.created_at.desc()).limit(limit).all()

    return [
        {
            "id": str(exp.id),
            "name": exp.name,
            "description": exp.description,
            "model_name": exp.model_name,
            "framework": exp.framework,
            "dataset_name": exp.dataset_name,
            "target_device": exp.target_device,
            "optimization_goal": exp.optimization_goal,
            "status": exp.status.value if hasattr(exp.status, "value") else str(exp.status),
            "progress_percent": exp.progress_percent,
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
            "updated_at": exp.updated_at.isoformat() if exp.updated_at else None,
            "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
        }
        for exp in experiments
    ]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: UUID, db: Session = Depends(get_db)) -> ExperimentResponse:
    """
    Get experiment details by ID.

    Args:
        experiment_id: Experiment UUID
        db: Database session

    Returns:
        Experiment details

    Raises:
        HTTPException: If experiment not found
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id {experiment_id} not found",
        )

    return ExperimentResponse.model_validate(experiment)


@router.patch("/{experiment_id}", response_model=ExperimentResponse)
async def update_experiment(
    experiment_id: UUID, update_data: ExperimentUpdate, db: Session = Depends(get_db)
) -> ExperimentResponse:
    """
    Update experiment details.

    Args:
        experiment_id: Experiment UUID
        update_data: Fields to update
        db: Database session

    Returns:
        Updated experiment details

    Raises:
        HTTPException: If experiment not found
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id {experiment_id} not found",
        )

    update_dict = update_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(experiment, field, value)

    db.commit()
    db.refresh(experiment)

    return ExperimentResponse.model_validate(experiment)


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_experiment(experiment_id: UUID, db: Session = Depends(get_db)) -> None:
    """
    Delete an experiment and all associated data.

    Args:
        experiment_id: Experiment UUID
        db: Database session

    Raises:
        HTTPException: If experiment not found
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id {experiment_id} not found",
        )

    db.delete(experiment)
    db.commit()
