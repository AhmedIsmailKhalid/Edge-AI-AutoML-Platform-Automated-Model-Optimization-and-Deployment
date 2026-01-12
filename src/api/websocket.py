"""
WebSocket endpoints for real-time progress updates.
"""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from src.database import get_db
from src.models.experiment import Experiment


router = APIRouter()

# Store active WebSocket connections per experiment
active_connections: dict[str, set[WebSocket]] = {}


class ConnectionManager:
    """Manages WebSocket connections for experiments."""

    def __init__(self):
        self.active_connections: dict[str, set[WebSocket]] = {}

    async def connect(self, experiment_id: str, websocket: WebSocket):
        """
        Accept and register a new WebSocket connection.

        Args:
            experiment_id: Experiment UUID as string
            websocket: WebSocket connection
        """
        await websocket.accept()

        if experiment_id not in self.active_connections:
            self.active_connections[experiment_id] = set()

        self.active_connections[experiment_id].add(websocket)

    def disconnect(self, experiment_id: str, websocket: WebSocket):
        """
        Remove a WebSocket connection.

        Args:
            experiment_id: Experiment UUID as string
            websocket: WebSocket connection
        """
        if experiment_id in self.active_connections:
            self.active_connections[experiment_id].discard(websocket)

            # Clean up empty sets
            if not self.active_connections[experiment_id]:
                del self.active_connections[experiment_id]

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """
        Send a message to a specific WebSocket.

        Args:
            message: Message dictionary
            websocket: Target WebSocket
        """
        await websocket.send_json(message)

    async def broadcast_to_experiment(self, experiment_id: str, message: dict):
        """
        Broadcast a message to all connections for an experiment.

        Args:
            experiment_id: Experiment UUID as string
            message: Message dictionary
        """
        if experiment_id not in self.active_connections:
            return

        # Send to all connected clients for this experiment
        disconnected = []
        for connection in self.active_connections[experiment_id]:
            try:
                await connection.send_json(message)
            except Exception:
                # Mark for removal if connection is dead
                disconnected.append(connection)

        # Clean up dead connections
        for connection in disconnected:
            self.active_connections[experiment_id].discard(connection)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/experiments/{experiment_id}")
async def websocket_endpoint(
    websocket: WebSocket, experiment_id: UUID, db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for real-time experiment progress updates.

    Clients connect to this endpoint to receive live updates about
    optimization progress, including status changes, progress percentage,
    and currently running techniques.

    Args:
        websocket: WebSocket connection
        experiment_id: Experiment UUID
        db: Database session

    Message Format:
        {
            "type": "progress" | "status" | "error" | "complete",
            "experiment_id": "uuid",
            "status": "running" | "completed" | "failed",
            "progress_percent": 0-100,
            "current_technique": "technique_name",
            "message": "Human readable message",
            "timestamp": "ISO timestamp"
        }
    """
    experiment_id_str = str(experiment_id)

    # Verify experiment exists
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if not experiment:
        await websocket.close(code=1008, reason="Experiment not found")
        return

    # Connect client
    await manager.connect(experiment_id_str, websocket)

    try:
        # Send initial status
        await manager.send_personal_message(
            {
                "type": "connected",
                "experiment_id": experiment_id_str,
                "status": experiment.status.value,
                "progress_percent": experiment.progress_percent,
                "message": "Connected to experiment updates",
            },
            websocket,
        )

        # Keep connection alive and listen for messages
        while True:
            # Wait for any messages from client (ping/pong, etc)
            data = await websocket.receive_text()

            # Handle client messages if needed
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        # Client disconnected normally
        manager.disconnect(experiment_id_str, websocket)
    except Exception as e:
        # Handle other errors
        print(f"WebSocket error for experiment {experiment_id_str}: {e}")
        manager.disconnect(experiment_id_str, websocket)


async def send_progress_update(
    experiment_id: str,
    status: str,
    progress_percent: int,
    current_technique: str = None,
    message: str = None,
):
    """
    Send a progress update to all connected clients for an experiment.

    This function is called by the orchestrator during optimization.

    Args:
        experiment_id: Experiment UUID as string
        status: Current status (running, completed, failed)
        progress_percent: Progress percentage (0-100)
        current_technique: Currently executing technique name
        message: Optional human-readable message
    """

    update = {
        "type": "progress",
        "experiment_id": experiment_id,
        "status": status,
        "progress_percent": progress_percent,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if current_technique:
        update["current_technique"] = current_technique

    if message:
        update["message"] = message

    await manager.broadcast_to_experiment(experiment_id, update)


async def send_status_update(experiment_id: str, status: str, message: str = None):
    """
    Send a status change update.

    Args:
        experiment_id: Experiment UUID as string
        status: New status
        message: Optional message
    """

    update = {
        "type": "status",
        "experiment_id": experiment_id,
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if message:
        update["message"] = message

    await manager.broadcast_to_experiment(experiment_id, update)


async def send_error(experiment_id: str, error_message: str):
    """
    Send an error notification.

    Args:
        experiment_id: Experiment UUID as string
        error_message: Error description
    """

    update = {
        "type": "error",
        "experiment_id": experiment_id,
        "error_message": error_message,
        "timestamp": datetime.utcnow().isoformat(),
    }

    await manager.broadcast_to_experiment(experiment_id, update)


async def send_completion(experiment_id: str, completed_techniques: int, failed_techniques: int):
    """
    Send completion notification.

    Args:
        experiment_id: Experiment UUID as string
        completed_techniques: Number of successfully completed techniques
        failed_techniques: Number of failed techniques
    """

    update = {
        "type": "complete",
        "experiment_id": experiment_id,
        "status": "completed",
        "completed_techniques": completed_techniques,
        "failed_techniques": failed_techniques,
        "timestamp": datetime.utcnow().isoformat(),
    }

    await manager.broadcast_to_experiment(experiment_id, update)
