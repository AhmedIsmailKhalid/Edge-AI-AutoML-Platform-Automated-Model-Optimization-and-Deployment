"""
WebSocket manager for real-time progress updates.
"""

import json
import logging
from uuid import UUID

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""

    def __init__(self):
        # Map experiment_id to set of connected websockets
        self.active_connections: dict[str, set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, experiment_id: UUID):
        """Connect a client to an experiment's updates."""
        await websocket.accept()

        exp_id_str = str(experiment_id)
        if exp_id_str not in self.active_connections:
            self.active_connections[exp_id_str] = set()

        self.active_connections[exp_id_str].add(websocket)
        logger.info(f"WebSocket client connected to experiment {experiment_id}")

    def disconnect(self, websocket: WebSocket, experiment_id: UUID):
        """Disconnect a client."""
        exp_id_str = str(experiment_id)
        if exp_id_str in self.active_connections:
            self.active_connections[exp_id_str].discard(websocket)

            # Clean up empty sets
            if not self.active_connections[exp_id_str]:
                del self.active_connections[exp_id_str]

        logger.info(f"WebSocket client disconnected from experiment {experiment_id}")

    async def broadcast(self, experiment_id: UUID, message: dict):
        """Broadcast message to all clients watching this experiment."""
        exp_id_str = str(experiment_id)

        if exp_id_str not in self.active_connections:
            return

        # Convert message to JSON
        message_json = json.dumps(message)

        # Send to all connected clients
        disconnected = set()
        for websocket in self.active_connections[exp_id_str]:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                disconnected.add(websocket)

        # Clean up disconnected clients
        for websocket in disconnected:
            self.disconnect(websocket, experiment_id)


# Global connection manager instance
manager = ConnectionManager()
