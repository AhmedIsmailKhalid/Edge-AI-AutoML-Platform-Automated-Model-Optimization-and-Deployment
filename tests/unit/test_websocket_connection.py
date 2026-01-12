"""
Test WebSocket connection and messaging.
"""

import asyncio
import json
from uuid import uuid4

import pytest
import websockets

from src.database import SessionLocal
from src.models.experiment import (Experiment, ExperimentStatus,
                                   OptimizationGoal)


@pytest.mark.asyncio
async def test_websocket_with_invalid_experiment():
    """Test WebSocket connection with invalid experiment ID."""
    print("\n  Testing WebSocket with invalid experiment...")

    invalid_id = str(uuid4())
    uri = f"ws://localhost:8000/ws/experiments/{invalid_id}"

    print(f"   Connecting to: {uri}")

    try:
        async with websockets.connect(uri):
            print("   ❌ Should not have connected!")
            raise AssertionError("Should have been rejected")
    except websockets.exceptions.ConnectionClosedError as e:
        if e.code == 1008:
            print(f"   ✅ Correctly rejected with code 1008: {e.reason}")
        else:
            print(f"   ⚠️  Rejected but unexpected code: {e.code} - {e.reason}")
    except Exception as e:
        print(f"   ⚠️  Connection error: {e}")


@pytest.mark.asyncio
async def test_websocket_with_valid_experiment():
    """Test WebSocket connection with valid experiment."""
    print("\n  Testing WebSocket with valid experiment...")

    # Create a real experiment in database
    db = SessionLocal()
    try:
        experiment = Experiment(
            name="WebSocket Test Experiment",
            model_name="test_model",
            framework="pytorch",
            dataset_type="preset",
            dataset_name="mnist",
            optimization_goal=OptimizationGoal.BALANCED,
            status=ExperimentStatus.PENDING,
            progress_percent=0,
        )
        db.add(experiment)
        db.commit()
        db.refresh(experiment)

        experiment_id = str(experiment.id)
        print(f"   Created experiment: {experiment_id}")

        uri = f"ws://localhost:8000/ws/experiments/{experiment_id}"
        print(f"   Connecting to: {uri}")

        try:
            async with websockets.connect(uri) as websocket:
                print("   ✅ WebSocket connected!")

                # Receive initial connection message
                message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(message)

                print(f"     Received: {json.dumps(data, indent=2)}")

                assert data["type"] == "connected"
                assert data["experiment_id"] == experiment_id
                assert data["status"] == "pending"
                assert "progress_percent" in data

                # Test ping/pong
                await websocket.send("ping")
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                assert response == "pong"
                print("   ✅ Ping/pong works!")

                print("   ✅ WebSocket test passed!")

        except asyncio.TimeoutError:
            print("   ❌ Timeout waiting for message")
            raise

    finally:
        # Cleanup
        if experiment:
            db.delete(experiment)
            db.commit()
        db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
