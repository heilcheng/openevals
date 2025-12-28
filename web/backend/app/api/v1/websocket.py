"""WebSocket endpoint for real-time benchmark progress."""

import json
import asyncio
from datetime import datetime
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

# Active WebSocket connections per run_id
active_connections: Dict[str, Set[WebSocket]] = {}


class ConnectionManager:
    """Manages WebSocket connections for benchmark progress updates."""

    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, run_id: str):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        if run_id not in self.connections:
            self.connections[run_id] = set()
        self.connections[run_id].add(websocket)

    def disconnect(self, websocket: WebSocket, run_id: str):
        """Remove a WebSocket connection."""
        if run_id in self.connections:
            self.connections[run_id].discard(websocket)
            if not self.connections[run_id]:
                del self.connections[run_id]

    async def broadcast(self, run_id: str, message: dict):
        """Send a message to all connections watching a run."""
        if run_id not in self.connections:
            return

        message_json = json.dumps(message, default=str)
        disconnected = set()

        for websocket in self.connections[run_id]:
            try:
                await websocket.send_text(message_json)
            except Exception:
                disconnected.add(websocket)

        # Clean up disconnected clients
        for ws in disconnected:
            self.connections[run_id].discard(ws)


manager = ConnectionManager()


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance."""
    return manager


@router.websocket("/ws/benchmark/{run_id}")
async def benchmark_progress(websocket: WebSocket, run_id: str):
    """
    WebSocket endpoint for receiving real-time benchmark progress.

    Clients connect to this endpoint to receive updates about a specific
    benchmark run's progress, including:
    - Status changes (pending -> running -> completed/failed)
    - Current model being evaluated
    - Current task being run
    - Progress percentage
    - Log messages
    """
    await manager.connect(websocket, run_id)

    try:
        # Send initial connection confirmation
        await websocket.send_text(
            json.dumps(
                {
                    "type": "connected",
                    "run_id": run_id,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        )

        # Keep connection alive and handle any incoming messages
        while True:
            try:
                # Wait for messages (ping/pong, or client requests)
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=30.0  # Heartbeat interval
                )

                # Handle ping
                if data == "ping":
                    await websocket.send_text("pong")

            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "heartbeat",
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                    )
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, run_id)


async def send_progress_update(run_id: str, update: dict):
    """
    Send a progress update to all connected clients for a run.

    This is called from the benchmark execution code to broadcast
    progress updates in real-time.
    """
    update["type"] = "progress"
    update["timestamp"] = datetime.utcnow().isoformat()
    await manager.broadcast(run_id, update)
