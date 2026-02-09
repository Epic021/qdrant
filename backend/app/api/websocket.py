"""
WebSocket Manager
=================
Handles WebSocket connections for real-time log streaming.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("websocket")


class ConnectionManager:
    """Manages WebSocket connections for broadcasting logs."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
        
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return
            
        data = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(data)
            except Exception:
                disconnected.append(connection)
                
        for conn in disconnected:
            self.disconnect(conn)
            
    async def send_log(self, message: str, level: str = "info"):
        """Send a log message to all clients."""
        await self.broadcast({
            "type": "log",
            "message": message,
            "level": level
        })
        
    async def send_step(self, step: str, status: str = "active"):
        """Send workflow step update."""
        await self.broadcast({
            "type": "step",
            "step": step,
            "status": status
        })
        
    async def send_thinking(self, message: str):
        """Send agent thinking/planning message."""
        await self.broadcast({
            "type": "thinking",
            "message": message
        })
        
    async def send_result(self, data: Dict[str, Any]):
        """Send final result."""
        await self.broadcast({
            "type": "result",
            "data": data
        })
        
    async def send_error(self, message: str):
        """Send error message."""
        await self.broadcast({
            "type": "error",
            "message": message
        })
        
    async def send_complete(self):
        """Send workflow complete signal."""
        await self.broadcast({
            "type": "complete"
        })


# Global connection manager
manager = ConnectionManager()
