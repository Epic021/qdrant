"""
Convolve Backend Main Application
===============================
FastAPI application entry point.
"""

import os
# Fix protobuf compatibility issue
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import logging
import sys
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import API router and WebSocket manager
from .api import router as api_router
from .api.websocket import manager

# Create FastAPI app
app = FastAPI(
    title="Convolve Backend API",
    description="Patient Data Processing Workflow API",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)


@app.get("/")
async def root():
    return {
        "service": "Convolve Backend API",
        "version": "2.0.0",
        "status": "running"
    }


@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, wait for messages
            data = await websocket.receive_text()
            # Echo back for connection health
            await websocket.send_text(f"ack: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.on_event("startup")
async def startup_event():
    logging.info("Convolve Backend API starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    logging.info("Convolve Backend API shutting down...")
