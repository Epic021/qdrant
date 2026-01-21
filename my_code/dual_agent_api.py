"""
FastAPI Backend with Both Agents
=================================
Agent 1: Field Ingestion Agent
Agent 2: Patient Memory Agent
"""

import os
import time
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import agents
import sys
sys.path.append(os.path.dirname(__file__))

from agents import FieldIngestionAgent
from db import IngestionDatabase
from patient_memory_agent import PatientMemoryAgent

# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(
    title="Dual Agent API - Field Ingestion + Patient Memory",
    description="Process patient data and store in Qdrant",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = IngestionDatabase(db_path="ingestion.db", storage_dir="uploads")

# ============================================================================
# MODELS
# ============================================================================

class DualAgentResponse(BaseModel):
    success: bool
    interaction_id: str
    patient_hash: str
    
    # Agent 1 outputs
    patient_data: dict
    agent_1_logs: List[str]
    
    # Agent 2 outputs
    event_id: str
    context_memory_written: bool
    patient_current_state: dict
    agent_2_logs: List[str]
    
    # Combined
    all_logs: List[str]
    processing_time_ms: int


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Dual Agent API - Field Ingestion + Patient Memory",
        "endpoints": {
            "/api/process": "Process patient data through both agents",
            "/api/patient/{patient_hash}/history": "Get patient history from Qdrant",
            "/api/health": "Health check"
        }
    }


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "agents": ["field_ingestion", "patient_memory"]}


@app.post("/api/process", response_model=DualAgentResponse)
async def process_dual_agents(
    patient_hash: str = Form(...),
    text_notes: Optional[str] = Form(None),
    device_id: str = Form("web_device"),
    offline: bool = Form(False),
    audio: Optional[UploadFile] = File(None),
    images: Optional[List[UploadFile]] = File(None),
    documents: Optional[List[UploadFile]] = File(None)
):
    """
    Process patient data through BOTH agents:
    1. Field Ingestion Agent
    2. Patient Memory Agent
    
    Returns logs from both agents.
    """
    start_time = time.time()
    all_logs = []
    
    # Generate interaction ID
    import uuid
    interaction_id = str(uuid.uuid4())
    timestamp = int(time.time())
    
    try:
        # ===== AGENT 1: FIELD INGESTION =====
        agent_1_logs = []
        
        def agent_1_callback(msg):
            agent_1_logs.append(msg)
            all_logs.append(f"[Agent 1] {msg}")
        
        agent_1 = FieldIngestionAgent(log_callback=agent_1_callback)
        
        # Save files
        audio_path = None
        image_paths = []
        document_paths = []
        
        if audio and audio.filename:
            content = await audio.read()
            audio_path = db.save_uploaded_file(
                content, audio.filename, "audio", 
                interaction_id, patient_hash
            )
        
        if images:
            for img in images:
                if img.filename:
                    content = await img.read()
                    path = db.save_uploaded_file(
                        content, img.filename, "images",
                        interaction_id, patient_hash
                    )
                    image_paths.append(path)
        
        if documents:
            for doc in documents:
                if doc.filename:
                    content = await doc.read()
                    path = db.save_uploaded_file(
                        content, doc.filename, "documents",
                        interaction_id, patient_hash
                    )
                    document_paths.append(path)
        
        # Process through Agent 1
        patient_data = agent_1.process(
            patient_hash=patient_hash,
            raw_text=text_notes,
            audio_path=audio_path,
            image_paths=image_paths,
            document_paths=document_paths,
            device_id=device_id,
            offline=offline,
            timestamp=timestamp
        )
        
        # Store in database
        db.store_raw_interaction(
            interaction_id, patient_hash, text_notes,
            audio_path, image_paths, document_paths, timestamp
        )
        
        db.store_processed_interaction(
            interaction_id, patient_hash, patient_data,
            patient_data.get("uncertainty_flags", {}),
            timestamp
        )
        
        # ===== AGENT 2: PATIENT MEMORY =====
        agent_2_logs = []
        
        def agent_2_callback(msg):
            agent_2_logs.append(msg)
            all_logs.append(f"[Agent 2] {msg}")
        
        agent_2 = PatientMemoryAgent(log_callback=agent_2_callback)
        
        # Process through Agent 2
        result_2 = agent_2.process(patient_data)
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return DualAgentResponse(
            success=True,
            interaction_id=interaction_id,
            patient_hash=patient_hash,
            patient_data=patient_data,
            agent_1_logs=agent_1_logs,
            event_id=result_2["event_id"],
            context_memory_written=result_2["context_memory_written"],
            patient_current_state=result_2["patient_current_state"],
            agent_2_logs=agent_2_logs,
            all_logs=all_logs,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": str(e),
            "logs": all_logs
        })


@app.get("/api/patient/{patient_hash}/history")
async def get_patient_history(patient_hash: str):
    """Get patient history from Qdrant"""
    try:
        from qdrant_client_module import PatientMemoryQdrantClient
        client = PatientMemoryQdrantClient()
        
        history = client.get_patient_history(patient_hash)
        
        return {
            "patient_hash": patient_hash,
            "total_visits": len(history),
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("🚀 Dual Agent API - Field Ingestion + Patient Memory")
    print("=" * 60)
    print("\n📍 Server starting at http://localhost:8000")
    print("\nEndpoints:")
    print("  POST /api/process       - Process patient data")
    print("  GET  /api/patient/{hash}/history - Get patient history")
    print("  GET  /api/health        - Health check")
    print("\n" + "=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
