"""
FastAPI Backend for Field Ingestion Agent
==========================================
RESTful API with file upload support and streaming logs.
"""

import os
import time
import uuid
import json
import asyncio
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our modules
from agents import FieldIngestionAgent
from db import IngestionDatabase
from patient_memory_agent import PatientMemoryAgent  # AGENT 2
from context_builder_agent import ContextBuilderAgent  # AGENT 3
from similar_case_retrieval_agent import SimilarCaseRetrievalAgent  # AGENT 4
from explanation_and_referral_agents import (  # AGENTS 5 & 6
    CaseSummaryBuilder,
    ExplanationAndTrustAgent,
    ReferralAndFollowupAgent
)


# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(
    title="Field Ingestion Agent API",
    description="Process multimodal patient data for rural healthcare",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = IngestionDatabase(
    db_path="ingestion.db",
    storage_dir="uploads"
)

# Ensure static directory exists
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============================================================================
# MODELS
# ============================================================================

class ProcessingResponse(BaseModel):
    success: bool
    interaction_id: str
    patient_data: dict
    logs: List[str]  # Combined logs from both agents
    processing_time_ms: int
    
    # Agent 2 outputs
    event_id: str = None
    context_memory_written: bool = False
    patient_current_state: dict = None
    
    # Agent 3 outputs
    retrieval_plan: dict = None
    
    # Agent 4 outputs
    retrieved_cases: list = None
    retrieval_metadata: dict = None
    
    # Agents 5 & 6 outputs
    case_summary: dict = None
    explanation_text: str = None
    followup_output: dict = None


class StatsResponse(BaseModel):
    raw_interactions: int
    processed_interactions: int
    uncertainty_flag_counts: dict


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend"""
    html_path = static_dir / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Frontend not found. Place index.html in static/</h1>")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/process", response_model=ProcessingResponse)
async def process_patient_data(
    patient_hash: str = Form(...),
    text_notes: Optional[str] = Form(None),
    device_id: str = Form("web_device"),
    offline: bool = Form(False),
    audio: Optional[UploadFile] = File(None),
    images: Optional[List[UploadFile]] = File(None),
    documents: Optional[List[UploadFile]] = File(None)
):
    """
    Process patient data through the Field Ingestion Agent.
    
    Accepts:
    - patient_hash: Anonymized patient identifier
    - text_notes: Free-text clinical notes
    - audio: Audio recording file
    - images: One or more image files
    - documents: Document files (PDF, etc.)
    
    Returns:
    - patient_data: Canonical JSON
    - logs: Step-by-step processing logs
    """
    start_time = time.time()
    
    # Generate interaction ID
    interaction_id = str(uuid.uuid4())
    timestamp = int(time.time())
    
    # Collect logs
    log_messages = []
    
    def log_callback(message: str):
        log_messages.append(message)
    
    # Create agent with logging callback
    agent = FieldIngestionAgent(log_callback=log_callback)
    
    try:
        # ===== Save uploaded files =====
        audio_path = None
        image_paths = []
        document_paths = []
        
        # Save audio
        if audio and audio.filename:
            log_messages.append(f"[✓] Audio file received: {audio.filename}")
            content = await audio.read()
            audio_path = db.save_uploaded_file(
                file_content=content,
                filename=audio.filename,
                file_type="audio",
                interaction_id=interaction_id,
                patient_hash=patient_hash
            )
        
        # Save images
        if images:
            for img in images:
                if img.filename:
                    log_messages.append(f"[✓] Image file received: {img.filename}")
                    content = await img.read()
                    path = db.save_uploaded_file(
                        file_content=content,
                        filename=img.filename,
                        file_type="images",
                        interaction_id=interaction_id,
                        patient_hash=patient_hash
                    )
                    image_paths.append(path)
        
        # Save documents
        if documents:
            for doc in documents:
                if doc.filename:
                    log_messages.append(f"[✓] Document file received: {doc.filename}")
                    content = await doc.read()
                    path = db.save_uploaded_file(
                        file_content=content,
                        filename=doc.filename,
                        file_type="documents",
                        interaction_id=interaction_id,
                        patient_hash=patient_hash
                    )
                    document_paths.append(path)
        
        # Log text input
        if text_notes:
            log_messages.append(f"[✓] Text notes received ({len(text_notes)} chars)")
            # Save text to file
            text_path = db.save_text_input(
                text=text_notes,
                interaction_id=interaction_id,
                patient_hash=patient_hash
            )
            log_messages.append(f"[✓] Text notes saved to file")
        
        # ===== Store raw interaction =====
        db.store_raw_interaction(
            interaction_id=interaction_id,
            patient_hash=patient_hash,
            raw_text=text_notes,
            raw_audio_path=audio_path,
            raw_image_paths=image_paths if image_paths else None,
            raw_document_paths=document_paths if document_paths else None,
            timestamp=timestamp
        )
        log_messages.append(f"[✓] Raw data stored in database")
        
        # ===== Process through agent =====
        patient_data = agent.process(
            patient_hash=patient_hash,
            raw_text=text_notes,
            audio_path=audio_path,
            image_paths=image_paths if image_paths else None,
            document_paths=document_paths if document_paths else None,
            device_id=device_id,
            offline=offline,
            timestamp=timestamp
        )
        
        # Update interaction_id in patient_data (agent generates its own)
        patient_data["interaction_id"] = interaction_id
        
        # ===== Save patient_data JSON to file =====
        json_path = db.save_patient_data_json(
            patient_data=patient_data,
            interaction_id=interaction_id,
            patient_hash=patient_hash
        )
        log_messages.append(f"[✓] patient_data JSON saved to file")
        
        # ===== Store processed interaction =====
        db.store_processed_interaction(
            interaction_id=interaction_id,
            patient_hash=patient_hash,
            patient_data=patient_data,
            uncertainty_flags=patient_data.get("uncertainty_flags", {}),
            processing_timestamp=int(time.time())
        )
        log_messages.append(f"[✓] Processed data stored in database")
        
        # ===== AGENT 2: PATIENT MEMORY AGENT =====
        log_messages.append(f"")
        log_messages.append(f"[ℹ] ========== AGENT 2: Patient Memory Agent ==========")
        
        # Create Agent 2 with separate logging
        agent_2_logs = []
        def agent_2_callback(msg):
            agent_2_logs.append(msg)
            log_messages.append(f"[Agent 2] {msg}")
        
        agent_2 = PatientMemoryAgent(log_callback=agent_2_callback)
        
        try:
            # Process through Patient Memory Agent
            result_2 = agent_2.process(patient_data)
            
            event_id = result_2.get("event_id")
            context_memory_written = result_2.get("context_memory_written")
            patient_current_state = result_2.get("patient_current_state")
            
            if context_memory_written:
                log_messages.append(f"[✓] Event stored in Qdrant cluster: {event_id}")
                log_messages.append(f"[✓] Patient has {patient_current_state.get('total_visits', 0)} total visit(s)")
            else:
                log_messages.append(f"[⚠] Warning: Event not written to Qdrant")
                
        except Exception as e:
            log_messages.append(f"[✗] Agent 2 Error: {str(e)}")
            event_id = None
            context_memory_written = False
            patient_current_state = {}
        
        log_messages.append(f"[ℹ] ====================================================")
        
        # ===== STORE PATIENT STATE IN DATABASE =====
        if context_memory_written and patient_current_state:
            try:
                db.store_patient_state(
                    interaction_id=interaction_id,
                    patient_hash=patient_hash,
                    event_id=event_id,
                    patient_current_state=patient_current_state
                )
                log_messages.append(f"[✓] Patient current state saved to database")
                
                # Save patient_current_state to JSON file
                state_json_path = db.save_patient_current_state_json(
                    patient_current_state=patient_current_state,
                    interaction_id=interaction_id,
                    patient_hash=patient_hash,
                    event_id=event_id
                )
                log_messages.append(f"[✓] Patient current state saved to file")
                
            except Exception as e:
                log_messages.append(f"[⚠] Could not save patient state to database: {e}")
        
        log_messages.append(f"[ℹ] ====================================================")
        
        # ===== AGENT 3: CONTEXT BUILDER AGENT =====
        log_messages.append(f"")
        log_messages.append(f"[ℹ] ========== AGENT 3: Context Builder Agent ==========")
        
        # Create Agent 3
        agent_3 = ContextBuilderAgent()
        
        try:
            # Process through Context Builder Agent
            state_3 = {
                "patient_hash": patient_hash,
                "event_id": event_id,
                "patient_current_state": patient_current_state
            }
            
            result_3 = agent_3.process(state_3)
            retrieval_plan = result_3.get("retrieval_plan")
            
            if retrieval_plan:
                log_messages.append(f"[✓] Retrieval plan generated")
                log_messages.append(f"[ℹ] Modality policy: text={retrieval_plan['modality_policy']['use_text']}, image={retrieval_plan['modality_policy']['use_image']}, audio={retrieval_plan['modality_policy']['use_audio']}")
                log_messages.append(f"[ℹ] Risk flags: {len(retrieval_plan.get('risk_flags', []))}")
                
                # Store retrieval plan in database
                db.store_retrieval_plan(
                    interaction_id=interaction_id,
                    patient_hash=patient_hash,
                    event_id=event_id,
                    retrieval_plan=retrieval_plan
                )
                log_messages.append(f"[✓] Retrieval plan saved to database")
                
                # Save retrieval plan to JSON file
                plan_json_path = db.save_retrieval_plan_json(
                    retrieval_plan=retrieval_plan,
                    interaction_id=interaction_id,
                    patient_hash=patient_hash,
                    event_id=event_id
                )
                log_messages.append(f"[✓] Retrieval plan saved to file")
                
            else:
                log_messages.append(f"[⚠] No retrieval plan generated")
                retrieval_plan = {}
                
        except Exception as e:
            log_messages.append(f"[✗] Agent 3 Error: {str(e)}")
            retrieval_plan = {}
        
        log_messages.append(f"[ℹ] ====================================================")
        
        # ===== AGENT 4: SIMILAR CASE RETRIEVAL AGENT =====
        log_messages.append(f"")
        log_messages.append(f"[ℹ] ========== AGENT 4: Similar Case Retrieval ==========")
        
        agent_4 = SimilarCaseRetrievalAgent(log_callback=lambda msg: log_messages.append(msg))
        
        try:
            state_4 = {
                "patient_hash": patient_hash,
                "event_id": event_id,
                "retrieval_plan": retrieval_plan,
                "patient_current_state": patient_current_state
            }
            
            result_4 = agent_4.process(state_4)
            retrieved_cases = result_4.get("retrieved_cases", [])
            retrieval_metadata = result_4.get("retrieval_metadata", {})
            
            if retrieved_cases:
                log_messages.append(f"[✓] Retrieved {len(retrieved_cases)} similar case(s)")
                db.store_retrieval_results(
                    interaction_id=interaction_id,
                    patient_hash=patient_hash,
                    event_id=event_id,
                    retrieved_cases=retrieved_cases,
                    retrieval_metadata=retrieval_metadata
                )
                log_messages.append(f"[✓] Retrieval results saved to database")
                
                # Save retrieved cases to JSON file
                cases_json_path = db.save_retrieved_cases_json(
                    retrieved_cases=retrieved_cases,
                    retrieval_metadata=retrieval_metadata,
                    interaction_id=interaction_id,
                    patient_hash=patient_hash,
                    event_id=event_id
                )
                log_messages.append(f"[✓] Retrieved cases saved to file")
                
            else:
                log_messages.append(f"[ℹ] No similar cases found")
                
        except Exception as e:
            log_messages.append(f"[✗] Agent 4 Error: {str(e)}")
            retrieved_cases = []
            retrieval_metadata = {}
        
        log_messages.append(f"[ℹ] ====================================================")
        
        # ===== AGENTS 5 & 6: EXPLANATION & REFERRAL =====
        log_messages.append(f"")
        log_messages.append(f"[ℹ] ========== AGENTS 5 & 6: Explanation & Referral =========")
        
        # Build case summary (deterministic)
        log_messages.append(f"[CaseSummary] Building case summary...")
        case_summary = CaseSummaryBuilder.build({
            "patient_data": patient_data,
            "patient_current_state": patient_current_state,
            "retrieval_plan": retrieval_plan,
            "retrieved_cases": retrieved_cases,
            "retrieval_metadata": retrieval_metadata
        })
        log_messages.append(f"[CaseSummary] ✓ Case summary built")
        
        # Agent 5: Explanation
        agent_5 = ExplanationAndTrustAgent(log_callback=lambda msg: log_messages.append(msg))
        
        try:
            result_5 = agent_5.process({
                "case_summary": case_summary,
                "retrieved_cases": retrieved_cases
            })
            explanation_text = result_5.get("explanation_text", "")
            log_messages.append(f"[✓] Explanation generated")
        except Exception as e:
            log_messages.append(f"[✗] Agent 5 Error: {str(e)}")
            explanation_text = "Unable to generate explanation at this time."
        
        # Agent 6: Referral & Follow-up
        agent_6 = ReferralAndFollowupAgent(log_callback=lambda msg: log_messages.append(msg))
        
        try:
            result_6 = agent_6.process({
                "patient_data": patient_data,
                "patient_current_state": patient_current_state
            })
            followup_output = result_6.get("followup_output", {})
            log_messages.append(f"[✓] Follow-up status evaluated")
            
            if followup_output.get("escalation_flag"):
                log_messages.append(f"[⚠] Escalation flag raised", "[Warning]")
                
        except Exception as e:
            log_messages.append(f"[✗] Agent 6 Error: {str(e)}")
            followup_output = {}
        
        # Store final outputs in database
        try:
            db.store_final_case_outputs(
                interaction_id=interaction_id,
                patient_hash=patient_hash,
                case_summary=case_summary,
                explanation_text=explanation_text,
                followup_output=followup_output
            )
            log_messages.append(f"[✓] Final outputs saved to database")
            
            # Save final recommendations to JSON file
            final_json_path = db.save_final_recommendations_json(
                case_summary=case_summary,
                explanation_text=explanation_text,
                followup_output=followup_output,
                interaction_id=interaction_id,
                patient_hash=patient_hash
            )
            log_messages.append(f"[✓] Final recommendations saved to file")
            
        except Exception as e:
            log_messages.append(f"[⚠] Could not save final outputs: {e}")
        
        log_messages.append(f"[ℹ] ====================================================")
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return ProcessingResponse(
            success=True,
            interaction_id=interaction_id,
            patient_data=patient_data,
            logs=log_messages,
            processing_time_ms=processing_time_ms,
            event_id=event_id,
            context_memory_written=context_memory_written,
            patient_current_state=patient_current_state,
            retrieval_plan=retrieval_plan,
            retrieved_cases=retrieved_cases,
            retrieval_metadata=retrieval_metadata,
            case_summary=case_summary,
            explanation_text=explanation_text,
            followup_output=followup_output
        )
        
    except Exception as e:
        log_messages.append(f"[✗] Error: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "error": str(e),
            "logs": log_messages
        })


@app.get("/api/interactions/{interaction_id}")
async def get_interaction(interaction_id: str):
    """Get a specific interaction by ID"""
    raw = db.get_raw_interaction(interaction_id)
    processed = db.get_processed_interaction(interaction_id)
    
    if not raw and not processed:
        raise HTTPException(status_code=404, detail="Interaction not found")
    
    return {
        "raw": raw,
        "processed": processed
    }


@app.get("/api/interactions")
async def list_interactions(limit: int = 20):
    """List recent processed interactions"""
    interactions = db.get_all_processed(limit=limit)
    return {"interactions": interactions, "count": len(interactions)}


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get database statistics"""
    stats = db.get_stats()
    return StatsResponse(**stats)


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("Field Ingestion Agent API")
    print("=" * 60)
    print("\n📍 Open http://localhost:8000 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
