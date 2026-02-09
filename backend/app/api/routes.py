"""
FastAPI Routes
==============
API endpoints for the workflow.
"""

import logging
import asyncio
import os
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from ..agents.orchestrator import create_workflow, WorkflowState
from .websocket import manager

logger = logging.getLogger("api")
router = APIRouter(prefix="/api", tags=["workflow"])

# Uploads directory - project_root/uploads
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # my_new_code/
UPLOADS_DIR = PROJECT_ROOT / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ProcessRequest(BaseModel):
    """Request model for processing patient data"""
    patient_hash: str
    raw_text: Optional[str] = None
    device_id: str = "device_001"
    offline: bool = False


class ProcessResponse(BaseModel):
    """Response model after processing"""
    success: bool
    patient_hash: str
    analysis: Optional[dict] = None
    verdict: Optional[dict] = None
    approved: bool = False
    logs: List[str] = []
    error: Optional[str] = None


class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    context: Optional[dict] = None


class ChatResponse(BaseModel):
    """Chat response model"""
    reply: str
    

# ============================================================================
# ROUTES
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "convolve-backend"}


@router.post("/process", response_model=ProcessResponse)
async def process_patient(request: ProcessRequest):
    """Process patient data through the full workflow."""
    try:
        logger.info(f"Processing request for patient: {request.patient_hash}")
        await manager.send_log(f"Starting processing for patient: {request.patient_hash}")
        
        workflow = create_workflow()
        await manager.send_log("Workflow created, initializing agents...")
        
        initial_state: WorkflowState = {
            "patient_hash": request.patient_hash,
            "raw_text": request.raw_text,
            "audio_path": None,
            "image_paths": None,
            "document_paths": None,
            "device_id": request.device_id,
            "offline": request.offline,
            "patient_data": None,
            "context_payload": None,
            "retrieval_plan": None,
            "patient_current_state": None,
            "retrieved_cases": None,
            "case_analysis": None,
            "review_verdict": None,
            "final_output": None,
            "logs": [],
            "error": None
        }
        
        await manager.send_step("Ingestion", "active")
        await manager.send_thinking("Planning: Starting data ingestion and processing...")
        
        final_state = workflow.invoke(initial_state)
        
        for log in final_state.get("logs", []):
            await manager.send_log(log)
        
        if final_state.get("error"):
            await manager.send_error(final_state["error"])
            return ProcessResponse(
                success=False,
                patient_hash=request.patient_hash,
                error=final_state["error"],
                logs=final_state.get("logs", [])
            )
        
        output = final_state.get("final_output", {})
        await manager.send_result(output)
        await manager.send_complete()
        
        return ProcessResponse(
            success=True,
            patient_hash=request.patient_hash,
            analysis=output.get("analysis"),
            verdict=output.get("verdict"),
            approved=output.get("approved", False),
            logs=final_state.get("logs", [])
        )
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        await manager.send_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process/files", response_model=ProcessResponse)
async def process_with_files(
    patient_hash: str = Form(...),
    raw_text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    images: Optional[List[UploadFile]] = File(None),
    documents: Optional[List[UploadFile]] = File(None),
    device_id: str = Form("device_001"),
    offline: bool = Form(False)
):
    """Process patient data with file uploads - uses AsyncOrchestrator for real-time logs."""
    from ..agents.orchestrator import AsyncOrchestrator, WorkflowState
    from datetime import datetime
    
    try:
        logger.info(f"Processing files for patient: {patient_hash}")
        await manager.send_log(f"Received request for patient: {patient_hash}")
        
        # Create folder name: patient_hash_YYYYMMDD_HHMMSS
        safe_patient_hash = patient_hash.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{safe_patient_hash}_{timestamp}"
        
        # Create structured folders: uploads/patient_hash_timestamp/{audio,images,documents}
        patient_folder = os.path.join(UPLOADS_DIR, folder_name)
        audio_folder = os.path.join(patient_folder, "audio")
        images_folder = os.path.join(patient_folder, "images")
        docs_folder = os.path.join(patient_folder, "documents")
        
        try:
            os.makedirs(audio_folder, exist_ok=True)
            os.makedirs(images_folder, exist_ok=True)
            os.makedirs(docs_folder, exist_ok=True)
            logger.info(f"Created uploads folder: {patient_folder}")
        except Exception as e:
            logger.error(f"Failed to create folder: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create uploads folder: {e}")
        
        audio_path = None
        image_paths = []
        document_paths = []
        
        if audio:
            await manager.send_log(f"Saving audio: {audio.filename}")
            audio_path = os.path.join(audio_folder, audio.filename)
            with open(audio_path, "wb") as f:
                f.write(await audio.read())
                
        if images:
            await manager.send_log(f"Saving {len(images)} image(s)")
            for img in images:
                img_path = os.path.join(images_folder, img.filename)
                with open(img_path, "wb") as f:
                    f.write(await img.read())
                image_paths.append(img_path)
                
        if documents:
            await manager.send_log(f"Saving {len(documents)} document(s)")
            for doc in documents:
                doc_path = os.path.join(docs_folder, doc.filename)
                with open(doc_path, "wb") as f:
                    f.write(await doc.read())
                document_paths.append(doc_path)
        
        await manager.send_log(f"Files saved to: uploads/{folder_name}/")
        
        # Create async orchestrator with WebSocket callbacks
        orchestrator = AsyncOrchestrator(
            log_callback=manager.send_log,
            step_callback=manager.send_step,
            thinking_callback=manager.send_thinking
        )
        
        initial_state: WorkflowState = {
            "patient_hash": patient_hash,
            "raw_text": raw_text,
            "audio_path": audio_path,
            "image_paths": image_paths if image_paths else None,
            "document_paths": document_paths if document_paths else None,
            "device_id": device_id,
            "offline": offline,
            "patient_data": None,
            "context_payload": None,
            "retrieval_plan": None,
            "patient_current_state": None,
            "retrieved_cases": None,
            "case_analysis": None,
            "review_verdict": None,
            "final_output": None,
            "logs": [],
            "error": None
        }
        
        # Run async workflow with real-time logging
        final_state = await orchestrator.run(initial_state)
        
        if final_state.get("error"):
            await manager.send_error(final_state["error"])
            return ProcessResponse(
                success=False,
                patient_hash=patient_hash,
                error=final_state["error"],
                logs=final_state.get("logs", [])
            )
        
        output = final_state.get("final_output", {})
        await manager.send_result(output)
        await manager.send_log("Workflow completed successfully!", "success")
        await manager.send_complete()
        
        return ProcessResponse(
            success=True,
            patient_hash=patient_hash,
            analysis=output.get("analysis"),
            verdict=output.get("verdict"),
            approved=output.get("approved", False),
            logs=final_state.get("logs", [])
        )
        
    except Exception as e:
        logger.exception(f"File processing error: {e}")
        await manager.send_error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat_with_context(request: ChatRequest):
    """
    Chat endpoint with workflow context.
    Acts as an intelligent supervisor for ASHA workers.
    """
    from ..tools.llm_client import GeminiClient
    
    try:
        llm = GeminiClient()
        
        # Build rich context from workflow results
        context_str = ""
        if request.context:
            patient = request.context.get('patient_hash', 'Unknown Patient')
            analysis = request.context.get('analysis', {})
            verdict = request.context.get('verdict', {})
            
            context_str = f"""
PATIENT ANALYSIS CONTEXT:
========================
Patient ID: {patient}

Analysis Summary: {analysis.get('summary', 'No summary available')}
Detailed Explanation: {analysis.get('explanation', 'No explanation available')}

Review Verdict: {'APPROVED' if request.context.get('approved') else 'NEEDS ATTENTION'}
Reviewer Notes: {verdict.get('reasoning', 'No notes')}

Full Analysis Data: {analysis}
"""
        
        prompt = f"""
You are a Medical Supervisor helping an ASHA worker. Give SHORT, CLEAR responses.

PATIENT DATA:
{context_str}

QUESTION: {request.message}

RESPOND WITH:
- 2-3 bullet points MAX
- Simple language (ASHA worker can understand)
- Specific action if needed
- Keep it under 100 words

DO NOT use markdown headers or excessive formatting. Just plain text bullets.
"""
        
        reply = llm.generate(prompt)
        
        if not reply:
            reply = "I couldn't generate a response. Please try again or contact the supervisor directly."
            
        return ChatResponse(reply=reply)
        
    except Exception as e:
        logger.exception(f"Chat error: {e}")
        return ChatResponse(reply=f"Error connecting to AI: {str(e)}")
