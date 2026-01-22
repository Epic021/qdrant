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
from typing import Optional, List, Dict, Any
from datetime import datetime
import google.generativeai as genai
import numpy as np

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj


from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our modules
from db import IngestionDatabase

# LangGraph workflow integration
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from typing import TypedDict, Dict, Any
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# Import all agents for workflow
from agents import field_ingestion_agent
from patient_memory_agent import patient_memory_agent
from context_builder_agent import context_builder_agent
from similar_case_retrieval_agent import similar_case_retrieval_agent
from explanation_and_referral_agents import (
    CaseSummaryBuilder,
    ExplanationAndTrustAgent,
    ReferralAndFollowupAgent
)




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

# Ensure uploads directory exists
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")




class ProcessingResponse(BaseModel):
    success: bool
    interaction_id: str
    patient_data: dict
    logs: List[str]  # Combined logs from both agents
    processing_time_ms: int
    
    # Agent 2 outputs
    event_id: Optional[str] = None
    context_memory_written: bool = False
    patient_current_state: Optional[Dict[str, Any]] = None
    
    # Agent 3 outputs
    retrieval_plan: Optional[Dict[str, Any]] = None
    
    # Agent 4 outputs
    retrieved_cases: Optional[List[Dict[str, Any]]] = None
    retrieval_metadata: Optional[Dict[str, Any]] = None
    
    # Agents 5 & 6 outputs
    case_summary: Optional[Dict[str, Any]] = None
    explanation_text: Optional[str] = None
    followup_output: Optional[Dict[str, Any]] = None


class StatsResponse(BaseModel):
    raw_interactions: int
    processed_interactions: int
    uncertainty_flag_counts: dict


class ChatRequest(BaseModel):
    interaction_id: str
    message: str




@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend"""
    html_path = static_dir / "index_new.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Frontend not found. Place index.html in static/</h1>")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}




class GeminiClient:
    """Shared Gemini LLM client for all agents"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """Initialize Gemini client with API key from environment"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # Check if using Vertex AI or other auth, otherwise warn
            # For now, just print warning if not found, but don't crash app startup
            print("Warning: GEMINI_API_KEY not found in environment")
        
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
    
    def generate(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.3) -> str:
        """Generate text using Gemini"""
        if not hasattr(self, 'model'):
            return "Error: Gemini API key not configured"
            
        try:
            generation_config = genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def extract_structured_data(self, text: str, fields: Dict[str, str]) -> Dict[str, Any]:
        """Extract structured data from unstructured text"""
        field_descriptions = "\n".join([f"- {name}: {desc}" for name, desc in fields.items()])
        
        prompt = f'''Extract the following information from the clinical text.
If a field cannot be determined, return "unknown".
Return ONLY a JSON object with these fields:

{field_descriptions}

Clinical text:
{text}

JSON output:'''
        
        response = self.generate(prompt, temperature=0.1)
        
        # Try to parse JSON from response
        import json
        try:
            # Find JSON in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        return {}
    
    def summarize(self, text: str, max_sentences: int = 3) -> str:
        """Summarize text concisely"""
        prompt = f'''Summarize the following text in {max_sentences} sentence(s) or less.
Be concise and focus on key medical information.

Text: {text}

Summary:'''
        
        return self.generate(prompt, max_tokens=200, temperature=0.2)


# Singleton instance
_gemini_client = None

def get_gemini_client() -> GeminiClient:
    """Get or create singleton Gemini client"""
    global _gemini_client
    if _gemini_client is None:
        try:
            _gemini_client = GeminiClient()
        except Exception as e:
            print(f"Warning: Could not initialize Gemini client: {e}")
            _gemini_client = None
    return _gemini_client




class AgentState(TypedDict, total=False):
    """State schema for LangGraph workflow"""
    # Input
    patient_hash: str
    raw_text: Optional[str]
    audio_path: Optional[str]
    image_paths: Optional[List[str]]
    document_paths: Optional[List[str]]
    device_id: str
    offline: bool
    timestamp: int
    
    # Agent outputs
    patient_data: Dict[str, Any]
    event_id: str
    context_memory_written: bool
    patient_current_state: Dict[str, Any]
    retrieval_plan: Dict[str, Any]
    retrieved_cases: List[Dict[str, Any]]
    retrieval_metadata: Dict[str, Any]
    case_summary: Dict[str, Any]
    explanation_text: str
    followup_output: Dict[str, Any]
    
    # Logs
    all_logs: List[str]
    agent_1_logs: List[str]
    agent_2_logs: List[str]

def create_workflow_graph():
    """Create the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Agent 1: Field Ingestion
    def agent_1_node(state: AgentState) -> AgentState:
        print(f"\n[BOSS CHECK] ===== AGENT 1 NODE EXECUTING =====")
        print(f"[BOSS CHECK] Input state keys: {list(state.keys())}")
        print(f"[BOSS CHECK] Patient hash: {state.get('patient_hash')}")
        print(f"[BOSS CHECK] Raw text length: {len(state.get('raw_text', '')) if state.get('raw_text') else 0}")
        print(f"[BOSS CHECK] Audio path: {state.get('audio_path')}")
        print(f"[BOSS CHECK] Image paths count: {len(state.get('image_paths', []))}")
        print(f"[BOSS CHECK] Document paths count: {len(state.get('document_paths', []))}")
        
        result = field_ingestion_agent({
            "patient_hash": state["patient_hash"],
            "raw_text": state.get("raw_text"),
            "audio_path": state.get("audio_path"),
            "image_paths": state.get("image_paths"),
            "document_paths": state.get("document_paths"),
            "device_id": state.get("device_id", "device_001"),
            "offline": state.get("offline", False),
            "timestamp": state.get("timestamp")
        })
        
        print(f"[BOSS CHECK] Agent 1 result keys: {list(result.keys())}")
        print(f"[BOSS CHECK] Patient data keys: {list(result['patient_data'].keys())}")
        print(f"[BOSS CHECK] Patient data content: {json.dumps(result['patient_data'], indent=2)[:500]}...")
        print(f"[BOSS CHECK] ===== AGENT 1 COMPLETE =====")
        
        return {
            "patient_data": result["patient_data"],
            "agent_1_logs": result["logs"]
        }
    
    # Agent 2: Patient Memory
    def agent_2_node(state: AgentState) -> AgentState:
        print(f"\n[BOSS CHECK] ===== AGENT 2 NODE EXECUTING =====")
        print(f"[BOSS CHECK] Received patient_data keys: {list(state['patient_data'].keys())}")
        
        result = patient_memory_agent({"patient_data": state["patient_data"]})
        
        print(f"[BOSS CHECK] Agent 2 result keys: {list(result.keys())}")
        print(f"[BOSS CHECK] Event ID: {result.get('event_id')}")
        print(f"[BOSS CHECK] ===== AGENT 2 COMPLETE =====")
        
        return {
            "event_id": result["event_id"],
            "patient_hash": result["patient_hash"],
            "context_memory_written": result["context_memory_written"],
            "patient_current_state": result["patient_current_state"],
            "agent_2_logs": result["logs"]
        }
    
    # Agent 3: Context Builder
    def agent_3_node(state: AgentState) -> AgentState:
        result = context_builder_agent({
            "patient_hash": state["patient_hash"],
            "event_id": state.get("event_id"),
            "patient_current_state": state["patient_current_state"]
        })
        return {"retrieval_plan": result["retrieval_plan"]}
    
    # Agent 4: Similar Case Retrieval
    def agent_4_node(state: AgentState) -> AgentState:
        result = similar_case_retrieval_agent({
            "patient_hash": state["patient_hash"],
            "event_id": state["event_id"],
            "retrieval_plan": state["retrieval_plan"],
            "patient_current_state": state["patient_current_state"]
        })
        return {
            "retrieved_cases": result["retrieved_cases"],
            "retrieval_metadata": result["retrieval_metadata"]
        }
    
    # Agents 5 & 6: Explanation and Referral
    def agent_5_6_node(state: AgentState) -> AgentState:
        case_summary = CaseSummaryBuilder.build({
            "patient_data": state["patient_data"],
            "patient_current_state": state["patient_current_state"],
            "retrieval_plan": state["retrieval_plan"],
            "retrieved_cases": state["retrieved_cases"],
            "retrieval_metadata": state["retrieval_metadata"]
        })
        
        agent_5 = ExplanationAndTrustAgent()
        result_5 = agent_5.process({
            "case_summary": case_summary,
            "retrieved_cases": state["retrieved_cases"]
        })
        
        agent_6 = ReferralAndFollowupAgent()
        result_6 = agent_6.process({
            "patient_data": state["patient_data"],
            "patient_current_state": state["patient_current_state"]
        })
        
        return {
            "case_summary": case_summary,
            "explanation_text": result_5["explanation_text"],
            "followup_output": result_6["followup_output"]
        }
    
    # Finalize
    def finalize_node(state: AgentState) -> AgentState:
        all_logs = []
        all_logs.extend([f"Agent 1: {log}" for log in state.get("agent_1_logs", [])])
        all_logs.extend([f"Agent 2: {log}" for log in state.get("agent_2_logs", [])])
        # Return full state with logs added
        return {**state, "all_logs": all_logs}
    
    # Add nodes
    workflow.add_node("agent_1", agent_1_node)
    workflow.add_node("agent_2", agent_2_node)
    workflow.add_node("agent_3", agent_3_node)
    workflow.add_node("agent_4", agent_4_node)
    workflow.add_node("agent_5_6", agent_5_6_node)
    workflow.add_node("finalize", finalize_node)
    
    # Define flow
    workflow.set_entry_point("agent_1")
    workflow.add_edge("agent_1", "agent_2")
    workflow.add_edge("agent_2", "agent_3")
    workflow.add_edge("agent_3", "agent_4")
    workflow.add_edge("agent_4", "agent_5_6")
    workflow.add_edge("agent_5_6", "finalize")
    workflow.add_edge("finalize", END)
    
    # Compile without checkpointer to avoid msgpack serialization issues with numpy
    return workflow.compile()

# Create workflow at startup if LangGraph available
workflow_app = None
if LANGGRAPH_AVAILABLE:
    try:
        workflow_app = create_workflow_graph()
        print("[SUCCESS] LangGraph workflow initialized")
    except Exception as e:
        print(f"[WARNING] Could not initialize LangGraph: {e}")
        LANGGRAPH_AVAILABLE = False


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
    Process patient data through complete LangGraph workflow.
    Orchestrates all 6 agents automatically.
    """
    start_time = time.time()
    
    # Generate interaction ID
    interaction_id = str(uuid.uuid4())
    timestamp = int(time.time())
    
    log_messages = []
    
    try:
        # ===== Save uploaded files =====
        audio_path = None
        image_paths = []
        document_paths = []
        
        # Save audio
        if audio and audio.filename:
            log_messages.append(f"Audio file received: {audio.filename}")
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
                    log_messages.append(f"Image file received: {img.filename}")
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
                    log_messages.append(f"Document file received: {doc.filename}")
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
            log_messages.append(f"Text notes received ({len(text_notes)} chars)")
            # Save text to file
            text_path = db.save_text_input(
                text=text_notes,
                interaction_id=interaction_id,
                patient_hash=patient_hash
            )
            log_messages.append(f"Text notes saved to file")
        
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
        log_messages.append(f"Raw data stored in database")
        
        # ===== EXECUTE LANGGRAPH WORKFLOW =====
        if workflow_app is None:
            raise HTTPException(status_code=500, detail="LangGraph workflow not available")
        
        log_messages.append(f"Starting LangGraph workflow with 6 agents...")
        
        # Prepare initial state for workflow
        initial_state = {
            "patient_hash": patient_hash,
            "raw_text": text_notes,
            "audio_path": audio_path,
            "image_paths": image_paths,
            "document_paths": document_paths,
            "device_id": device_id,
            "offline": offline,
            "timestamp": timestamp
        }
        
        # Execute workflow
        config = {"configurable": {"thread_id": f"{patient_hash}_{interaction_id}"}}
        
        final_state = None
        for state_update in workflow_app.stream(initial_state, config):
            # Stream through workflow states
            final_state = state_update
        
        # Extract final state
        if final_state:
            final_state = list(final_state.values())[-1]
        else:
            raise Exception("Workflow did not complete")
        
        # Extract outputs from workflow
        patient_data = final_state.get("patient_data", {})
        event_id = final_state.get("event_id")
        context_memory_written = final_state.get("context_memory_written", False)
        patient_current_state = final_state.get("patient_current_state", {})
        retrieval_plan = final_state.get("retrieval_plan", {})
        retrieved_cases = final_state.get("retrieved_cases", [])
        retrieval_metadata = final_state.get("retrieval_metadata", {})
        case_summary = final_state.get("case_summary", {})
        explanation_text = final_state.get("explanation_text", "")
        followup_output = final_state.get("followup_output", {})
        
        # Get workflow logs
        workflow_logs = final_state.get("all_logs", [])
        log_messages.extend(workflow_logs)
        
        log_messages.append(f"LangGraph workflow completed successfully")
        
        # Update interaction_id in patient_data
        patient_data["interaction_id"] = interaction_id
        
        # ===== Save patient_data JSON to file =====
        json_path = db.save_patient_data_json(
            patient_data=patient_data,
            interaction_id=interaction_id,
            patient_hash=patient_hash
        )
        log_messages.append(f"patient_data JSON saved to file")
        
        # ===== Store processed interaction =====
        db.store_processed_interaction(
            interaction_id=interaction_id,
            patient_hash=patient_hash,
            patient_data=patient_data,
            uncertainty_flags=patient_data.get("uncertainty_flags", {}),
            processing_timestamp=int(time.time())
        )
        log_messages.append(f"Processed data stored in database")
        
        # ===== STORE ALL RESULTS IN DATABASE =====
        if context_memory_written and patient_current_state:
            try:
                db.store_patient_state(
                    interaction_id=interaction_id,
                    patient_hash=patient_hash,
                    event_id=event_id,
                    patient_current_state=patient_current_state
                )
                log_messages.append(f"Patient current state saved to database")
                
                state_json_path = db.save_patient_current_state_json(
                    patient_current_state=patient_current_state,
                    interaction_id=interaction_id,
                    patient_hash=patient_hash,
                    event_id=event_id
                )
                log_messages.append(f"Patient current state saved to file")
            except Exception as e:
                log_messages.append(f"Could not save patient state: {e}")
        
        if retrieval_plan:
            try:
                db.store_retrieval_plan(
                    interaction_id=interaction_id,
                    patient_hash=patient_hash,
                    event_id=event_id,
                    retrieval_plan=retrieval_plan
                )
                db.save_retrieval_plan_json(
                    retrieval_plan=retrieval_plan,
                    interaction_id=interaction_id,
                    patient_hash=patient_hash,
                    event_id=event_id
                )
                log_messages.append(f"Retrieval plan saved")
            except Exception as e:
                log_messages.append(f"Could not save retrieval plan: {e}")
        
        if retrieved_cases:
            try:
                db.store_retrieval_results(
                    interaction_id=interaction_id,
                    patient_hash=patient_hash,
                    event_id=event_id,
                    retrieved_cases=retrieved_cases,
                    retrieval_metadata=retrieval_metadata
                )
                db.save_retrieved_cases_json(
                    retrieved_cases=retrieved_cases,
                    retrieval_metadata=retrieval_metadata,
                    interaction_id=interaction_id,
                    patient_hash=patient_hash,
                    event_id=event_id
                )
                log_messages.append(f"Retrieved {len(retrieved_cases)} similar cases")
            except Exception as e:
                log_messages.append(f"Could not save retrieval results: {e}")
        
        if explanation_text or followup_output:
            try:
                db.store_final_case_outputs(
                    interaction_id=interaction_id,
                    patient_hash=patient_hash,
                    case_summary=case_summary,
                    explanation_text=explanation_text,
                    followup_output=followup_output
                )
                db.save_final_recommendations_json(
                    case_summary=case_summary,
                    explanation_text=explanation_text,
                    followup_output=followup_output,
                    interaction_id=interaction_id,
                    patient_hash=patient_hash
                )
                log_messages.append(f"Final recommendations saved")
            except Exception as e:
                log_messages.append(f"Could not save final outputs: {e}")
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return ProcessingResponse(
            success=True,
            interaction_id=interaction_id,
            patient_data=convert_numpy(patient_data),
            logs=log_messages,
            processing_time_ms=processing_time_ms,
            event_id=event_id,
            context_memory_written=context_memory_written,
            patient_current_state=convert_numpy(patient_current_state),
            retrieval_plan=convert_numpy(retrieval_plan),
            retrieved_cases=convert_numpy(retrieved_cases),
            retrieval_metadata=convert_numpy(retrieval_metadata),
            case_summary=convert_numpy(case_summary),
            explanation_text=explanation_text,
            followup_output=convert_numpy(followup_output)
        )
        
    except Exception as e:
        log_messages.append(f"Error: {str(e)}")
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


@app.get("/api/search")
async def search_similar_cases(
    query: str,
    date_range: Optional[str] = "all",
    min_visits: Optional[int] = None,
    quality_filter: str = "all",
    risk_filter: str = "all",
    limit: int = 10
):
    """
    Search for similar cases using natural language query and filters.
    
    Args:
        query: Natural language search query
        date_range: Filter by date ("all", "7", "30", "90" days)
        min_visits: Minimum number of visits
        quality_filter: Data quality filter ("all", "high", "medium")
        risk_filter: Risk level filter ("all", "low", "medium", "high")
        limit: Maximum number of results
    
    Returns:
        List of similar cases with metadata
    """
    from qdrant_manager import PatientMemoryQdrantClient
    from embeddings import MultiModalEmbedder
    import time
    from datetime import datetime, timedelta
    
    try:
        # Initialize clients
        qdrant_client = PatientMemoryQdrantClient()
        embedder = MultiModalEmbedder()
        
        # Generate query embedding
        query_vector = embedder.embed_text(query)
        
        if not query_vector:
            return {
                "success": False,
                "error": "Could not generate query embedding",
                "results": []
            }
        
        # Build filters
        search_filter = {}
        
        # Date range filter
        if date_range != "all":
            days = int(date_range)
            cutoff_timestamp = int((datetime.now() - timedelta(days=days)).timestamp())
            search_filter["timestamp"] = {"gte": cutoff_timestamp}
        
        # Perform vector search
        search_results = qdrant_client.client.search(
            collection_name="patient_context_events_v1",
            query_vector=("text_event", query_vector),
            limit=limit * 2,  # Get more for filtering
            with_payload=True
        )
        
        # Process and filter results
        results = []
        for hit in search_results:
            payload = hit.payload
            
            # Apply filters
            if min_visits and payload.get("visit_count", 0) < min_visits:
                continue
            
            # Quality filter
            if quality_filter == "high":
                uncertainty = payload.get("uncertainty", {})
                if sum(uncertainty.values()) > 1:
                    continue
            elif quality_filter == "medium":
                uncertainty = payload.get("uncertainty", {})
                if sum(uncertainty.values()) > 2:
                    continue
            
            # Build result
            results.append({
                "case_id": payload.get("event_id"),
                "patient_hash": payload.get("patient_hash"),
                "score": round(hit.score, 4),
                "summary": payload.get("processed_text", "")[:200],
                "visits": payload.get("visit_count", 1),
                "last_visit": datetime.fromtimestamp(payload.get("timestamp", 0)).strftime("%Y-%m-%d"),
                "days_ago": (int(time.time()) - payload.get("timestamp", 0)) // 86400,
                "uncertainty": payload.get("uncertainty", {}),
                "metadata": {
                    "age": payload.get("age"),
                    "gender": payload.get("gender"),
                    "program": payload.get("program"),
                    "village": payload.get("village")
                }
            })
            
            if len(results) >= limit:
                break
        
        return {
            "success": True,
            "query": query,
            "total_results": len(results),
            "results": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "results": []
        }




@app.post("/api/chat")
async def chat_with_context(request: ChatRequest):
    """Chat with the system about a specific interaction"""
    print(f"\n[BOSS CHECK] ========== CHAT ENDPOINT CALLED ==========")
    print(f"[BOSS CHECK] Interaction ID: {request.interaction_id}")
    print(f"[BOSS CHECK] User message: {request.message}")
    
    # 1. Retrieve context
    print(f"[BOSS CHECK] Step 1: Fetching processed interaction...")
    processed = db.get_processed_interaction(request.interaction_id)
    if not processed:
        print(f"[BOSS CHECK] ERROR: No processed interaction found!")
        raise HTTPException(status_code=404, detail="Interaction context not found")
    
    print(f"[BOSS CHECK] Processed keys: {list(processed.keys())}")
    
    # 2. Build prompt context
    print(f"[BOSS CHECK] Step 2: Building context...")
    patient_data = processed.get("patient_data", {})
    print(f"[BOSS CHECK] Patient data keys: {list(patient_data.keys())}")
    
    print(f"[BOSS CHECK] Step 3: Fetching final outputs...")
    outputs = db.get_final_case_outputs(request.interaction_id)
    print(f"[BOSS CHECK] Final outputs found: {outputs is not None}")
    case_summary = outputs.get("case_summary", {}) if outputs else {}
    print(f"[BOSS CHECK] Case summary keys: {list(case_summary.keys()) if case_summary else 'EMPTY'}")
    
    print(f"[BOSS CHECK] Step 4: Fetching retrieval results...")
    similar_cases = processed.get("retrieved_cases", [])
    
    # Check if retrieval results stored separately
    if not similar_cases:
        print(f"[BOSS CHECK] No cases in processed, checking retrieval_results table...")
        retrieval_data = db.get_retrieval_results(request.interaction_id)
        similar_cases = retrieval_data.get("retrieved_cases", []) if retrieval_data else []
    
    print(f"[BOSS CHECK] Total similar cases: {len(similar_cases)}")

    print(f"[BOSS CHECK] Step 5: Building context string...")
    
    # Get additional context - DO NOT include explanation_text (it may contain "limited data" phrases)
    
    # Fetch patient state from DB
    patient_state_row = db.get_patient_state_by_interaction(request.interaction_id)
    patient_state_summary = patient_state_row.get("patient_current_state", {}) if patient_state_row else processed.get("patient_current_state", {})
    
    # Format similar cases for context
    cases_text = ""
    if similar_cases:
        cases_text = "Historical Cases from Similar Patients:\n"
        for i, case in enumerate(similar_cases[:3]):  # Show top 3
            cases_text += f"  • Case {i+1}: {case.get('summary', case.get('action_taken', 'Medical case'))} → Outcome: {case.get('outcome', 'Resolved')}\n"
    else:
        cases_text = "Historical Cases: None found in database."
    
    # Get themes
    themes = patient_state_summary.get('recurring_themes', [])
    themes_text = ', '.join(themes) if themes else 'No specific recurring themes'

    context_str = f"""
Patient Themes: {themes_text}

{cases_text}

Input Data: Image uploaded by healthcare worker.
"""
    print(f"[BOSS CHECK] Context length: {len(context_str)} chars")
    print(f"[BOSS CHECK] Context preview:\n{context_str[:300]}...")

    print(f"[BOSS CHECK] Step 6: Getting Gemini client...")
    gemini = get_gemini_client()
    if not gemini:
        print(f"[BOSS CHECK] ERROR: Gemini client unavailable!")
        raise HTTPException(status_code=500, detail="LLM service unavailable")
         
    # 3. Generate response
    print(f"[BOSS CHECK] Step 7: Generating response...")
    prompt = f"""You are helping a healthcare worker make quick decisions.
Based on the patient info and similar past cases, give simple, practical next steps.

Rules:
- Use simple language
- Give 2-3 basic recommendations (like: see a doctor, apply ointment, take rest, etc.)
- Reference the similar cases if available
- Keep it under 50 words
- NEVER say negated things or anything similar - just provide recommendations based on the context given
- NEVER use markdown formatting like ** or * for bold/italics - use plain text only
- Be direct and helpful, do not explain limitations

{context_str}

Question: {request.message}

Answer (plain text, no formatting):"""

    print(f"[BOSS CHECK] Prompt length: {len(prompt)} chars")
    response = gemini.generate(prompt)
    
    # Clean any markdown formatting that might slip through
    import re
    response = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)  # Remove **bold**
    response = re.sub(r'\*([^*]+)\*', r'\1', response)      # Remove *italic*
    response = re.sub(r'__([^_]+)__', r'\1', response)      # Remove __bold__
    response = re.sub(r'_([^_]+)_', r'\1', response)        # Remove _italic_
    
    print(f"[BOSS CHECK] Response length: {len(response)} chars")
    print(f"[BOSS CHECK] ========== CHAT COMPLETE ==========\n")
    
    return {"reply": response}


@app.get("/api/final_output/{interaction_id}")
async def get_final_output(interaction_id: str):
    """
    Get COMPLETE output for an interaction to render on frontend.
    Returns every single piece of data produced by the agents.
    """
    print(f"\n[BOSS CHECK] ========== FINAL OUTPUT ENDPOINT CALLED ==========")
    print(f"[BOSS CHECK] Interaction ID: {interaction_id}")
    
    try:
        # 1. Fetch all data components from DB
        print(f"[BOSS CHECK] Step 1: Fetching raw interaction...")
        raw = db.get_raw_interaction(interaction_id) or {}
        print(f"[BOSS CHECK] Raw data keys: {list(raw.keys()) if raw else 'EMPTY'}")
        
        print(f"[BOSS CHECK] Step 2: Fetching processed interaction...")
        processed = db.get_processed_interaction(interaction_id) or {}
        print(f"[BOSS CHECK] Processed data keys: {list(processed.keys()) if processed else 'EMPTY'}")
        
        # Merge basic info
        print(f"[BOSS CHECK] Step 3: Extracting patient_data...")
        patient_data = processed.get("patient_data", {})
        print(f"[BOSS CHECK] Patient data keys: {list(patient_data.keys()) if patient_data else 'EMPTY'}")
        
        # Patient State
        print(f"[BOSS CHECK] Step 4: Fetching patient state...")
        state_data = db.get_patient_state_by_interaction(interaction_id)
        print(f"[BOSS CHECK] State data: {state_data is not None}")
        current_state = state_data.get("patient_current_state") if state_data else processed.get("patient_current_state")
        print(f"[BOSS CHECK] Current state keys: {list(current_state.keys()) if current_state else 'EMPTY'}")
        
        # Retrieval
        print(f"[BOSS CHECK] Step 5: Fetching retrieval results...")
        retrieval_data = db.get_retrieval_results(interaction_id)
        print(f"[BOSS CHECK] Retrieval data: {retrieval_data is not None}")
        retrieved_cases = retrieval_data.get("retrieved_cases") if retrieval_data else processed.get("retrieved_cases")
        print(f"[BOSS CHECK] Retrieved cases count: {len(retrieved_cases) if retrieved_cases else 0}")
        
        print(f"[BOSS CHECK] Step 6: Fetching retrieval plan...")
        retrieval_plan = db.get_retrieval_plan(interaction_id)
        print(f"[BOSS CHECK] Retrieval plan: {retrieval_plan is not None}")
        
        # Final Recommendations
        print(f"[BOSS CHECK] Step 7: Fetching final case outputs...")
        final_outputs = db.get_final_case_outputs(interaction_id)
        print(f"[BOSS CHECK] Final outputs: {final_outputs is not None}")
        case_summary = final_outputs.get("case_summary") if final_outputs else processed.get("case_summary")
        explanation = final_outputs.get("explanation_text") if final_outputs else processed.get("explanation_text")
        followup = final_outputs.get("followup_output") if final_outputs else processed.get("followup_output")
        print(f"[BOSS CHECK] Case summary: {case_summary is not None}")
        print(f"[BOSS CHECK] Explanation length: {len(explanation) if explanation else 0}")
        print(f"[BOSS CHECK] Followup: {followup is not None}")
        
        # 2. File Paths (Construct public URLs)
        print(f"[BOSS CHECK] Step 8: Constructing file paths...")
        files = {}
        try:
            if raw.get('raw_audio_path'):
                files["audio"] = f"/uploads/{raw.get('raw_audio_path').replace(chr(92), '/').split('/')[-1]}"
                print(f"[BOSS CHECK] Audio path: {files['audio']}")
            else:
                files["audio"] = None
                
            if raw.get('raw_image_paths'):
                files["images"] = [f"/uploads/{p.replace(chr(92), '/').split('/')[-1]}" for p in raw.get('raw_image_paths', [])]
                print(f"[BOSS CHECK] Image count: {len(files['images'])}")
            else:
                files["images"] = []
                
            if raw.get('raw_document_paths'):
                files["documents"] = [f"/uploads/{p.replace(chr(92), '/').split('/')[-1]}" for p in raw.get('raw_document_paths', [])]
                print(f"[BOSS CHECK] Document count: {len(files['documents'])}")
            else:
                files["documents"] = []
        except Exception as file_err:
            print(f"[BOSS CHECK] ERROR in file path construction: {file_err}")
            files = {"audio": None, "images": [], "documents": []}
        
        # 3. Construct Mega-Response
        print(f"[BOSS CHECK] Step 9: Building final response...")
        response = {
            "interaction_id": interaction_id,
            "timestamp": raw.get("timestamp"),
            "patient_data": patient_data,
            "files": files,
            "agent_outputs": {
                "patient_state": current_state,
                "retrieval_plan": retrieval_plan.get("retrieval_plan") if retrieval_plan else None,
                "retrieved_cases": retrieved_cases,
                "case_summary": case_summary,
                "explanation_text": explanation,
                "followup_output": followup
            },
            "status": "complete"
        }
        print(f"[BOSS CHECK] ========== SUCCESS - Returning response ==========")
        return response
        
    except Exception as e:
        print(f"[BOSS CHECK] ========== CRITICAL ERROR ==========")
        print(f"[BOSS CHECK] Error type: {type(e).__name__}")
        print(f"[BOSS CHECK] Error message: {str(e)}")
        import traceback
        print(f"[BOSS CHECK] Traceback:\n{traceback.format_exc()}")
        print(f"[BOSS CHECK] ============================================")
        
        # RETURN ERROR AS JSON so frontend can show it
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e), 
                "error_type": type(e).__name__,
                "status": "failed",
                "message": "Could not assemble final output"
            }
        )



if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("Field Ingestion Agent API")
    print("=" * 60)
    print("\n📍 Open http://localhost:8000 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
