"""
Payload Schema (DB + Qdrant)
============================
Defines persistent data structures for SQLite and Qdrant.
These are unrelated to LangGraph state, but are used for storage.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# ============================================================================
# QDRANT PAYLOADS
# ============================================================================

class EventPayload(BaseModel):
    """
    Payload stored in Qdrant context_events collection.
    Associated with a specific event_id.
    """
    patient_hash: str
    interaction_id: str
    timestamp: int
    
    # Context (from ContextBuilder)
    age_group: Optional[str] = None
    program: Optional[str] = None
    pregnancy_status: Optional[str] = None
    village: Optional[str] = None
    block: Optional[str] = None
    
    # Data Quality Flags (from FieldIngestion)
    text_sparse: bool = False
    audio_noisy: bool = False
    image_unclear: bool = False
    
    # Content Snippets (for quick display without DB fetch)
    processed_text_snippet: Optional[str] = None

# ============================================================================
# SQLite TABLE MODELS (Pydantic mirrors)
# ============================================================================

class RawInteraction(BaseModel):
    """Mirror of 'raw_interactions' table"""
    interaction_id: str
    patient_hash: str
    raw_text: Optional[str] = None
    raw_audio_path: Optional[str] = None
    raw_image_paths: Optional[str] = None # JSON string list
    raw_document_paths: Optional[str] = None # JSON string list
    timestamp: int
    created_at: str

class ProcessedInteraction(BaseModel):
    """Mirror of 'processed_interactions' table"""
    interaction_id: str
    patient_hash: str
    patient_data_json: str # Full canonical JSON
    uncertainty_flags: str # JSON string
    processing_timestamp: int
    created_at: str

class PatientState(BaseModel):
    """Mirror of 'patient_states' table"""
    patient_hash: str
    total_visits: int
    last_visit_timestamp: int
    current_state_summary: str # JSON
    updated_at: str

class RetrievalLog(BaseModel):
    """Mirror of 'retrieval_logs' table (plan + results)"""
    interaction_id: str
    retrieval_plan_json: str
    retrieved_cases_json: str
    metadata_json: str
    created_at: str

class FinalCaseOutput(BaseModel):
    """Mirror of 'final_case_outputs' table"""
    interaction_id: str
    case_summary: str
    explanation: str
    referral_signal: str # JSON
    follow_up_signal: str # JSON
    review_verdict: str # JSON
    created_at: str
