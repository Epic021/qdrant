from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .field_ingestion_agent import PatientData, UncertaintyFlags
from .context_builder_agent import ContextPayload

class VisitSummary(BaseModel):
    event_id: Optional[str] = None
    interaction_id: Optional[str] = None
    timestamp: Optional[int] = None
    summary: str 
    uncertainty: UncertaintyFlags

class PatientCurrentState(BaseModel):
    patient_hash: str
    total_visits: int
    first_visit: Optional[int] = None
    latest_visit: Optional[int] = None
    latest_event_id: Optional[str] = None
    visit_history: List[VisitSummary] = Field(default_factory=list)
    recurring_themes: List[str] = Field(default_factory=list)
    uncertainty_summary: Dict[str, int] = Field(default_factory=dict)

class PatientMemoryInput(BaseModel):
    patient_data: PatientData
    context_payload: ContextPayload

class PatientMemoryOutput(BaseModel):
    event_id: str
    patient_hash: str
    context_memory_written: bool
    patient_current_state: PatientCurrentState
    logs: List[str]
