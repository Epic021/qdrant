from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

# Import all agent schemas
from .field_ingestion_agent import PatientData
from .patient_memory_agent import PatientCurrentState, ContextPayload
from .context_builder_agent import RetrievalPlan
from .retriever_agent import RetrievedCase, RetrievalMetadata
from .recommender_agent import CaseAnalysis
from .reviewer_agent import ReviewVerdict

class GraphState(BaseModel):
    # Inputs
    initial_input: Dict[str, Any] = Field(default_factory=dict)
    
    # Outputs from agents
    patient_data: Optional[PatientData] = None
    context_payload: Optional[ContextPayload] = None
    patient_current_state: Optional[PatientCurrentState] = None
    retrieval_plan: Optional[RetrievalPlan] = None
    retrieved_cases: List[RetrievedCase] = Field(default_factory=list)
    retrieval_metadata: Optional[RetrievalMetadata] = None
    case_analysis: Optional[CaseAnalysis] = None
    review_verdict: Optional[ReviewVerdict] = None
    
    # Orchestration State
    current_node: str = "field_ingestion"
    history_log: List[str] = Field(default_factory=list)
    error: Optional[str] = None
