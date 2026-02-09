from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .context_builder_agent import RetrievalPlan
from .patient_memory_agent import PatientCurrentState

class RetrievedCase(BaseModel):
    case_id: str
    patient_hash: str
    similarity_score: float
    summary: str
    outcome: Optional[str] = None
    demographics: Dict[str, Any] = Field(default_factory=dict)
    relevance_reason: str = ""

class RetrievalMetadata(BaseModel):
    total_candidates: int
    filtered_count: int
    final_count: int
    search_strategy: str

class RetrieverInput(BaseModel):
    retrieval_plan: RetrievalPlan
    patient_current_state: PatientCurrentState

class RetrieverOutput(BaseModel):
    retrieved_cases: List[RetrievedCase]
    retrieval_metadata: RetrievalMetadata
    logs: List[str]
