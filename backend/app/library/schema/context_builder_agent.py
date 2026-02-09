from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .field_ingestion_agent import PatientData

class AgeRange(BaseModel):
    gte: Optional[int] = None
    lte: Optional[int] = None

class RequiredFilters(BaseModel):
    program: Optional[str] = None
    pregnancy_required: bool = False
    age_range: AgeRange = Field(default_factory=AgeRange)

class RetrievalConstraints(BaseModel):
    required_filters: RequiredFilters = Field(default_factory=RequiredFilters)
    geography_scope: str = "same_block"

class ModalityPolicy(BaseModel):
    use_text: bool = True
    use_image: bool = False
    use_audio: bool = False
    reason: str = ""

class CurrentStateSummary(BaseModel):
    age_group: str = "unknown"
    pregnancy_status: str = "unknown"
    program: Optional[str] = None
    key_trends: List[str] = Field(default_factory=list)

class RetrievalPlan(BaseModel):
    current_state_summary: CurrentStateSummary
    retrieval_constraints: RetrievalConstraints
    modality_policy: ModalityPolicy
    risk_flags: List[str] = Field(default_factory=list)

class ContextPayload(BaseModel):
    """Metadata to be stored in Qdrant/DB relative to this interaction"""
    # Extracted from PatientData for easier indexing
    age: Optional[int] = None
    gender: Optional[str] = None
    pregnancy_status: Optional[str] = None
    program: Optional[str] = None
    village: Optional[str] = None
    block: Optional[str] = None
    
    # Computed context
    visit_count: int = 0
    recurring_themes: List[str] = Field(default_factory=list)

class ContextBuilderInput(BaseModel):
    patient_data: PatientData

class ContextBuilderOutput(BaseModel):
    retrieval_plan: RetrievalPlan
    context_payload: ContextPayload
    logs: List[str]
