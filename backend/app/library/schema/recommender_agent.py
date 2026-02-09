from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .patient_memory_agent import PatientCurrentState
from .retriever_agent import RetrievedCase

class ReferralSignal(BaseModel):
    referral_needed: bool = False
    urgency: str = "routine" # routine, urgent, emergency
    reason: str = ""

class FollowUpSignal(BaseModel):
    follow_up_needed: bool = False
    days_recommended: int = 0
    reason: str = ""

class CaseAnalysis(BaseModel):
    summary: str
    explanation: str # The LLM generated text
    referral_signal: ReferralSignal
    follow_up_signal: FollowUpSignal
    safety_flags: List[str] = Field(default_factory=list)

class RecommenderInput(BaseModel):
    patient_current_state: PatientCurrentState
    retrieved_cases: List[RetrievedCase]

class RecommenderOutput(BaseModel):
    analysis: CaseAnalysis
    logs: List[str]
