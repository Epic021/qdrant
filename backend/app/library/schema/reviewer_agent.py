from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .recommender_agent import CaseAnalysis
from .retriever_agent import RetrievedCase

class HallucinationCheck(BaseModel):
    is_hallucinated: bool = False
    details: str = ""

class ReviewVerdict(BaseModel):
    approved: bool
    hallucination_check: HallucinationCheck
    feedback: str = ""
    suggested_revision: Optional[str] = None

class ReviewerInput(BaseModel):
    case_analysis: CaseAnalysis
    retrieved_cases: List[RetrievedCase]

class ReviewerOutput(BaseModel):
    verdict: ReviewVerdict
    logs: List[str]
