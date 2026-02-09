"""
Reviewer Agent
==============
Component: Safety & Hallucination Check

Role:
    - Receive CaseAnalysis (Explanation) + RetrievedCases (Evidence)
    - Verify that the explanation is grounded in the evidence
    - Check for forbidden content (Diagnosis, Advice) using GeminiClient
    - Output Verdict (Approved/Rejected)

Rules:
    ✅ If explanation cites facts not in retrieved cases -> REJECT (Hallucination)
    ✅ If explanation gives medical advice -> REJECT (Safety)
    ✅ If explanation is safe and grounded -> APPROVE
"""

import logging
from typing import List, Dict, Any, Optional

# Import Schemas
from ..schema.reviewer_agent import (
    ReviewerInput,
    ReviewerOutput,
    ReviewVerdict,
    HallucinationCheck
)

# Import Tools
from ...tools.llm_client import GeminiClient

# ============================================================================
# LOGGER
# ============================================================================

class AgentLogger:
    """Simple logger for the agent"""
    def __init__(self):
        self.logs: List[str] = []
    
    def info(self, msg: str):
        self.logs.append(f"[INFO] {msg}")
        print(f"[Reviewer] {msg}")
        
    def get_logs(self) -> List[str]:
        return self.logs


# ============================================================================
# REVIEWER AGENT
# ============================================================================

class ReviewerAgent:
    """
    Reviewer Agent implementation using Tools.
    """
    
    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.llm = llm_client or GeminiClient()
        self.logger = AgentLogger()
        
    def _verify_with_llm(self, explanation: str, cases: List[Any]) -> ReviewVerdict:
        """
        Verify explanation using LLM.
        """
        evidence_text = "\n".join([
            f"- Case {c.case_id}: {c.summary} (Outcome: {c.outcome})"
            for c in cases
        ])
        
        self.logger.info("Verifying explanation with Gemini...")
        
        # 1. Safety Check (Keywords)
        forbidden = ["diagnose", "prescription", "should take", "must take"]
        is_unsafe = any(w in explanation.lower() for w in forbidden)
        
        if is_unsafe:
            return ReviewVerdict(
                approved=False,
                hallucination_check=HallucinationCheck(is_hallucinated=False),
                feedback="Explanation contains forbidden medical advice/diagnosis keywords.",
                suggested_revision="Please remove any direct medical advice."
            )
        
        # 2. Hallucination Check (LLM)
        # Using the specific verification method in LLM client if available, or generic generate
        verification = self.llm.verify_hallucination(explanation, evidence_text)
        
        if verification.get("is_hallucinated"):
            return ReviewVerdict(
                approved=False,
                hallucination_check=HallucinationCheck(
                    is_hallucinated=True, 
                    details=verification.get("details", "Unsupported claims detected.")
                ),
                feedback="Explanation contains facts not supported by evidence.",
                suggested_revision="Stick strictly to the retrieved cases."
            )

        return ReviewVerdict(
            approved=True,
            hallucination_check=HallucinationCheck(is_hallucinated=False, details="Grounded in evidence."),
            feedback="Explanation is safe and grounded."
        )

    def run(self, input_state: ReviewerInput) -> ReviewerOutput:
        """
        Execute the agent logic.
        """
        explanation = input_state.case_analysis.explanation
        cases = input_state.retrieved_cases
        
        self.logger.info("Starting Review...")
        
        verdict = self._verify_with_llm(explanation, cases)
        
        if verdict.approved:
            self.logger.info("Review Passed.")
        else:
            self.logger.info(f"Review Failed: {verdict.feedback}")
        
        return ReviewerOutput(
            verdict=verdict,
            logs=self.logger.get_logs()
        )