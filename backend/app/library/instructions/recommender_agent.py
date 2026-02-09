"""
Recommender Agent
=================
Component: Synthesis & Recommendation

Role:
    - Receive Patient State + Retrieved Cases
    - Generate Deterministic Case Analysis
    - Generate Natural Language Explanation using GeminiClient
    - Check for Referral/Follow-up signals (deterministic)
    - Output `CaseAnalysis`

Flow:
    Input -> Case Summary -> LLM Explanation -> Signal Check -> Output
"""

import logging
from typing import List, Dict, Any, Optional

# Import Schemas
from ..schema.recommender_agent import (
    RecommenderInput,
    RecommenderOutput,
    CaseAnalysis,
    ReferralSignal,
    FollowUpSignal
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
        print(f"[Recommender] {msg}")
        
    def get_logs(self) -> List[str]:
        return self.logs


# ============================================================================
# RECOMMENDER AGENT
# ============================================================================

class RecommenderAgent:
    """
    Recommender Agent implementation using Tools.
    """
    
    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.llm = llm_client or GeminiClient()
        self.logger = AgentLogger()
        
    def _generate_explanation(self, summary: str, cases: List[Any]) -> str:
        """
        Generate explanation using LLM tool.
        """
        if not cases:
            return "No similar cases found to support a recommendation."
            
        case_summaries = "\n".join([f"- Case {c.case_id}: {c.summary} (Outcome: {c.outcome})" for c in cases])
        
        prompt = f"""
        You are a medical assistant support tool.
        
        PATIENT CONTEXT:
        {summary}
        
        SIMILAR CASES:
        {case_summaries}
        
        TASK:
        Explain the relevance of these similar cases to the current patient.
        
        RULES:
        - NO diagnosis
        - NO medical advice
        - Use counts (e.g., "3 similar cases...")
        - Be concise (3-4 sentences)
        """
        
        self.logger.info("Generating explanation with Gemini...")
        explanation = self.llm.generate(prompt)
        
        if not explanation:
            self.logger.info("LLM generation failed, using fallback.")
            return f"Found {len(cases)} similar cases. Please review the case history for details."
            
        return explanation

    def run(self, input_state: RecommenderInput) -> RecommenderOutput:
        """
        Execute the agent logic.
        """
        patient_state = input_state.patient_current_state
        cases = input_state.retrieved_cases
        
        self.logger.info(f"Analyzing for patient {patient_state.patient_hash}")
        
        # 1. Deterministic Analysis
        state_summary = f"Visits: {patient_state.total_visits}, Themes: {patient_state.recurring_themes}"
        
        # 2. LLM Explanation
        explanation = self._generate_explanation(state_summary, cases)
        
        # 3. Signals (Deterministic)
        referral_count = sum(1 for c in cases if c.outcome == "referred")
        referral_needed = referral_count > 0
        
        referral_signal = ReferralSignal(
            referral_needed=referral_needed,
            urgency="routine" if referral_needed else "none",
            reason=f"{referral_count} similar cases resulted in referral."
        )
        
        follow_up = FollowUpSignal(
            follow_up_needed=True,  # Default for safety
            days_recommended=7,
            reason="Routine follow-up based on standard protocol."
        )
        
        analysis = CaseAnalysis(
            summary=state_summary,
            explanation=explanation,
            referral_signal=referral_signal,
            follow_up_signal=follow_up
        )
        
        return RecommenderOutput(
            analysis=analysis,
            logs=self.logger.get_logs()
        )