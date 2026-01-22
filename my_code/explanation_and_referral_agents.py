"""
Explanation & Trust Agent + Referral & Follow-up Agent
=======================================================
Final stage agents for pipeline completion.

Agent 5: Explanation & Trust Agent (Gemini LLM, constrained)
Agent 6: Referral & Follow-up Agent (Deterministic)

These agents:
- Build case summary (deterministic)
- Generate honest explanations (LLM-based)
- Detect follow-up gaps (deterministic)
- Display results on frontend

Rules:
    ❌ No diagnosis
    ❌ No medical advice
    ❌ No confidence percentages
    ❌ No inventing facts
    
    ✅ Cite uncertainty
    ✅ Show evidence counts
    ✅ Make gaps visible
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment
load_dotenv()


# ============================================================================
# CASE SUMMARY BUILDER (Deterministic)
# ============================================================================

class CaseSummaryBuilder:
    """
    Builds canonical case summary from all prior agent outputs.
    Pure code, no LLM.
    """
    
    @staticmethod
    def build(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build case summary from all agent outputs.
        
        Args:
            state: Complete LangGraph state with all agent outputs
        
        Returns:
            Canonical case summary dict
        """
        patient_data = state.get("patient_data", {})
        patient_state = state.get("patient_current_state", {})
        retrieval_plan = state.get("retrieval_plan", {})
        retrieved_cases = state.get("retrieved_cases", [])
        retrieval_metadata = state.get("retrieval_metadata", {})
        
        # Patient snapshot
        patient_snapshot = {
            "age": patient_data.get("age"),
            "gender": patient_data.get("gender"),
            "pregnancy_status": patient_data.get("pregnancy_status", "unknown"),
            "program": patient_data.get("program")
        }
        
        # Key signals from patient state
        key_signals = []
        
        # Visit frequency
        total_visits = patient_state.get("total_visits", 0)
        if total_visits > 1:
            key_signals.append(f"Patient has {total_visits} recorded visit(s)")
        else:
            key_signals.append("First visit for this patient")
        
        # Recurring themes
        themes = patient_state.get("recurring_themes", [])
        if themes:
            key_signals.append(f"Recurring concerns: {', '.join(themes[:3])}")
        
        # Data quality flags
        uncertainty_summary = patient_state.get("uncertainty_summary", {})
        data_quality = {
            "image_unclear": uncertainty_summary.get("image_unclear", 0) > 0,
            "audio_noisy": uncertainty_summary.get("audio_noisy", 0) > 0,
            "text_sparse": uncertainty_summary.get("text_sparse", 0) > 0,
            "history_partial": uncertainty_summary.get("history_partial", 0) > 0
        }
        
        # Retrieval summary
        retrieval_summary = {
            "similar_cases_found": len(retrieved_cases),
            "geography": retrieval_plan.get("retrieval_constraints", {}).get("geography_scope", "unknown"),
            "program_match": retrieval_plan.get("current_state_summary", {}).get("program") is not None,
            "dense_candidates": retrieval_metadata.get("dense_candidates", 0),
            "final_selected": retrieval_metadata.get("final_selected", 0)
        }
        
        # Outcome pattern from retrieved cases
        outcome_counts = {}
        for case in retrieved_cases:
            outcome = case.get("outcome", "unknown")
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        
        outcome_pattern = {
            "total_cases": len(retrieved_cases),
            "outcomes": outcome_counts
        }
        
        return {
            "patient_snapshot": patient_snapshot,
            "key_signals": key_signals,
            "data_quality": data_quality,
            "retrieval_summary": retrieval_summary,
            "outcome_pattern": outcome_pattern
        }


# ============================================================================
# EXPLANATION & TRUST AGENT (Gemini LLM)
# ============================================================================

class ExplanationAndTrustAgent:
    """
    Generates human-readable explanation using Gemini LLM.
    
    Constraints:
    - NO diagnosis
    - NO medical advice
    - Must cite uncertainty
    - Must show evidence counts
    """
    
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    def _log(self, message: str, prefix: str = "[Explanation]"):
        """Log message"""
        formatted = f"{prefix} {message}"
        print(formatted)
        if self.log_callback:
            self.log_callback(formatted)
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanation from case summary and retrieved cases.
        
        Args:
            state: LangGraph state with all outputs
        
        Returns:
            explanation_text: Human-readable explanation
        """
        self._log("Starting Explanation & Trust Agent")
        
        # Get inputs
        case_summary = state.get("case_summary", {})
        retrieved_cases = state.get("retrieved_cases", [])
        
        if not self.gemini_api_key:
            self._log("⚠ No Gemini API key, generating basic explanation", "[Warning]")
            return {"explanation_text": self._generate_basic_explanation(case_summary, retrieved_cases)}
        
        # Generate explanation using Gemini
        self._log("Generating explanation with Gemini...")
        
        try:
            explanation = self._generate_gemini_explanation(case_summary, retrieved_cases)
            self._log("✓ Explanation generated")
            return {"explanation_text": explanation}
            
        except Exception as e:
            self._log(f"✗ Gemini error: {e}", "[Error]")
            # Fallback to basic explanation
            return {"explanation_text": self._generate_basic_explanation(case_summary, retrieved_cases)}
    
    def _generate_gemini_explanation(
        self, 
        case_summary: Dict[str, Any], 
        retrieved_cases: List[Dict[str, Any]]
    ) -> str:
        """Generate explanation using Gemini API"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.gemini_api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Build constrained prompt
            prompt = self._build_explanation_prompt(case_summary, retrieved_cases)
            
            response = model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            raise Exception(f"Gemini API error: {e}")
    
    def _build_explanation_prompt(
        self, 
        case_summary: Dict[str, Any], 
        retrieved_cases: List[Dict[str, Any]]
    ) -> str:
        """Build constrained prompt for Gemini"""
        
        prompt = f"""You are explaining healthcare system findings to a human user.

STRICT RULES:
- You MUST NOT give medical diagnoses
- You MUST NOT give medical advice
- You MUST use counts, not percentages
- Keep language simple and honest
- NEVER use markdown formatting like ** or * for bold/italics - use plain text only
- NEVER say "I can't see the image" or similar negative statements - just provide analysis based on available context
- Be direct and helpful

CASE SUMMARY:
{json.dumps(case_summary, indent=2)}

SIMILAR CASES FOUND:
{len(retrieved_cases)} similar cases were retrieved.

Outcomes:
{json.dumps(case_summary.get('outcome_pattern', {}), indent=2)}

TASK:
Write a 3-4 sentence explanation for a healthcare worker that:
1. States what similar cases were found (with counts)
2. Briefly mentions key patterns
3. Does NOT diagnose or advise
4. Maintains a helpful, professional tone
5. Uses PLAIN TEXT only - no markdown or special formatting

Your explanation (plain text, no formatting):"""

        return prompt
    
    def _generate_basic_explanation(
        self, 
        case_summary: Dict[str, Any], 
        retrieved_cases: List[Dict[str, Any]]
    ) -> str:
        """Fallback basic explanation"""
        
        num_cases = len(retrieved_cases)
        data_quality = case_summary.get("data_quality", {})
        
        limitations = []
        if data_quality.get("image_unclear"):
            limitations.append("image quality is limited")
        if data_quality.get("audio_noisy"):
            limitations.append("audio quality is limited")
        if data_quality.get("text_sparse"):
            limitations.append("text information is sparse")
        
        explanation = f"Analysis identified {num_cases} similar verified cases from the same geography. "
        explanation += "These cases provide historical context to support your clinical decision-making. "
        
        if limitations:
             explanation += f"Please note: {', '.join(limitations)}. "
        
        explanation += "Review the full case details below for specific outcomes."
        return explanation


# ============================================================================
# REFERRAL & FOLLOW-UP AGENT (Deterministic)
# ============================================================================

class ReferralAndFollowupAgent:
    """
    Detects follow-up gaps and referral delays.
    Pure deterministic logic, no LLM.
    """
    
    # Thresholds (days)
    REFERRAL_PENDING_THRESHOLD = 14
    HIGH_RISK_THRESHOLD = 7
    
    def __init__(self, log_callback=None):
        self.log_callback = log_callback
    
    def _log(self, message: str, prefix: str = "[Followup]"):
        """Log message"""
        formatted = f"{prefix} {message}"
        print(formatted)
        if self.log_callback:
            self.log_callback(formatted)
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect follow-up gaps and generate nudges.
        
        Args:
            state: LangGraph state
        
        Returns:
            followup_output: Status and nudges
        """
        self._log("Starting Referral & Follow-up Agent")
        
        patient_data = state.get("patient_data", {})
        patient_state = state.get("patient_current_state", {})
        
        # Get timestamps
        latest_visit = patient_state.get("latest_visit")
        current_time = int(datetime.now().timestamp())
        
        days_since_visit = 0
        if latest_visit:
            days_since_visit = (current_time - latest_visit) // 86400
        
        self._log(f"Days since last visit: {days_since_visit}")
        
        # Detect gaps
        followup_status = "up_to_date"
        nudges = []
        escalation_flag = False
        
        # Check for pending referrals (from uncertainty flags or patient history)
        uncertainty = patient_data.get("uncertainty_flags", {})
        
        if days_since_visit > self.REFERRAL_PENDING_THRESHOLD:
            followup_status = "overdue"
            nudges.append(f"Patient has not visited in {days_since_visit} days")
            escalation_flag = True
        elif days_since_visit > self.HIGH_RISK_THRESHOLD:
            followup_status = "pending"
            nudges.append(f"Follow-up pending for {days_since_visit} days")
        
        # Check uncertainty flags
        if uncertainty.get("document_unclear") or uncertainty.get("history_partial"):
            nudges.append("Incomplete records - consider verifying patient history")
        
        # Default nudge if everything is current
        if not nudges:
            nudges.append("No immediate follow-up needed based on timeline")
        
        self._log(f"✓ Follow-up status: {followup_status}")
        if escalation_flag:
            self._log("⚠ Escalation flag raised", "[Warning]")
        
        return {
            "followup_output": {
                "followup_status": followup_status,
                "days_since_last_visit": days_since_visit,
                "nudges": nudges,
                "escalation_flag": escalation_flag
            }
        }


# ============================================================================
# COMBINED AGENT (LangGraph Entry Point)
# ============================================================================

def explanation_and_followup_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph-compatible entry point for final agents.
    
    Combines:
    1. Case summary building (deterministic)
    2. Explanation generation (Gemini)
    3. Follow-up detection (deterministic)
    
    Input state:
        - patient_data
        - patient_current_state
        - retrieval_plan
        - retrieved_cases
        - retrieval_metadata
    
    Output state:
        - case_summary
        - explanation_text
        - followup_output
    """
    print("\n" + "=" * 70)
    print("FINAL AGENTS - Explanation & Follow-up")
    print("=" * 70)
    
    # Step 1: Build case summary (deterministic)
    print("\n[CaseSummary] Building case summary...")
    case_summary = CaseSummaryBuilder.build(state)
    print(f"[CaseSummary] ✓ Case summary built")
    
    # Add to state
    state["case_summary"] = case_summary
    
    # Step 2: Generate explanation (Gemini)
    explanation_agent = ExplanationAndTrustAgent()
    result_explanation = explanation_agent.process(state)
    
    # Step 3: Generate follow-up signals (deterministic)
    followup_agent = ReferralAndFollowupAgent()
    result_followup = followup_agent.process(state)
    
    print("\n" + "=" * 70)
    print("✓ Final agents completed")
    print("=" * 70 + "\n")
    
    return {
        "case_summary": case_summary,
        "explanation_text": result_explanation.get("explanation_text"),
        "followup_output": result_followup.get("followup_output")
    }


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EXPLANATION & FOLLOW-UP AGENTS - DEMO")
    print("=" * 70 + "\n")
    
    # Sample state (from all previous agents)
    sample_state = {
        "patient_data": {
            "patient_hash": "demo_patient",
            "age": 24,
            "gender": "female",
            "pregnancy_status": "second_trimester",
            "program": "MCH_Anemia",
            "uncertainty_flags": {
                "image_unclear": True,
                "history_partial": False
            }
        },
        "patient_current_state": {
            "total_visits": 3,
            "latest_visit": int(datetime.now().timestamp()) - (10 * 86400),  # 10 days ago
            "recurring_themes": ["anemia", "fatigue"],
            "uncertainty_summary": {
                "image_unclear": 1,
                "audio_noisy": 0,
                "text_sparse": 0,
                "history_partial": 0
            }
        },
        "retrieval_plan": {
            "current_state_summary": {
                "program": "MCH_Anemia"
            },
            "retrieval_constraints": {
                "geography_scope": "same_block"
            }
        },
        "retrieved_cases": [
            {"case_id": "case1", "outcome": "improved", "final_score": 0.85},
            {"case_id": "case2", "outcome": "improved", "final_score": 0.78},
            {"case_id": "case3", "outcome": "stable", "final_score": 0.65}
        ],
        "retrieval_metadata": {
            "dense_candidates": 15,
            "final_selected": 3
        }
    }
    
    # Run agents
    result = explanation_and_followup_agent(sample_state)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("\n📊 CASE SUMMARY:")
    print(json.dumps(result["case_summary"], indent=2))
    print("\n💬 EXPLANATION:")
    print(result["explanation_text"])
    print("\n📅 FOLLOW-UP STATUS:")
    print(json.dumps(result["followup_output"], indent=2))
    
    print("\n✅ Agents ready!")
