"""
Context Builder Agent
======================
Component: Retrieval Planner & Constraint Builder

Role:
    - Takes patient current state
    - Fetches recent events from Qdrant
    - Derives retrieval constraints
    - Outputs retrieval PLAN (not results)

This agent exists to prevent unsafe similarity search.

Rules:
    ❌ No vector search
    ❌ No medical reasoning
    ❌ No diagnosis
    ❌ No confidence scores
    
    ✅ Plans retrieval only
    ✅ Enforces biological constraints
    ✅ Flags data quality issues
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import Qdrant client
from qdrant_manager import PatientMemoryQdrantClient

# Load environment variables
load_dotenv()


# ============================================================================
# CONTEXT BUILDER AGENT
# ============================================================================

class ContextBuilderAgent:
    """
    Context Builder Agent - Retrieval Planner with Safety Constraints.
    
    This agent:
    1. Fetches recent patient events from Qdrant (payload query only)
    2. Extracts deterministic signals (demographics, program, quality)
    3. Optionally normalizes with Gemini (constrained)
    4. Outputs retrieval plan with constraints
    
    It does NOT perform similarity search.
    """
    
    def __init__(self, max_history_events: int = 10):
        """
        Initialize Context Builder Agent.
        
        Args:
            max_history_events: Maximum number of recent events to fetch
        """
        self.max_history_events = max_history_events
        self.qdrant_client = None
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
    def _ensure_qdrant_client(self):
        """Lazy initialize Qdrant client"""
        if self.qdrant_client is None:
            self.qdrant_client = PatientMemoryQdrantClient()
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing function for Context Builder Agent.
        
        Args:
            state: LangGraph state containing patient_current_state
        
        Returns:
            LangGraph state with retrieval_plan
        """
        print("\n" + "=" * 70)
        print("CONTEXT BUILDER AGENT - Retrieval Planner")
        print("=" * 70)
        
        # Ensure Qdrant client
        self._ensure_qdrant_client()
        
        # Extract inputs
        patient_hash = state.get("patient_hash")
        patient_current_state = state.get("patient_current_state", {})
        
        if not patient_hash:
            print("⚠ No patient_hash provided")
            return {"retrieval_plan": self._empty_plan()}
        
        print(f"Patient Hash: {patient_hash}")
        print(f"Current Total Visits: {patient_current_state.get('total_visits', 0)}")
        
        # ===== STEP 1: Fetch Recent Events =====
        print("\n--- Step 1: Fetching Recent Events ---")
        recent_events = self._fetch_recent_events(patient_hash)
        print(f"✓ Retrieved {len(recent_events)} event(s)")
        
        # ===== STEP 2: Extract Deterministic Signals =====
        print("\n--- Step 2: Extracting Signals ---")
        signals = self._extract_signals(recent_events, patient_current_state)
        print(f"✓ Extracted signals:")
        print(f"  - Age Group: {signals['age_group']}")
        print(f"  - Pregnancy Status: {signals['pregnancy_status']}")
        print(f"  - Program: {signals['program']}")
        print(f"  - Data Quality Issues: {len(signals['quality_flags'])}")
        
        # ===== STEP 3: Optional LLM Normalization =====
        print("\n--- Step 3: Optional Normalization ---")
        normalized = self._optional_normalization(signals, recent_events)
        print(f"✓ Normalization complete")
        
        # ===== STEP 4: Build Retrieval Plan =====
        print("\n--- Step 4: Building Retrieval Plan ---")
        retrieval_plan = self._build_retrieval_plan(signals, normalized)
        print(f"✓ Retrieval plan created")
        print(f"  - Required filters: {len(retrieval_plan['retrieval_constraints']['required_filters'])}")
        print(f"  - Risk flags: {len(retrieval_plan['risk_flags'])}")
        
        print("\n" + "=" * 70)
        print("✓ Context Builder completed")
        print("=" * 70 + "\n")
        
        return {
            "retrieval_plan": retrieval_plan
        }
    
    def _fetch_recent_events(self, patient_hash: str) -> List[Dict[str, Any]]:
        """
        Fetch recent events for patient (payload query only, no vector search).
        
        Args:
            patient_hash: Patient identifier
        
        Returns:
            List of recent events, sorted chronologically
        """
        # Use existing get_patient_history method
        # This is a payload-only query filtered by patient_hash
        events = self.qdrant_client.get_patient_history(
            patient_hash=patient_hash
        )
        
        # Limit to max_history_events
        if len(events) > self.max_history_events:
            events = events[:self.max_history_events]
        
        # Sort chronologically (oldest to newest for timeline)
        events_sorted = sorted(events, key=lambda e: e.get("timestamp", 0))
        
        return events_sorted
    
    def _extract_signals(
        self, 
        events: List[Dict[str, Any]], 
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract deterministic signals from events (NO LLM, pure code).
        
        Args:
            events: List of patient events
            current_state: Patient current state
        
        Returns:
            Dictionary of extracted signals
        """
        signals = {
            "age_group": "unknown",
            "age_raw": None,
            "pregnancy_status": "unknown",
            "program": None,
            "quality_flags": [],
            "temporal_signals": {},
            "visit_count": len(events),
            "recurring_themes": current_state.get("recurring_themes", [])
        }
        
        if not events:
            signals["quality_flags"].append("insufficient_history")
            return signals
        
        # Extract from most recent event
        latest_event = events[-1] if events else {}
        
        # Demographics
        age = latest_event.get("age")
        if age:
            signals["age_raw"] = age
            if age < 18:
                signals["age_group"] = "pediatric"
            else:
                signals["age_group"] = "adult"
        
        # Pregnancy status
        pregnancy = latest_event.get("pregnancy_status")
        if pregnancy:
            signals["pregnancy_status"] = pregnancy
        else:
            signals["pregnancy_status"] = "none"
        
        # Program
        program = latest_event.get("program")
        if program:
            signals["program"] = program
        
        # Data quality from uncertainty summary
        uncertainty_summary = current_state.get("uncertainty_summary", {})
        
        if uncertainty_summary.get("text_sparse", 0) > 0:
            signals["quality_flags"].append("text_sparse")
        if uncertainty_summary.get("image_unclear", 0) > 0:
            signals["quality_flags"].append("image_unclear")
        if uncertainty_summary.get("audio_noisy", 0) > 0:
            signals["quality_flags"].append("audio_noisy")
        if uncertainty_summary.get("history_partial", 0) > 0:
            signals["quality_flags"].append("history_partial")
        
        # Temporal signals
        if len(events) >= 2:
            first_visit = events[0].get("timestamp", 0)
            latest_visit = events[-1].get("timestamp", 0)
            
            if first_visit and latest_visit:
                days_span = (latest_visit - first_visit) / 86400  # Convert to days
                signals["temporal_signals"]["days_since_first_visit"] = days_span
        
        return signals
    
    def _optional_normalization(
        self, 
        signals: Dict[str, Any], 
        events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optional LLM normalization (CONSTRAINED).
        
        This is optional and must not override hard constraints.
        Used only for:
        - Normalizing messy free-text
        - Collapsing repeated symptoms
        - Identifying non-medical patterns
        
        Args:
            signals: Extracted signals
            events: Raw events
        
        Returns:
            Normalized data (key_trends, etc.)
        """
        # For now, skip LLM and use simple keyword extraction
        # TODO: Add Gemini normalization if needed
        
        normalized = {
            "key_trends": signals.get("recurring_themes", [])[:3],  # Top 3
            "visit_pattern": "single_visit" if signals["visit_count"] == 1 else "repeat_visitor"
        }
        
        return normalized
    
    def _build_retrieval_plan(
        self, 
        signals: Dict[str, Any], 
        normalized: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build the retrieval plan with constraints.
        
        This is the main output - a structured plan for retrieval.
        
        Args:
            signals: Extracted signals
            normalized: Normalized data
        
        Returns:
            Retrieval plan dictionary
        """
        # Current state summary
        current_state_summary = {
            "age_group": signals["age_group"],
            "pregnancy_status": signals["pregnancy_status"],
            "program": signals.get("program"),
            "key_trends": normalized.get("key_trends", [])
        }
        
        # Retrieval constraints
        required_filters = {}
        
        # Program filter (if present)
        if signals.get("program"):
            required_filters["program"] = signals["program"]
        
        # Pregnancy constraint
        pregnancy_required = signals["pregnancy_status"] not in ["none", "unknown"]
        required_filters["pregnancy_required"] = pregnancy_required
        
        # Age range constraint
        age_range = {"gte": None, "lte": None}
        if signals["age_raw"]:
            # Allow ±10 years for adults, ±2 for pediatric
            if signals["age_group"] == "pediatric":
                age_range["gte"] = max(0, signals["age_raw"] - 2)
                age_range["lte"] = signals["age_raw"] + 2
            else:
                age_range["gte"] = max(18, signals["age_raw"] - 10)
                age_range["lte"] = signals["age_raw"] + 10
        
        required_filters["age_range"] = age_range
        
        # Geography scope (default to same block for safety)
        geography_scope = "same_block"
        
        retrieval_constraints = {
            "required_filters": required_filters,
            "geography_scope": geography_scope
        }
        
        # Modality policy
        modality_policy = self._determine_modality_policy(signals)
        
        # Risk flags  
        risk_flags = self._determine_risk_flags(signals)
        
        # Assemble final plan
        retrieval_plan = {
            "current_state_summary": current_state_summary,
            "retrieval_constraints": retrieval_constraints,
            "modality_policy": modality_policy,
            "risk_flags": risk_flags
        }
        
        return retrieval_plan
    
    def _determine_modality_policy(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine which modalities to use based on data quality.
        
        Rules:
        - Text is always used
        - Image disabled if image_unclear flag
        - Audio disabled if audio_noisy flag
        
        Args:
            signals: Extracted signals
        
        Returns:
            Modality policy dict
        """
        quality_flags = signals.get("quality_flags", [])
        
        use_image = "image_unclear" not in quality_flags
        use_audio = "audio_noisy" not in quality_flags
        
        reasons = []
        if not use_image:
            reasons.append("image_unclear")
        if not use_audio:
            reasons.append("audio_noisy")
        if not reasons:
            reasons.append("all_modalities_clear")
        
        return {
            "use_text": True,  # Always true
            "use_image": use_image,
            "use_audio": use_audio,
            "reason": ", ".join(reasons)
        }
    
    def _determine_risk_flags(self, signals: Dict[str, Any]) -> List[str]:
        """
        Determine risk flags based on data quality and patient history.
        
        Args:
            signals: Extracted signals
        
        Returns:
            List of risk flags
        """
        risk_flags = []
        
        # Insufficient history
        if signals["visit_count"] == 0:
            risk_flags.append("insufficient_history")
        
        # Low data quality
        quality_flags = signals.get("quality_flags", [])
        if len(quality_flags) >= 2:
            risk_flags.append("low_data_quality")
        
        # High dropout risk (only 1 visit and it's been a while)
        if signals["visit_count"] == 1:
            risk_flags.append("single_visit_only")
        
        return risk_flags
    
    def _empty_plan(self) -> Dict[str, Any]:
        """Return empty retrieval plan for error cases"""
        return {
            "current_state_summary": {
                "age_group": "unknown",
                "pregnancy_status": "unknown",
                "program": None,
                "key_trends": []
            },
            "retrieval_constraints": {
                "required_filters": {
                    "program": None,
                    "pregnancy_required": False,
                    "age_range": {"gte": None, "lte": None}
                },
                "geography_scope": "same_block"
            },
            "modality_policy": {
                "use_text": True,
                "use_image": False,
                "use_audio": False,
                "reason": "no_patient_data"
            },
            "risk_flags": ["insufficient_history"]
        }


# ============================================================================
# LANGGRAPH-COMPATIBLE ENTRY POINT
# ============================================================================

def context_builder_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph-compatible entry point for Context Builder Agent.
    
    Input state:
        - patient_hash: str
        - patient_current_state: Dict
    
    Output state:
        - retrieval_plan: Dict
    
    Usage in LangGraph:
        from my_code.context_builder_agent import context_builder_agent
        
        graph.add_node("context_builder", context_builder_agent)
    """
    agent = ContextBuilderAgent()
    return agent.process(state)


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CONTEXT BUILDER AGENT - DEMO")
    print("=" * 70 + "\n")
    
    # Sample input (from Patient Memory Agent output)
    sample_state = {
        "event_id": "654ddf20-0308-411a-9540-cecb47efd0e2",
        "interaction_id": "6bc433a4-568b-412e-a40d-46fdb8e87020",
        "patient_hash": "naman_jain",
        "patient_current_state": {
            "patient_hash": "naman_jain",
            "total_visits": 1,
            "first_visit": 1769017430,
            "latest_visit": 1769017430,
            "latest_event_id": "654ddf20-0308-411a-9540-cecb47efd0e2",
            "visit_history": [
                {
                    "event_id": "654ddf20-0308-411a-9540-cecb47efd0e2",
                    "interaction_id": "6bc433a4-568b-412e-a40d-46fdb8e87020",
                    "timestamp": 1769017430,
                    "summary": "Anger causing irritation in the skin",
                    "uncertainty": {
                        "text_sparse": False,
                        "audio_noisy": False,
                        "image_unclear": False,
                        "document_unclear": False,
                        "history_partial": False
                    }
                }
            ],
            "recurring_themes": ["skin"],
            "uncertainty_summary": {
                "text_sparse": 0,
                "audio_noisy": 0,
                "image_unclear": 0,
                "document_unclear": 0,
                "history_partial": 0
            }
        }
    }
    
    # Process through agent
    agent = ContextBuilderAgent()
    result = agent.process(sample_state)
    
    print("\n" + "=" * 70)
    print("RETRIEVAL PLAN OUTPUT")
    print("=" * 70)
    print(json.dumps(result["retrieval_plan"], indent=2))
    
    print("\n✅ Context Builder Agent ready!")
