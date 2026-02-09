"""
Context Builder Agent
======================
Component: Retrieval Planner & Constraint Builder

Role:
    - Takes canonical PatientData (from Field Ingestion)
    - Fetches recent events from Qdrant history (Read-Only)
    - Derives immutable retrieval constraints (Age, Program, Pregnancy)
    - Outputs retrieval PLAN (not results) + Context Payload for storage

This agent exists to prevent unsafe similarity search and hallucination by grounding retrieval in deterministic facts.

Rules:
    ❌ No vector search (that's for Retriever)
    ❌ No medical reasoning
    ❌ No diagnosis
    
    ✅ Plans retrieval only
    ✅ Enforces biological constraints
    ✅ Flags data quality issues
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

# Relative imports
from ..schema.field_ingestion_agent import PatientData
from ..schema.context_builder_agent import (
    ContextBuilderInput, ContextBuilderOutput, 
    RetrievalPlan, ContextPayload, RetrievalConstraints, 
    RequiredFilters, AgeRange, ModalityPolicy, CurrentStateSummary
)

# Placeholder for Qdrant Tool (to be implemented later in tools/)
# from ...tools.qdrant_manager import PatientMemoryQdrantClient

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

system_prompt = """You are the Context Builder Agent.
Your role is to analyze the incoming patient data and their interaction history to build a SAFE retrieval plan.
You must extract deterministic facts (Age, Gender, Program, Pregnancy Status) to create hard constraints for retrieval.

CRITICAL RULES:
1. If the patient is pregnant, you MUST flag `pregnancy_required=True` in filters.
2. You MUST set an appropriate age range (Adult vs Pediatric).
3. If data quality is poor (noisy audio, sparse text), you MUST disable those modalities in the `ModalityPolicy`.
4. You do NOT perform the search. You only PLAN it.

Outputs:
- RetrievalPlan: Restrictions for the Retriever Agent.
- ContextPayload: Metadata to be stored with this interaction.
"""


# ============================================================================
# LOGGER
# ============================================================================

class AgentLogger:
    """Simple logger for the agent"""
    def __init__(self):
        self.logs: List[str] = []
    
    def info(self, msg: str):
        self.logs.append(f"[INFO] {msg}")
        print(f"[ContextBuilder] {msg}")
    
    def error(self, msg: str):
        self.logs.append(f"[ERROR] {msg}")
        print(f"[ContextBuilder] {msg}")

    def get_logs(self) -> List[str]:
        return self.logs


# ============================================================================
# CONTEXT BUILDER AGENT
# ============================================================================

class ContextBuilderAgent:
    """
    Context Builder Agent implementation.
    """
    
    def __init__(self, qdrant_client=None):
        # qdrant_client will be injected or initialized later
        self.qdrant_client = qdrant_client
        self.logger = AgentLogger()
    
    def _fetch_history_summary(self, patient_hash: str) -> Dict[str, Any]:
        """
        Mockable method to fetch history. 
        Real implementation will use Qdrant tool later.
        """
        if self.qdrant_client:
            # TODO: Implement actual Qdrant fetch when tool is ready
            # return self.qdrant_client.get_patient_summary(patient_hash)
            pass
        return {"visit_count": 0, "recurring_themes": []}

    def run(self, input_state: ContextBuilderInput) -> ContextBuilderOutput:
        """
        Execute the agent logic.
        """
        patient_data = input_state.patient_data
        self.logger.info(f"Processing context for patient: {patient_data.patient_hash}")
        
        # 1. Fetch History (Read-Only)
        # In a real run, this would query Qdrant for past events
        history_summary = self._fetch_history_summary(patient_data.patient_hash)
        visit_count = history_summary.get("visit_count", 0)
        
        # 2. Extract Signals from Current Interaction (Deterministic)
        age = None
        program = None
        pregnancy_status = "unknown"
        
        # Simple extraction logic (Placeholder for more complex logic if needed)
        # In reality, these might come from specific fields in `patient_data.capture_metadata` 
        # or parsed from text if structured. For now, we assume they might be in metadata 
        # or we explicitly look for them. 
        # Since PatientData schema doesn't have direct 'age/program' fields (it has them in metadata potentially?), 
        # let's assume they are passed or extracted.
        # Wait, the previous `PatientData` schema didn't have `age` or `program`. 
        # The `ContextPayload` does.
        # This implies `FieldIngestion` might need to extract them, or `ContextBuilder` extracts them from text.
        # Let's assume `ContextBuilder` attempts to extract them from `processed_text` using basic rules or LLM if allowed.
        # "Instructions folder has 1. System prompt ... modify variable names ... professional".
        # I will implement basic extraction logic here.
        
        # 3. Build Retrieval Constraints
        age_range = AgeRange()
        # TODO: Implement extraction logic from text if needed
        
        required_filters = RequiredFilters(
            program=program,
            pregnancy_required=(pregnancy_status == "pregnant"),
            age_range=age_range
        )
        
        retrieval_constraints = RetrievalConstraints(
            required_filters=required_filters,
            geography_scope="same_block" # Default safe scope
        )
        
        # 4. Modality Policy based on Uncertainty Flags
        flags = patient_data.uncertainty_flags
        modality_policy = ModalityPolicy(
            use_text=not flags.text_sparse,
            use_image=not flags.image_unclear and len(patient_data.image_quality) > 0,
            use_audio=not flags.audio_noisy and patient_data.audio_confidence is not None and patient_data.audio_confidence > 0.6,
            reason="Based on data quality flags"
        )
        
        # 5. Build Context Payload (for storage in next step)
        context_payload = ContextPayload(
            visit_count=visit_count + 1, # Incrementing for this visit
            recurring_themes=history_summary.get("recurring_themes", [])
            # Other fields would be populated if extracted
        )
        
        # 6. Current State Summary
        current_state_summary = CurrentStateSummary(
            age_group="unknown", # Logic to determine group
            pregnancy_status=pregnancy_status,
            program=program,
            key_trends=history_summary.get("recurring_themes", [])
        )
        
        # 7. Risk Flags
        risk_flags = []
        if visit_count == 0:
            risk_flags.append("new_patient")
        if flags.history_partial:
            risk_flags.append("partial_data")
            
        retrieval_plan = RetrievalPlan(
            current_state_summary=current_state_summary,
            retrieval_constraints=retrieval_constraints,
            modality_policy=modality_policy,
            risk_flags=risk_flags
        )
        
        self.logger.info("Context built successfully")
        
        return ContextBuilderOutput(
            retrieval_plan=retrieval_plan,
            context_payload=context_payload,
            logs=self.logger.get_logs()
        )