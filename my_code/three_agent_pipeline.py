"""
3-Agent Pipeline Demo
=====================
Chains together:
1. Field Ingestion Agent
2. Patient Memory Agent
3. Context Builder Agent

This demonstrates the full pipeline up to retrieval planning.
"""

import json
from typing import Dict, Any

from agents import field_ingestion_agent
from patient_memory_agent import patient_memory_agent
from context_builder_agent import context_builder_agent


def run_three_agent_pipeline(
    patient_hash: str,
    raw_text: str = None,
    audio_path: str = None,
    image_paths: list = None,
    document_paths: list = None
) -> Dict[str, Any]:
    """
    Run all three agents in sequence.
    
    Pipeline:
        Field Ingestion → Patient Memory → Context Builder
    
    Returns:
        Complete state with retrieval_plan
    """
    print("\n" + "=" * 80)
    print("3-AGENT PIPELINE - Field Ingestion + Patient Memory + Context Builder")
    print("=" * 80)
    
    all_logs = []
    
    # ========== AGENT 1: FIELD INGESTION ==========
    print("\n┌─ AGENT 1: Field Ingestion Agent")
    print("│")
    
    state_1 = {
        "patient_hash": patient_hash,
        "raw_text": raw_text,
        "audio_path": audio_path,
        "image_paths": image_paths or [],
        "document_paths": document_paths or []
    }
    
    result_1 = field_ingestion_agent(state_1)
    patient_data = result_1.get("patient_data")
    
    for log in result_1.get("logs", []):
        all_logs.append(f"[Agent 1] {log}")
    
    print("└─ ✓ Field Ingestion completed")
    print()
    
    # ========== AGENT 2: PATIENT MEMORY ==========
    print("┌─ AGENT 2: Patient Memory Agent")
    print("│")
    
    state_2 = {
        "patient_data": patient_data
    }
    
    result_2 = patient_memory_agent(state_2)
    
    for log in result_2.get("logs", []):
        all_logs.append(f"[Agent 2] {log}")
    
    print("└─ ✓ Patient Memory completed")
    print()
    
    # ========== AGENT 3: CONTEXT BUILDER ==========
    print("┌─ AGENT 3: Context Builder Agent")
    print("│")
    
    state_3 = {
        "patient_hash": result_2.get("patient_hash"),
        "event_id": result_2.get("event_id"),
        "patient_current_state": result_2.get("patient_current_state")
    }
    
    result_3 = context_builder_agent(state_3)
    retrieval_plan = result_3.get("retrieval_plan")
    
    print("└─ ✓ Context Builder completed")
    
    # ========== FINAL STATE ==========
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    
    final_state = {
        # From Agent 1
        "patient_data": patient_data,
        
        # From Agent 2
        "event_id": result_2.get("event_id"),
        "patient_hash": result_2.get("patient_hash"),
        "context_memory_written": result_2.get("context_memory_written"),
        "patient_current_state": result_2.get("patient_current_state"),
        
        # From Agent 3
        "retrieval_plan": retrieval_plan,
        
        # Combined logs
        "all_logs": all_logs
    }
    
    return final_state


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("\nDEMO: Running 3-agent pipeline with sample data\n")
    
    # Sample input
    result = run_three_agent_pipeline(
        patient_hash="demo_patient_context",
        raw_text="Patient has skin rash on arms. Itching reported. No fever.",
        image_paths=[],
        document_paths=[]
    )
    
    # Print summary
    print("\n📊 SUMMARY")
    print("-" * 80)
    print(f"Patient Hash:       {result['patient_hash']}")
    print(f"Event ID:           {result['event_id']}")
    print(f"Memory Written:     {result['context_memory_written']}")
    print(f"Total Visits:       {result['patient_current_state'].get('total_visits', 0)}")
    print(f"\n🎯 RETRIEVAL PLAN:")
    print(json.dumps(result['retrieval_plan'], indent=2))
    print("-" * 80)
    
    print("\n✅ 3-agent pipeline ready!")
    print("\n💡 TIP: The retrieval plan constrains what similarity search can do")
