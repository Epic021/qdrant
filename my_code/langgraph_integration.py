"""
LangGraph Integration - Field Ingestion + Patient Memory
=========================================================
Chains together:
1. Field Ingestion Agent
2. Patient Memory Agent

This demonstrates the full pipeline with both agents' logs.
"""

import sys
import json
from typing import Dict, Any
from pathlib import Path

# Import both agents
from my_code.agents import field_ingestion_agent
from my_code.patient_memory_agent import patient_memory_agent


def run_dual_agent_pipeline(
    patient_hash: str,
    raw_text: str = None,
    audio_path: str = None,
    image_paths: list = None,
    document_paths: list = None
) -> Dict[str, Any]:
    """
    Run both agents in sequence and collect all logs.
    
    This simulates the LangGraph workflow:
        Field Ingestion Agent → Patient Memory Agent
    
    Args:
        patient_hash: Patient identifier
        raw_text: Free-text notes
        audio_path: Path to audio file
        image_paths: List of image paths
        document_paths: List of document paths
    
    Returns:
        Complete state with logs from both agents
    """
    print("\n" + "=" * 80)
    print("DUAL AGENT PIPELINE - Field Ingestion + Patient Memory")
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
    logs_1 = result_1.get("logs", [])
    
    # Print Field Ingestion logs
    for log in logs_1:
        print(f"│ {log}")
        all_logs.append(f"[Agent 1] {log}")
    
    print("│")
    print("└─ ✓ Field Ingestion completed")
    print()
    
    # ========== AGENT 2: PATIENT MEMORY ==========
    print("┌─ AGENT 2: Patient Memory Agent")
    print("│")
    
    state_2 = {
        "patient_data": patient_data
    }
    
    result_2 = patient_memory_agent(state_2)
    logs_2 = result_2.get("logs", [])
    
    # Print Patient Memory logs
    for log in logs_2:
        print(f"│ {log}")
        all_logs.append(f"[Agent 2] {log}")
    
    print("│")
    print("└─ ✓ Patient Memory completed")
    
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
        
        # Combined logs
        "all_logs": all_logs,
        "agent_1_logs": logs_1,
        "agent_2_logs": logs_2
    }
    
    return final_state


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("\nDEMO: Running dual agent pipeline with sample data\n")
    
    # Sample input
    result = run_dual_agent_pipeline(
        patient_hash="demo_patient_xyz",
        raw_text="Patient has skin rash on arms. Itching reported. No fever.",
        image_paths=[],  # Add actual paths if available
        document_paths=[]
    )
    
    # Print summary
    print("\n📊 SUMMARY")
    print("-" * 80)
    print(f"Patient Hash:       {result['patient_hash']}")
    print(f"Event ID:           {result['event_id']}")
    print(f"Memory Written:     {result['context_memory_written']}")
    print(f"Total Visits:       {result['patient_current_state'].get('total_visits', 0)}")
    print(f"Total Logs:         {len(result['all_logs'])}")
    print("-" * 80)
    
    print("\n✅ Dual agent pipeline ready!")
    print("\n💡 TIP: Use these logs in your frontend to show real-time progress")
