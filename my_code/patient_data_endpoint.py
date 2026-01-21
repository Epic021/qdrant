"""
API endpoint to get all patient data
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import json
from pathlib import Path

router = APIRouter()

@router.get("/api/patient/{patient_hash}/all")
async def get_patient_all_data(patient_hash: str):
    """
    Get all data for a patient across all agents.
    
    Returns:
        - All JSON files for this patient
        - Database records
        - Qdrant history
    """
    try:
        from db import IngestionDatabase
        from qdrant_manager import PatientMemoryQdrantClient
        
        db = IngestionDatabase()
        qdrant = PatientMemoryQdrantClient()
        
        # Get all JSON files for this patient
        json_dir = Path("uploads/json")
        patient_files = list(json_dir.glob(f"{patient_hash}_*.json"))
        
        files_data = {}
        for file_path in patient_files:
            with open(file_path, 'r') as f:
                file_data = json.load(f)
                files_data[file_path.stem] = file_data
        
        # Get database records
        processed = db.get_all_processed(limit=100)
        patient_processed = [p for p in processed if p["patient_hash"] == patient_hash]
        
        # Get patient states
        patient_states = db.get_patient_states(patient_hash, limit=10)
        
        # Get Qdrant history
        qdrant_history = qdrant.get_patient_history(patient_hash)
        
        return {
            "patient_hash": patient_hash,
            "total_interactions": len(patient_processed),
            "files": files_data,
            "database": {
                "processed_interactions": patient_processed,
                "patient_states": patient_states
            },
            "qdrant_history": qdrant_history
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
