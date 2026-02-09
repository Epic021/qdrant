"""
Database Manager Tool
=====================
SQLite database wrapper for persistent storage.
Handles raw interactions, processed data, and agent outputs.
"""

import sqlite3
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from contextlib import contextmanager

logger = logging.getLogger("db_manager")

class DatabaseManager:
    """
    SQLite database for persistent storage.
    """
    
    def __init__(self, db_path: str = "ingestion.db", storage_dir: str = "uploads"):
        self.db_path = db_path
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ["audio", "images", "documents", "text", "json"]:
            (self.storage_dir / subdir).mkdir(exist_ok=True)
        
        self._init_database()
        logger.info(f"Database initialized at {self.db_path}")
    
    def _init_database(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Raw Interactions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_interactions (
                    interaction_id TEXT PRIMARY KEY,
                    patient_hash TEXT NOT NULL,
                    raw_text TEXT,
                    raw_audio_path TEXT,
                    raw_image_paths TEXT,
                    raw_document_paths TEXT,
                    timestamp INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # 2. Processed Interactions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_interactions (
                    interaction_id TEXT PRIMARY KEY,
                    patient_hash TEXT NOT NULL,
                    patient_data_json TEXT NOT NULL,
                    uncertainty_flags TEXT NOT NULL,
                    processing_timestamp INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (interaction_id) REFERENCES raw_interactions(interaction_id)
                )
            """)
            
            # 3. Patient States
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patient_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id TEXT NOT NULL,
                    patient_hash TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    total_visits INTEGER NOT NULL,
                    first_visit INTEGER,
                    latest_visit INTEGER,
                    patient_current_state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (interaction_id) REFERENCES processed_interactions(interaction_id)
                )
            """)
            
            # 4. Retrieval Plans
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retrieval_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id TEXT NOT NULL,
                    patient_hash TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    retrieval_plan_json TEXT NOT NULL,
                    risk_flags TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (interaction_id) REFERENCES processed_interactions(interaction_id)
                )
            """)
            
            # 5. Retrieval Results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retrieval_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id TEXT NOT NULL,
                    patient_hash TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    retrieved_cases_json TEXT NOT NULL,
                    retrieval_metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (interaction_id) REFERENCES processed_interactions(interaction_id)
                )
            """)
            
            # 6. Final Outputs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS final_case_outputs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id TEXT NOT NULL,
                    patient_hash TEXT NOT NULL,
                    case_summary_json TEXT NOT NULL,
                    explanation_text TEXT NOT NULL,
                    followup_output_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (interaction_id) REFERENCES processed_interactions(interaction_id)
                )
            """)
            
            # Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_patient ON raw_interactions(patient_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_patient ON processed_interactions(patient_hash)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # =========================================================================
    # GENERIC STORE METHODS
    # =========================================================================

    def store_raw_interaction(self, interaction_id: str, patient_hash: str, **kwargs) -> str:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO raw_interactions 
                (interaction_id, patient_hash, raw_text, raw_audio_path, raw_image_paths, raw_document_paths, timestamp, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                interaction_id, patient_hash,
                kwargs.get("raw_text"), kwargs.get("raw_audio_path"),
                json.dumps(kwargs.get("raw_image_paths")) if kwargs.get("raw_image_paths") else None,
                json.dumps(kwargs.get("raw_document_paths")) if kwargs.get("raw_document_paths") else None,
                kwargs.get("timestamp", int(datetime.now().timestamp())),
                datetime.now().isoformat()
            ))
            conn.commit()
        return interaction_id

    def store_processed_interaction(self, interaction_id: str, patient_hash: str, patient_data: dict, uncertainty_flags: dict) -> str:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO processed_interactions 
                (interaction_id, patient_hash, patient_data_json, uncertainty_flags, processing_timestamp, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                interaction_id, patient_hash,
                json.dumps(patient_data), json.dumps(uncertainty_flags),
                int(datetime.now().timestamp()), datetime.now().isoformat()
            ))
            conn.commit()
        return interaction_id

    def store_patient_state(self, interaction_id: str, patient_hash: str, event_id: str, state: dict) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO patient_states (interaction_id, patient_hash, event_id, total_visits, first_visit, latest_visit, patient_current_state_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                interaction_id, patient_hash, event_id,
                state.get("total_visits", 0), state.get("first_visit"), state.get("latest_visit"),
                json.dumps(state), datetime.now().isoformat()
            ))
            conn.commit()
            return cursor.lastrowid
            
    def get_patient_state(self, patient_hash: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM patient_states WHERE patient_hash = ? ORDER BY created_at DESC LIMIT 1", (patient_hash,))
            row = cursor.fetchone()
            if row:
                return json.loads(row["patient_current_state_json"])
        return None
