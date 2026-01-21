"""
Database Layer for Field Ingestion Agent
=========================================
SQLite database with two tables:
- raw_interactions: Store raw uploaded data
- processed_interactions: Store processed patient_data JSON
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


class IngestionDatabase:
    """
    SQLite database for Field Ingestion Agent.
    
    Tables:
        raw_interactions: Store raw input data and file paths
        processed_interactions: Store processed patient_data JSON
    """
    
    def __init__(self, db_path: str = "ingestion.db", storage_dir: str = "uploads"):
        self.db_path = db_path
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each file type
        (self.storage_dir / "audio").mkdir(exist_ok=True)
        (self.storage_dir / "images").mkdir(exist_ok=True)
        (self.storage_dir / "documents").mkdir(exist_ok=True)
        (self.storage_dir / "text").mkdir(exist_ok=True)
        (self.storage_dir / "json").mkdir(exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table 1: raw_interactions
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
            
            # Table 2: processed_interactions
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
            
            # Table 3: patient_states (NEW - for Patient Memory Agent output)
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
            
            # Table 4: retrieval_plans (NEW - for Context Builder Agent output)
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
            
            # Table 5: retrieval_results (NEW - for Similar Case Retrieval Agent output)
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
            
            # Table 6: final_case_outputs (NEW - for Explanation & Referral Agents output)
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
            
            # Indexes for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_patient ON raw_interactions(patient_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_timestamp ON raw_interactions(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_processed_patient ON processed_interactions(patient_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_patient_states ON patient_states(patient_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_retrieval_plans ON retrieval_plans(patient_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_retrieval_results ON retrieval_results(patient_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_final_outputs ON final_case_outputs(patient_hash)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # =========================================================================
    # RAW INTERACTIONS
    # =========================================================================
    
    def store_raw_interaction(
        self,
        interaction_id: str,
        patient_hash: str,
        raw_text: Optional[str] = None,
        raw_audio_path: Optional[str] = None,
        raw_image_paths: Optional[List[str]] = None,
        raw_document_paths: Optional[List[str]] = None,
        timestamp: Optional[int] = None
    ) -> str:
        """
        Store raw interaction data.
        
        Args:
            interaction_id: Unique interaction identifier
            patient_hash: Anonymized patient identifier
            raw_text: Raw text input
            raw_audio_path: Path to audio file
            raw_image_paths: List of paths to image files
            raw_document_paths: List of paths to document files
            timestamp: Unix timestamp (defaults to now)
        
        Returns:
            The interaction_id
        """
        import time
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO raw_interactions 
                (interaction_id, patient_hash, raw_text, raw_audio_path, 
                 raw_image_paths, raw_document_paths, timestamp, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                interaction_id,
                patient_hash,
                raw_text,
                raw_audio_path,
                json.dumps(raw_image_paths) if raw_image_paths else None,
                json.dumps(raw_document_paths) if raw_document_paths else None,
                timestamp or int(time.time()),
                datetime.now().isoformat()
            ))
            
            conn.commit()
        
        return interaction_id
    
    def get_raw_interaction(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """Get a raw interaction by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM raw_interactions WHERE interaction_id = ?",
                (interaction_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return {
                    "interaction_id": row["interaction_id"],
                    "patient_hash": row["patient_hash"],
                    "raw_text": row["raw_text"],
                    "raw_audio_path": row["raw_audio_path"],
                    "raw_image_paths": json.loads(row["raw_image_paths"]) if row["raw_image_paths"] else [],
                    "raw_document_paths": json.loads(row["raw_document_paths"]) if row["raw_document_paths"] else [],
                    "timestamp": row["timestamp"],
                    "created_at": row["created_at"]
                }
        
        return None
    
    # =========================================================================
    # PROCESSED INTERACTIONS
    # =========================================================================
    
    def store_processed_interaction(
        self,
        interaction_id: str,
        patient_hash: str,
        patient_data: Dict[str, Any],
        uncertainty_flags: Dict[str, bool],
        processing_timestamp: Optional[int] = None
    ) -> str:
        """
        Store processed interaction data.
        
        Args:
            interaction_id: Unique interaction identifier
            patient_hash: Anonymized patient identifier
            patient_data: Full patient_data JSON
            uncertainty_flags: Uncertainty flags dict
            processing_timestamp: When processing occurred
        
        Returns:
            The interaction_id
        """
        import time
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO processed_interactions 
                (interaction_id, patient_hash, patient_data_json, 
                 uncertainty_flags, processing_timestamp, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                interaction_id,
                patient_hash,
                json.dumps(patient_data, indent=2),
                json.dumps(uncertainty_flags),
                processing_timestamp or int(time.time()),
                datetime.now().isoformat()
            ))
            
            conn.commit()
        
        return interaction_id
    
    def get_processed_interaction(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """Get a processed interaction by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM processed_interactions WHERE interaction_id = ?",
                (interaction_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return {
                    "interaction_id": row["interaction_id"],
                    "patient_hash": row["patient_hash"],
                    "patient_data": json.loads(row["patient_data_json"]),
                    "uncertainty_flags": json.loads(row["uncertainty_flags"]),
                    "processing_timestamp": row["processing_timestamp"],
                    "created_at": row["created_at"]
                }
        
        return None
    
    def get_all_processed(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all processed interactions"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM processed_interactions ORDER BY processing_timestamp DESC LIMIT ?",
                (limit,)
            )
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "interaction_id": row["interaction_id"],
                    "patient_hash": row["patient_hash"],
                    "patient_data": json.loads(row["patient_data_json"]),
                    "uncertainty_flags": json.loads(row["uncertainty_flags"]),
                    "processing_timestamp": row["processing_timestamp"],
                    "created_at": row["created_at"]
                })
            
            return results
    
    # =========================================================================
    # PATIENT STATES (for Patient Memory Agent)
    # =========================================================================
    
    def store_patient_state(
        self,
        interaction_id: str,
        patient_hash: str,
        event_id: str,
        patient_current_state: Dict[str, Any]
    ) -> int:
        """
        Store patient current state from Patient Memory Agent.
        
        Args:
            interaction_id: Interaction ID
            patient_hash: Patient identifier
            event_id: Event ID from Qdrant
            patient_current_state: The current state dict
        
        Returns:
            The database row ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO patient_states 
                (interaction_id, patient_hash, event_id, total_visits, 
                 first_visit, latest_visit, patient_current_state_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                interaction_id,
                patient_hash,
                event_id,
                patient_current_state.get("total_visits", 0),
                patient_current_state.get("first_visit"),
                patient_current_state.get("latest_visit"),
                json.dumps(patient_current_state, indent=2),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_patient_states(self, patient_hash: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get patient state history for a specific patient"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM patient_states 
                WHERE patient_hash = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (patient_hash, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row["id"],
                    "interaction_id": row["interaction_id"],
                    "patient_hash": row["patient_hash"],
                    "event_id": row["event_id"],
                    "total_visits": row["total_visits"],
                    "first_visit": row["first_visit"],
                    "latest_visit": row["latest_visit"],
                    "patient_current_state": json.loads(row["patient_current_state_json"]),
                    "created_at": row["created_at"]
                })
            
            return results
    
    def store_retrieval_plan(
        self,
        interaction_id: str,
        patient_hash: str,
        event_id: str,
        retrieval_plan: Dict[str, Any]
    ) -> int:
        """
        Store retrieval plan from Context Builder Agent.
        
        Args:
            interaction_id: Interaction ID
            patient_hash: Patient identifier
            event_id: Event ID from Qdrant
            retrieval_plan: The retrieval plan dict
        
        Returns:
            The database row ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Extract risk flags for indexing
            risk_flags = json.dumps(retrieval_plan.get("risk_flags", []))
            
            cursor.execute("""
                INSERT INTO retrieval_plans 
                (interaction_id, patient_hash, event_id, retrieval_plan_json, risk_flags, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                interaction_id,
                patient_hash,
                event_id,
                json.dumps(retrieval_plan, indent=2),
                risk_flags,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def store_retrieval_results(
        self,
        interaction_id: str,
        patient_hash: str,
        event_id: str,
        retrieved_cases: List[Dict[str, Any]],
        retrieval_metadata: Dict[str, Any]
    ) -> int:
        """
        Store retrieval results from Similar Case Retrieval Agent.
        
        Args:
            interaction_id: Interaction ID
            patient_hash: Patient identifier  
            event_id: Event ID from Qdrant
            retrieved_cases: List of retrieved similar cases
            retrieval_metadata: Metadata about retrieval process
        
        Returns:
            The database row ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO retrieval_results 
                (interaction_id, patient_hash, event_id, retrieved_cases_json, retrieval_metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                interaction_id,
                patient_hash,
                event_id,
                json.dumps(retrieved_cases, indent=2),
                json.dumps(retrieval_metadata, indent=2),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def store_final_case_outputs(
        self,
        interaction_id: str,
        patient_hash: str,
        case_summary: Dict[str, Any],
        explanation_text: str,
        followup_output: Dict[str, Any]
    ) -> int:
        """
        Store final case outputs from Explanation & Referral Agents.
        
        Args:
            interaction_id: Interaction ID
            patient_hash: Patient identifier
            case_summary: Case summary dict
            explanation_text: Generated explanation
            followup_output: Follow-up status and nudges
        
        Returns:
            The database row ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO final_case_outputs 
                (interaction_id, patient_hash, case_summary_json, explanation_text, followup_output_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                interaction_id,
                patient_hash,
                json.dumps(case_summary, indent=2),
                explanation_text,
                json.dumps(followup_output, indent=2),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    # =========================================================================
    # FILE STORAGE HELPERS
    # =========================================================================
    
    def save_uploaded_file(
        self,
        file_content: bytes,
        filename: str,
        file_type: str,  # "audio", "images", "documents"
        interaction_id: str,
        patient_hash: str
    ) -> str:
        """
        Save an uploaded file to disk.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            file_type: Type of file (determines subdirectory)
            interaction_id: Interaction ID for naming
            patient_hash: Patient hash for naming (primary identifier)
        
        Returns:
            Full path to saved file
        """
        # Sanitize filename
        safe_name = "".join(c for c in filename if c.isalnum() or c in "._-")
        
        # Create unique filename with patient_hash first for better sorting
        unique_name = f"{patient_hash}_{interaction_id[:8]}_{safe_name}"
        
        # Determine path
        file_path = self.storage_dir / file_type / unique_name
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        return str(file_path.absolute())
    
    def save_text_input(
        self,
        text: str,
        interaction_id: str,
        patient_hash: str
    ) -> str:
        """
        Save text input to a file.
        
        Args:
            text: Raw text content
            interaction_id: Interaction ID for naming
            patient_hash: Patient hash for naming
        
        Returns:
            Full path to saved file
        """
        filename = f"{patient_hash}_{interaction_id[:8]}_notes.txt"
        file_path = self.storage_dir / "text" / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return str(file_path.absolute())
    
    def save_patient_data_json(
        self,
        patient_data: Dict[str, Any],
        interaction_id: str,
        patient_hash: str
    ) -> str:
        """
        Save patient_data JSON to a file.
        
        Args:
            patient_data: The patient_data dictionary
            interaction_id: Interaction ID for naming
            patient_hash: Patient hash for naming
        
        Returns:
            Full path to saved JSON file
        """
        filename = f"{patient_hash}_{interaction_id[:8]}_patient_data.json"
        file_path = self.storage_dir / "json" / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(patient_data, f, indent=2, ensure_ascii=False)
        
        return str(file_path.absolute())
    
    def save_patient_current_state_json(
        self,
        patient_current_state: Dict[str, Any],
        interaction_id: str,
        patient_hash: str,
        event_id: str
    ) -> str:
        """
        Save patient_current_state JSON to a file.
        
        Args:
            patient_current_state: The patient current state dictionary
            interaction_id: Interaction ID for naming
            patient_hash: Patient hash for naming
            event_id: Event ID from Qdrant
        
        Returns:
            Full path to saved JSON file
        """
        filename = f"{patient_hash}_{interaction_id[:8]}_patient_state.json"
        file_path = self.storage_dir / "json" / filename
        
        # Add metadata
        output = {
            "event_id": event_id,
            "interaction_id": interaction_id,
            "patient_hash": patient_hash,
            "patient_current_state": patient_current_state,
            "created_at": datetime.now().isoformat()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return str(file_path.absolute())
    
    def save_retrieval_plan_json(
        self,
        retrieval_plan: Dict[str, Any],
        interaction_id: str,
        patient_hash: str,
        event_id: str
    ) -> str:
        """
        Save retrieval_plan JSON to a file.
        
        Args:
            retrieval_plan: The retrieval plan dictionary
            interaction_id: Interaction ID for naming
            patient_hash: Patient hash for naming
            event_id: Event ID from Qdrant
        
        Returns:
            Full path to saved JSON file
        """
        filename = f"{patient_hash}_{interaction_id[:8]}_retrieval_plan.json"
        file_path = self.storage_dir / "json" / filename
        
        # Add metadata
        output = {
            "event_id": event_id,
            "interaction_id": interaction_id,
            "patient_hash": patient_hash,
            "retrieval_plan": retrieval_plan,
            "created_at": datetime.now().isoformat()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return str(file_path.absolute())
    
    def save_retrieved_cases_json(
        self,
        retrieved_cases: List[Dict[str, Any]],
        retrieval_metadata: Dict[str, Any],
        interaction_id: str,
        patient_hash: str,
        event_id: str
    ) -> str:
        """
        Save retrieved_cases JSON to a file.
        
        Args:
            retrieved_cases: List of retrieved similar cases
            retrieval_metadata: Metadata about retrieval
            interaction_id: Interaction ID for naming
            patient_hash: Patient hash for naming
            event_id: Event ID from Qdrant
        
        Returns:
            Full path to saved JSON file
        """
        filename = f"{patient_hash}_{interaction_id[:8]}_retrieved_cases.json"
        file_path = self.storage_dir / "json" / filename
        
        # Add metadata
        output = {
            "event_id": event_id,
            "interaction_id": interaction_id,
            "patient_hash": patient_hash,
            "retrieved_cases": retrieved_cases,
            "retrieval_metadata": retrieval_metadata,
            "created_at": datetime.now().isoformat()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return str(file_path.absolute())
    
    def save_final_recommendations_json(
        self,
        case_summary: Dict[str, Any],
        explanation_text: str,
        followup_output: Dict[str, Any],
        interaction_id: str,
        patient_hash: str
    ) -> str:
        """
        Save final recommendations (case summary + explanation + followup) to file.
        
        Args:
            case_summary: Case summary dict
            explanation_text: Generated explanation
            followup_output: Follow-up status
            interaction_id: Interaction ID for naming
            patient_hash: Patient hash for naming
        
        Returns:
            Full path to saved JSON file
        """
        filename = f"{patient_hash}_{interaction_id[:8]}_final_recommendations.json"
        file_path = self.storage_dir / "json" / filename
        
        # Build final recommendations
        output = {
            "interaction_id": interaction_id,
            "patient_hash": patient_hash,
            "case_summary": case_summary,
            "explanation": explanation_text,
            "followup_status": followup_output,
            "created_at": datetime.now().isoformat()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return str(file_path.absolute())
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Count raw interactions
            cursor.execute("SELECT COUNT(*) FROM raw_interactions")
            raw_count = cursor.fetchone()[0]
            
            # Count processed interactions
            cursor.execute("SELECT COUNT(*) FROM processed_interactions")
            processed_count = cursor.fetchone()[0]
            
            # Count by uncertainty flags
            cursor.execute("SELECT uncertainty_flags FROM processed_interactions")
            flag_counts = {
                "text_sparse": 0,
                "audio_noisy": 0,
                "image_unclear": 0,
                "document_unclear": 0,
                "history_partial": 0
            }
            
            for row in cursor.fetchall():
                flags = json.loads(row[0])
                for key, value in flags.items():
                    if value:
                        flag_counts[key] = flag_counts.get(key, 0) + 1
            
            return {
                "raw_interactions": raw_count,
                "processed_interactions": processed_count,
                "uncertainty_flag_counts": flag_counts
            }


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    import time
    import uuid
    
    print("=" * 60)
    print("Ingestion Database - Demo")
    print("=" * 60)
    
    db = IngestionDatabase(db_path="demo_ingestion.db")
    
    # Create sample interaction
    interaction_id = str(uuid.uuid4())
    patient_hash = "patient_test_123"
    
    # Store raw
    db.store_raw_interaction(
        interaction_id=interaction_id,
        patient_hash=patient_hash,
        raw_text="Sample clinical notes",
        timestamp=int(time.time())
    )
    print(f"\n✓ Stored raw interaction: {interaction_id}")
    
    # Store processed
    patient_data = {
        "patient_hash": patient_hash,
        "interaction_id": interaction_id,
        "processed_text": "Sample clinical notes",
        "uncertainty_flags": {"text_sparse": False, "history_partial": True}
    }
    
    db.store_processed_interaction(
        interaction_id=interaction_id,
        patient_hash=patient_hash,
        patient_data=patient_data,
        uncertainty_flags={"text_sparse": False, "history_partial": True}
    )
    print(f"✓ Stored processed interaction")
    
    # Get stats
    stats = db.get_stats()
    print(f"\n📊 Database Stats:")
    print(f"   Raw interactions: {stats['raw_interactions']}")
    print(f"   Processed interactions: {stats['processed_interactions']}")
    
    print("\n✅ Database layer ready!")
