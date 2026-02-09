"""
Patient Memory Agent
====================
Component: Long-term Memory

Role:
    - Receive processed `PatientData` and `ContextPayload`
    - Generate multimodal embeddings (Text, Audio, Image) using MultiModalEmbedder
    - Persist event to Qdrant (Vector DB) using QdrantManager
    - Persist event to SQL (Relational DB) using DatabaseManager
    - Retrieve patient history and generate `PatientCurrentState`

Flow:
    PatientData -> Embeddings -> Qdrant Store -> SQL Store -> History Retrieval -> State Summary
"""

import logging
import uuid
import json
import time
from typing import Dict, Any, List, Optional

# Import Schemas
from ..schema.patient_memory_agent import (
    PatientMemoryInput,
    PatientMemoryOutput,
    PatientCurrentState,
    VisitSummary
)
from ..schema.payload import EventPayload

# Import Tools
from ...tools.embeddings import MultiModalEmbedder
from ...tools.qdrant_manager import QdrantManager
from ...database.db_manager import DatabaseManager

# ============================================================================
# LOGGER
# ============================================================================

class AgentLogger:
    """Simple logger for the agent"""
    def __init__(self):
        self.logs: List[str] = []
    
    def info(self, msg: str):
        self.logs.append(f"[INFO] {msg}")
        print(f"[PatientMemory] {msg}")
    
    def success(self, msg: str):
        self.logs.append(f"[SUCCESS] {msg}")
        print(f"[PatientMemory] {msg}")

    def error(self, msg: str):
        self.logs.append(f"[ERROR] {msg}")
        print(f"[PatientMemory] {msg}")
        
    def get_logs(self) -> List[str]:
        return self.logs


# ============================================================================
# PATIENT MEMORY AGENT
# ============================================================================

class PatientMemoryAgent:
    """
    Patient Memory Agent implementation using Tools.
    """
    
    def __init__(self, 
                 qdrant_manager: Optional[QdrantManager] = None,
                 db_manager: Optional[DatabaseManager] = None,
                 embedder: Optional[MultiModalEmbedder] = None):
        
        self.logger = AgentLogger()
        
        # Initialize tools if not provided
        self.qdrant = qdrant_manager or QdrantManager()
        self.db = db_manager or DatabaseManager()
        self.embedder = embedder or MultiModalEmbedder()
        
        # Ensure collection exists
        if self.qdrant.client:
            self.qdrant.create_collection(recreate=False)

    def _generate_embeddings(self, patient_data) -> Dict[str, List[float]]:
        """Generate embeddings for available modalities"""
        embeddings = {}
        
        # Text Embedding
        if patient_data.processed_text:
            emb = self.embedder.embed_text(patient_data.processed_text)
            if emb:
                embeddings["text"] = emb
        
        # Image Embedding (Average of all images)
        if patient_data.image_quality:
            valid_paths = [img.path for img in patient_data.image_quality if img.score > 0.4]
            # Use the first valid image for now, or average if tool supports it (it does)
            # But the tool expects a path. Let's iterate.
            # actually embedder.embed_image takes a path.
            # MultiModalEmbedder has embed_images for list.
            if valid_paths:
                 # Check if embedder has embed_images, otherwise loop
                 if hasattr(self.embedder, 'embed_images'):
                     emb = self.embedder.embed_images(valid_paths)
                 else:
                     emb = self.embedder.embed_image(valid_paths[0])
                     
                 if emb:
                     embeddings["image"] = emb

        # Audio Embedding
        # The tool expects a path. We don't have the path in PatientData directly?
        # partial_data.capture_metadata doesn't have it.
        # Wait, PatientData has audio_transcript but not the path?
        # Ah, field_ingestion output schema doesn't keep the path, only the content/metadata.
        # But for embedding we need the file.
        # In field_ingestion_agent, we processed audio but didn't store the path in PatientData?
        # Let's check PatientData schema.
        # It has `audio_transcript`.
        # If we want to embed audio, we need the file path.
        # Ideally, PatientData should have it or we rely on text embedding of the transcript.
        # For now, let's skip audio embedding if path is missing, relying on transcript.
        
        return embeddings

    def _generate_patient_state(self, patient_hash: str) -> PatientCurrentState:
        """
        Retrieve history and generate summary.
        """
        # Get history from Qdrant
        history = self.qdrant.get_patient_history(patient_hash)
        
        total_visits = len(history)
        latest_visit = history[-1].get("timestamp") if history else None
        
        # Simple theme extraction (mock logic for now as we act on metadata)
        # In a real system, we'd analyze the text of past visits
        recurring_themes = []
        if total_visits > 1:
            recurring_themes.append("Follow-up")
            
        # Aggregated uncertainty
        uncertainty_summary = {
            "image_unclear": sum(1 for h in history if h.get("image_unclear")),
            "audio_noisy": sum(1 for h in history if h.get("audio_noisy")),
            "text_sparse": sum(1 for h in history if h.get("text_sparse"))
        }

        # Visit Summaries
        visit_summaries = []
        for h in history[-5:]: # Last 5
            # Build UncertaintyFlags from history payload
            from ..schema.field_ingestion_agent import UncertaintyFlags
            uncertainty = UncertaintyFlags(
                text_sparse=h.get("text_sparse", False),
                audio_noisy=h.get("audio_noisy", False),
                image_unclear=h.get("image_unclear", False)
            )
            visit_summaries.append(VisitSummary(
                interaction_id=h.get("interaction_id", "unknown"),
                timestamp=h.get("timestamp", 0),
                summary=h.get("processed_text_snippet", "No details"),
                uncertainty=uncertainty
            ))

        return PatientCurrentState(
            patient_hash=patient_hash,
            total_visits=total_visits,
            latest_visit=latest_visit,
            recurring_themes=recurring_themes,
            uncertainty_summary=uncertainty_summary,
            visit_history=visit_summaries
        )

    def run(self, input_state: PatientMemoryInput) -> PatientMemoryOutput:
        """
        Execute the agent logic.
        """
        patient_data = input_state.patient_data
        context_payload = input_state.context_payload
        
        event_id = str(uuid.uuid4())
        self.logger.info(f"Processing event {event_id} for patient {patient_data.patient_hash}")
        
        # 1. Generate Embeddings
        self.logger.info("Generating embeddings...")
        embeddings = self._generate_embeddings(patient_data)
        self.logger.success(f"Generated embeddings: {list(embeddings.keys())}")
        
        # 2. Prepare Payload (mixing PatientData and ContextPayload)
        # Create EventPayload object first to validate
        payload_obj = EventPayload(
            patient_hash=patient_data.patient_hash,
            interaction_id=patient_data.interaction_id,
            timestamp=patient_data.capture_metadata.timestamp,
            
            # Context - ContextPayload has direct fields, not nested
            age_group=None,  # Not available in current schema
            program=context_payload.program,
            pregnancy_status=context_payload.pregnancy_status,
            village=context_payload.village,
            block=context_payload.block,
            
            # Flags
            text_sparse=patient_data.uncertainty_flags.text_sparse,
            audio_noisy=patient_data.uncertainty_flags.audio_noisy,
            image_unclear=patient_data.uncertainty_flags.image_unclear,
            
            # Snippet
            processed_text_snippet=patient_data.processed_text[:200] if patient_data.processed_text else ""
        )
        
        # 3. Upsert to Qdrant
        success_qdrant = self.qdrant.upsert_event(
            event_id=event_id,
            payload=payload_obj.model_dump(),
            text_vector=embeddings.get("text"),
            image_vector=embeddings.get("image"),
            audio_vector=embeddings.get("audio")
        )
        
        if success_qdrant:
            self.logger.success("Event upserted to Qdrant")
        else:
            self.logger.error("Failed to upsert to Qdrant")

        # 4. Generate Patient State
        patient_current_state = self._generate_patient_state(patient_data.patient_hash)
        
        # 5. Persist State to SQLite
        try:
            self.db.store_patient_state(
                interaction_id=patient_data.interaction_id,
                patient_hash=patient_data.patient_hash,
                event_id=event_id,
                state=patient_current_state.model_dump()
            )
            self.logger.success("Patient state saved to DB")
            success_db = True
        except Exception as e:
            self.logger.error(f"Failed to save state to DB: {e}")
            success_db = False

        return PatientMemoryOutput(
            event_id=event_id,
            patient_hash=patient_data.patient_hash,
            context_memory_written=(success_qdrant and success_db),
            patient_current_state=patient_current_state,
            logs=self.logger.get_logs()
        )