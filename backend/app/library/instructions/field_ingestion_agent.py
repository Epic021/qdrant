"""
Field Ingestion Agent
=====================
Component: Input Processor

Role:
    - Receive raw data (Text, Audio, Image, Documents)
    - Process and normalize data using MediaProcessor tool
    - valid quality checks (Blur, Noise, Sparse text)
    - Persist raw and processed data using DatabaseManager
    - Output structured `PatientData`

Flow:
    Raw Input -> MediaProcessor -> Quality Check -> DB Store -> PatientData
"""

import logging
import time
from typing import Dict, Any, Optional, List
import uuid

# Import Schemas
from ..schema.field_ingestion_agent import (
    FieldIngestionInput, 
    FieldIngestionOutput,
    PatientData,
    ImageQuality,
    DocumentResult,
    UncertaintyFlags,
    CaptureMetadata
)

# Import Tools
from ...tools.media_processor import (
    TextProcessor,
    AudioProcessor,
    ImageProcessor,
    DocumentProcessor,
    Config as MediaConfig
)
from ...database.db_manager import DatabaseManager

# ============================================================================
# LOGGER
# ============================================================================

class AgentLogger:
    """Simple logger for the agent"""
    
    STATUS_START = "[START]"
    STATUS_PROCESS = "[PROCESS]"
    STATUS_SUCCESS = "[SUCCESS]"
    STATUS_WARNING = "[WARNING]"
    STATUS_ERROR = "[ERROR]"
    
    def __init__(self):
        self.logs: List[str] = []
        self._callback = None
    
    def set_callback(self, callback):
        self._callback = callback
        
    def _emit(self, icon: str, message: str):
        log_entry = f"{icon} {message}"
        self.logs.append(log_entry)
        print(f"[FieldIngestion] {log_entry}")
        if self._callback:
            try:
                self._callback(log_entry)
            except Exception:
                pass

    def start(self, message: str):
        self._emit(self.STATUS_START, message)
        
    def info(self, message: str):
        self._emit(self.STATUS_PROCESS, message)
        
    def success(self, message: str):
        self._emit(self.STATUS_SUCCESS, message)
        
    def warning(self, message: str):
        self._emit(self.STATUS_WARNING, message)
        
    def error(self, message: str):
        self._emit(self.STATUS_ERROR, message)
    
    def get_logs(self) -> List[str]:
        return self.logs


# ============================================================================
# FIELD INGESTION AGENT
# ============================================================================

class FieldIngestionAgent:
    """
    Field Ingestion Agent implementation using Tools.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.logger = AgentLogger()
        
        # Tools
        self.text_processor = TextProcessor()
        self.audio_processor = AudioProcessor()
        self.image_processor = ImageProcessor()
        self.document_processor = DocumentProcessor()
        
        # Database
        self.db = db_manager or DatabaseManager() # Initialize default if not provided
        
    def process(
        self,
        patient_hash: str,
        raw_text: Optional[str] = None,
        audio_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        document_paths: Optional[List[str]] = None,
        device_id: str = "device_001",
        offline: bool = False,
        timestamp: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main processing pipeline.
        """
        interaction_id = str(uuid.uuid4())
        timestamp = timestamp or int(time.time())
        
        self.logger.start(f"Starting ingestion (ID: {interaction_id[:8]})")
        
        # 1. Store Raw Data
        try:
            self.db.store_raw_interaction(
                interaction_id=interaction_id,
                patient_hash=patient_hash,
                raw_text=raw_text,
                raw_audio_path=audio_path,
                raw_image_paths=image_paths,
                raw_document_paths=document_paths,
                timestamp=timestamp
            )
            self.logger.info("Raw data persisted to database")
        except Exception as e:
            self.logger.error(f"DB Store Failed: {e}")

        # 2. Process Text
        self.logger.info("Processing text...")
        processed_text, text_sparse = self.text_processor.process(raw_text)
        
        # 3. Process Audio
        self.logger.info("Processing audio...")
        audio_transcript, audio_lang, audio_conf, audio_noisy = self.audio_processor.process(audio_path)
        if audio_transcript:
            self.logger.success(f"Audio transcribed ({audio_lang}, conf: {audio_conf})")
        
        # 4. Process Images
        self.logger.info("Processing images...")
        image_results, image_unclear = self.image_processor.process(image_paths)
        
        # 5. Process Documents
        self.logger.info("Processing documents...")
        document_results, document_unclear = self.document_processor.process(document_paths)
        
        # 6. Aggregate
        
        # Merge text sources (Raw + Audio + Documents)
        combined_text_parts = []
        if processed_text:
            combined_text_parts.append(f"[MANUAL_INPUT] {processed_text}")
        if audio_transcript:
            combined_text_parts.append(f"[AUDIO_TRANSCRIPT] {audio_transcript}")
        for doc in document_results:
            if doc['ocr_text']:
                combined_text_parts.append(f"[DOCUMENT_OCR] {doc['ocr_text']}")
        
        final_text = "\n\n".join(combined_text_parts)
        
        # Flags
        uncertainty_flags = UncertaintyFlags(
            text_sparse=text_sparse and not audio_transcript, # Only sparse if both text and audio missing
            audio_noisy=audio_noisy,
            image_unclear=image_unclear,
            document_unclear=document_unclear,
            history_partial=offline
        )
        
        # Metadata
        capture_metadata = CaptureMetadata(
            device_id=device_id,
            offline=offline,
            timestamp=timestamp
        )
        
        # Output Schema
        patient_data = PatientData(
            patient_hash=patient_hash,
            interaction_id=interaction_id,
            processed_text=final_text,
            audio_transcript=audio_transcript,
            audio_language=audio_lang,
            audio_confidence=audio_conf,
            image_quality=[ImageQuality(**img) for img in image_results],
            documents=[DocumentResult(**doc) for doc in document_results],
            uncertainty_flags=uncertainty_flags,
            capture_metadata=capture_metadata
        )
        
        # 7. Persist Processed Data
        try:
            self.db.store_processed_interaction(
                interaction_id=interaction_id,
                patient_hash=patient_hash,
                patient_data=patient_data.model_dump(),
                uncertainty_flags=uncertainty_flags.model_dump()
            )
            self.logger.success("Processed data saved to DB")
        except Exception as e:
            self.logger.error(f"Failed to save processed data: {e}")
            
        self.logger.success("Ingestion complete")
        
        # Return dict for internal use, logs separate
        return {
            "patient_data": patient_data,
            "logs": self.logger.get_logs()
        }

    def run(self, input_state: FieldIngestionInput) -> FieldIngestionOutput:
        """
        LangGraph entry point.
        """
        result = self.process(
            patient_hash=input_state.patient_hash,
            raw_text=input_state.raw_text,
            audio_path=input_state.audio_path,
            image_paths=input_state.image_paths,
            document_paths=input_state.document_paths,
            device_id=input_state.device_id,
            timestamp=input_state.timestamp
        )
        
        return FieldIngestionOutput(
            patient_data=result["patient_data"],
            logs=result["logs"]
        )