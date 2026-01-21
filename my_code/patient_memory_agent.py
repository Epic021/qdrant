"""
Patient Memory Agent
====================
Core agent for Context Memory (Qdrant) operations.

Responsibilities:
1. Receive patient_data JSON from Field Ingestion Agent
2. Generate multimodal embeddings (text, image, audio)
3. Upsert event to Qdrant with rich payload
4. Retrieve patient's complete history (filtered by patient_hash)
5. Generate "Patient Current State" summary
6. Return LangGraph-compatible state output

This agent writes to Context Memory. It does no medical reasoning.
"""

import uuid
import time
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

# Import our modules
from qdrant_manager import PatientMemoryQdrantClient
from embeddings import MultiModalEmbedder


# ============================================================================
# LOGGER (Frontend-compatible)
# ============================================================================

class PatientMemoryLogger:
    """
    Logger for Patient Memory Agent that outputs structured logs.
    Compatible with frontend display (mirrors Field Ingestion Agent style).
    """
    
    STATUS_SUCCESS = "✓"
    STATUS_WARNING = "⚠"
    STATUS_PROGRESS = "⏳"
    STATUS_ERROR = "✗"
    STATUS_INFO = "ℹ"
    
    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        self.logs: List[str] = []
        self.callback = callback
    
    def _emit(self, status: str, message: str):
        """Emit a log entry"""
        formatted = f"[{status}] {message}"
        self.logs.append(formatted)
        print(formatted)  # Console output
        
        if self.callback:
            self.callback(formatted)
    
    def success(self, message: str):
        self._emit(self.STATUS_SUCCESS, message)
    
    def warning(self, message: str):
        self._emit(self.STATUS_WARNING, message)
    
    def progress(self, message: str):
        self._emit(self.STATUS_PROGRESS, message)
    
    def error(self, message: str):
        self._emit(self.STATUS_ERROR, message)
    
    def info(self, message: str):
        self._emit(self.STATUS_INFO, message)
    
    def get_logs(self) -> List[str]:
        return self.logs.copy()


# ============================================================================
# PATIENT MEMORY AGENT
# ============================================================================

class PatientMemoryAgent:
    """
    Patient Memory Agent - writes events to Qdrant and retrieves history.
    """
    
    def __init__(self, log_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize Patient Memory Agent.
        
        Args:
            log_callback: Optional callback for frontend log streaming
        """
        self.logger = PatientMemoryLogger(callback=log_callback)
        self.qdrant_client = None
        self.embedder = None
    
    def _ensure_clients(self):
        """Lazy initialize Qdrant and embedding clients"""
        if self.qdrant_client is None:
            self.logger.progress("Connecting to Qdrant...")
            self.qdrant_client = PatientMemoryQdrantClient()
            self.logger.success("Connected to Qdrant")
            
            # Ensure collection exists
            self.qdrant_client.create_collection(recreate=False)
        
        if self.embedder is None:
            self.logger.progress("Initializing embedding generators...")
            self.embedder = MultiModalEmbedder()
            self.logger.success("Embedding generators ready")
    
    def process(
        self,
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main processing function for Patient Memory Agent.
        
        Args:
            patient_data: Output from Field Ingestion Agent
        
        Returns:
            LangGraph state dict with:
                - event_id: str
                - patient_hash: str
                - context_memory_written: bool
                - patient_current_state: Dict
                - logs: List[str]
        """
        self.logger.info("=" * 50)
        self.logger.info("Starting Patient Memory Agent")
        self.logger.info("=" * 50)
        
        # Ensure clients are initialized
        self._ensure_clients()
        
        # Extract key fields from patient_data
        patient_hash = patient_data.get("patient_hash", "unknown")
        interaction_id = patient_data.get("interaction_id", str(uuid.uuid4()))
        
        self.logger.info(f"Patient Hash: {patient_hash}")
        self.logger.info(f"Interaction ID: {interaction_id}")
        
        # ===== STEP 1: Generate Event ID =====
        event_id = str(uuid.uuid4())
        self.logger.success(f"Generated Event ID: {event_id}")
        
        # ===== STEP 2: Build Modality Inputs =====
        self.logger.info("--- Step 1: Processing Modality Inputs ---")
        
        # Text: concatenate processed_text + audio_transcript
        text_summary = self._build_text_summary(patient_data)
        
        # Images: get paths from image_quality
        image_paths = self._extract_image_paths(patient_data)
        
        # Audio: get audio path if present (NOT transcript)
        audio_path = patient_data.get("audio_path")  # Raw audio file
        
        # ===== STEP 3: Generate Embeddings =====
        self.logger.info("--- Step 2: Generating Embeddings ---")
        
        text_vector = None
        image_vector = None
        audio_vector = None
        
        # Text embedding (mandatory if text exists)
        if text_summary:
            self.logger.progress("Generating text embedding...")
            text_vector = self.embedder.embed_text(text_summary)
            if text_vector:
                self.logger.success(f"Text embedding generated (dim: {len(text_vector)})")
            else:
                self.logger.error("Text embedding failed")
        else:
            self.logger.warning("No text available for embedding")
        
        # Image embedding (optional)
        if image_paths:
            self.logger.progress(f"Generating image embeddings ({len(image_paths)} images)...")
            image_vector = self.embedder.embed_images(image_paths)
            if image_vector:
                self.logger.success(f"Image embedding generated (dim: {len(image_vector)})")
            else:
                self.logger.warning("Image embedding failed")
        
        # Audio embedding (optional)
        if audio_path:
            self.logger.progress("Generating audio embedding...")
            audio_vector = self.embedder.embed_audio(audio_path)
            if audio_vector:
                self.logger.success(f"Audio embedding generated (dim: {len(audio_vector)})")
            else:
                self.logger.warning("Audio embedding failed")
        
        # ===== STEP 4: Build Payload =====
        self.logger.info("--- Step 3: Building Payload ---")
        payload = self._build_payload(patient_data, event_id)
        self.logger.success("Payload constructed")
        
        # ===== STEP 5: Upsert to Qdrant =====
        self.logger.info("--- Step 4: Upserting to Qdrant ---")
        
        success = self.qdrant_client.upsert_event(
            event_id=event_id,
            patient_hash=patient_hash,
            payload=payload,
            text_vector=text_vector,
            image_vector=image_vector,
            audio_vector=audio_vector
        )
        
        if success:
            self.logger.success("Event upserted to Context Memory")
        else:
            self.logger.error("Upsert failed")
        
        # ===== STEP 6: Retrieve Patient History =====
        self.logger.info("--- Step 5: Retrieving Patient History ---")
        
        history = self.qdrant_client.get_patient_history(patient_hash)
        self.logger.success(f"Retrieved {len(history)} event(s) for patient {patient_hash}")
        
        # ===== STEP 7: Generate Patient Current State =====
        self.logger.info("--- Step 6: Generating Patient Current State ---")
        
        patient_current_state = self._generate_current_state(
            patient_hash=patient_hash,
            history=history,
            latest_event_id=event_id
        )
        
        self.logger.success("Patient Current State generated")
        
        # ===== RETURN STATE =====
        self.logger.info("=" * 50)
        self.logger.success("Patient Memory Agent completed")
        self.logger.info("=" * 50)
        
        return {
            "event_id": event_id,
            "patient_hash": patient_hash,
            "context_memory_written": success,
            "patient_current_state": patient_current_state,
            "logs": self.get_logs()
        }
    
    def _build_text_summary(self, patient_data: Dict[str, Any]) -> Optional[str]:
        """Build text summary from processed_text + audio_transcript"""
        parts = []
        
        if patient_data.get("processed_text"):
            parts.append(patient_data["processed_text"])
        
        if patient_data.get("audio_transcript"):
            parts.append(patient_data["audio_transcript"])
        
        if parts:
            summary = " ".join(parts)
            self.logger.info(f"Text summary created ({len(summary)} chars)")
            return summary
        
        return None
    
    def _extract_image_paths(self, patient_data: Dict[str, Any]) -> List[str]:
        """Extract image paths from image_quality list"""
        image_quality = patient_data.get("image_quality", [])
        paths = [item["path"] for item in image_quality if "path" in item]
        
        if paths:
            self.logger.info(f"Found {len(paths)} image(s)")
        
        return paths
    
    def _build_payload(self, patient_data: Dict[str, Any], event_id: str) -> Dict[str, Any]:
        """
        Build Qdrant payload from patient_data.
        Includes all metadata needed for filtering and downstream agents.
        """
        # Extract capture metadata
        capture_metadata = patient_data.get("capture_metadata", {})
        timestamp = capture_metadata.get("timestamp", int(time.time()))
        
        # Extract uncertainty flags
        uncertainty = patient_data.get("uncertainty_flags", {})
        
        # Build payload
        payload = {
            "event_id": event_id,
            "patient_hash": patient_data.get("patient_hash", "unknown"),
            "interaction_id": patient_data.get("interaction_id", event_id),
            
            # Demographics (extracted from patient_data if present)
            "age": patient_data.get("age"),
            "gender": patient_data.get("gender"),
            "pregnancy_status": patient_data.get("pregnancy_status"),
            
            # Program data
            "program": patient_data.get("program"),
            "village": patient_data.get("village"),
            "block": patient_data.get("block"),
            
            # Timestamp
            "timestamp": timestamp,
            
            # Uncertainty flags
            "uncertainty": uncertainty,
            
            # Store processed text for easy access
            "processed_text": patient_data.get("processed_text"),
            
            # Audio metadata
            "audio_transcript": patient_data.get("audio_transcript"),
            "audio_language": patient_data.get("audio_language"),
            "audio_confidence": patient_data.get("audio_confidence"),
            
            # Image/document counts
            "image_count": len(patient_data.get("image_quality", [])),
            "document_count": len(patient_data.get("documents", []))
        }
        
        return payload
    
    def _generate_current_state(
        self,
        patient_hash: str,
        history: List[Dict[str, Any]],
        latest_event_id: str
    ) -> Dict[str, Any]:
        """
        Generate Patient Current State summary from history.
        
        This is what gets passed to downstream agents.
        """
        if not history:
            return {
                "patient_hash": patient_hash,
                "total_visits": 0,
                "first_visit": None,
                "latest_visit": None,
                "visit_history": [],
                "recurring_themes": [],
                "uncertainty_summary": {}
            }
        
        # Extract timestamps
        timestamps = [event.get("timestamp", 0) for event in history]
        
        # Build visit summaries
        visit_history = []
        for event in history:
            visit_history.append({
                "event_id": event.get("event_id"),
                "interaction_id": event.get("interaction_id"),
                "timestamp": event.get("timestamp"),
                "summary": event.get("processed_text", "")[:200],  # First 200 chars
                "uncertainty": event.get("uncertainty", {})
            })
        
        # Aggregate uncertainty flags
        uncertainty_summary = self._aggregate_uncertainty(history)
        
        # Simple recurring theme extraction (keywords from texts)
        recurring_themes = self._extract_recurring_themes(history)
        
        current_state = {
            "patient_hash": patient_hash,
            "total_visits": len(history),
            "first_visit": min(timestamps) if timestamps else None,
            "latest_visit": max(timestamps) if timestamps else None,
            "latest_event_id": latest_event_id,
            "visit_history": visit_history,
            "recurring_themes": recurring_themes,
            "uncertainty_summary": uncertainty_summary
        }
        
        self.logger.info(f"Patient has {len(history)} visit(s) in history")
        
        return current_state
    
    def _aggregate_uncertainty(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate uncertainty flags across all visits"""
        flags = {
            "text_sparse": 0,
            "audio_noisy": 0,
            "image_unclear": 0,
            "document_unclear": 0,
            "history_partial": 0
        }
        
        for event in history:
            uncertainty = event.get("uncertainty", {})
            for key in flags.keys():
                if uncertainty.get(key, False):
                    flags[key] += 1
        
        return flags
    
    def _extract_recurring_themes(self, history: List[Dict[str, Any]]) -> List[str]:
        """
        Simple keyword extraction from processed texts.
        TODO: Replace with proper medical NER in future.
        """
        # Collect all texts
        all_text = " ".join([
            event.get("processed_text", "")
            for event in history
        ]).lower()
        
        # Simple keyword matching (very basic)
        medical_keywords = [
            "fever", "rash", "cough", "pain", "pregnant", "diabetes",
            "bp", "sugar", "headache", "skin", "wound", "infection"
        ]
        
        found_themes = []
        for keyword in medical_keywords:
            if keyword in all_text:
                found_themes.append(keyword)
        
        return found_themes[:5]  # Top 5
    
    def get_logs(self) -> List[str]:
        """Get all logs for display"""
        return self.logger.get_logs()


# ============================================================================
# LANGGRAPH-COMPATIBLE ENTRY POINT
# ============================================================================

def patient_memory_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph-compatible entry point for Patient Memory Agent.
    
    Input state:
        - patient_data: Dict (from Field Ingestion Agent)
    
    Output state:
        - event_id: str
        - patient_hash: str
        - context_memory_written: bool
        - patient_current_state: Dict
        - logs: List[str]
    
    Usage in LangGraph:
        from my_code.patient_memory_agent import patient_memory_agent
        
        graph.add_node("patient_memory", patient_memory_agent)
    """
    agent = PatientMemoryAgent()
    
    patient_data = state.get("patient_data")
    
    if not patient_data:
        return {
            "event_id": None,
            "patient_hash": None,
            "context_memory_written": False,
            "patient_current_state": {},
            "logs": ["✗ No patient_data in state"]
        }
    
    return agent.process(patient_data)


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PATIENT MEMORY AGENT - DEMO")
    print("=" * 70 + "\n")
    
    # Create fake patient_data
    fake_patient_data = {
        "patient_hash": "test_patient_123",
        "interaction_id": "demo-interaction-001",
        "processed_text": "Patient presents with mild fever and cough",
        "audio_transcript": "Patient says feeling better today",
        "audio_language": "en",
        "audio_confidence": 0.95,
        "image_quality": [],  # No images in demo
        "documents": [],
        "uncertainty_flags": {
            "text_sparse": False,
            "audio_noisy": False,
            "image_unclear": False,
            "document_unclear": False,
            "history_partial": True
        },
        "capture_metadata": {
            "device_id": "demo_device",
            "offline": False,
            "timestamp": int(time.time())
        }
    }
    
    # Process through agent
    agent = PatientMemoryAgent()
    result = agent.process(fake_patient_data)
    
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Event ID: {result['event_id']}")
    print(f"Patient Hash: {result['patient_hash']}")
    print(f"Memory Written: {result['context_memory_written']}")
    print(f"\nPatient Current State:")
    print(f"  Total Visits: {result['patient_current_state']['total_visits']}")
    print(f"  Visit History: {len(result['patient_current_state']['visit_history'])} event(s)")
    
    print("\n✅ Patient Memory Agent ready!")
