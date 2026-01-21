"""
Field Ingestion Agent
=====================
Core processing logic for rural healthcare data ingestion.
Handles text, audio, images, and documents with quality assessment and uncertainty tagging.

This agent stabilizes reality. It does not understand patients.
It prepares data so later agents can reason safely.
"""

import os
import re
import uuid
import time
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json

# Image processing
try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Audio processing (Whisper)
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# OCR processing
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Agent configuration thresholds"""
    TEXT_SPARSE_THRESHOLD = 20  # Characters below this = sparse
    AUDIO_CONFIDENCE_THRESHOLD = 0.7  # Below this = noisy
    IMAGE_QUALITY_THRESHOLD = 0.5  # Below this = unclear
    IMAGE_BLUR_THRESHOLD = 100  # Laplacian variance below this = blurry
    OCR_CONFIDENCE_THRESHOLD = 60  # Below this = unclear
    MIN_IMAGE_RESOLUTION = 100  # Minimum dimension in pixels


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ImageQuality:
    path: str
    score: float


@dataclass
class DocumentResult:
    path: str
    ocr_text: str
    ocr_confidence: float


@dataclass
class UncertaintyFlags:
    text_sparse: bool = False
    audio_noisy: bool = False
    image_unclear: bool = False
    document_unclear: bool = False
    history_partial: bool = False


@dataclass
class CaptureMetadata:
    device_id: str
    offline: bool
    timestamp: int


@dataclass
class PatientData:
    """Canonical patient data JSON structure"""
    patient_hash: str
    interaction_id: str
    processed_text: Optional[str] = None
    audio_transcript: Optional[str] = None
    audio_language: Optional[str] = None
    audio_confidence: Optional[float] = None
    image_quality: List[Dict[str, Any]] = field(default_factory=list)
    documents: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty_flags: Dict[str, bool] = field(default_factory=dict)
    capture_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching exact schema"""
        return {
            "patient_hash": self.patient_hash,
            "interaction_id": self.interaction_id,
            "processed_text": self.processed_text,
            "audio_transcript": self.audio_transcript,
            "audio_language": self.audio_language,
            "audio_confidence": self.audio_confidence,
            "image_quality": self.image_quality,
            "documents": self.documents,
            "uncertainty_flags": self.uncertainty_flags,
            "capture_metadata": self.capture_metadata
        }


# ============================================================================
# LOGGER
# ============================================================================

class AgentLogger:
    """
    Logger that captures step-by-step processing for frontend display.
    Emits structured log entries with status icons.
    """
    
    STATUS_SUCCESS = "✓"
    STATUS_WARNING = "⚠"
    STATUS_PROGRESS = "⏳"
    STATUS_ERROR = "✗"
    STATUS_INFO = "ℹ"
    
    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        self.logs: List[Dict[str, Any]] = []
        self.callback = callback
    
    def _emit(self, status: str, message: str, level: str = "info"):
        """Emit a log entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "message": message,
            "level": level
        }
        self.logs.append(entry)
        
        formatted = f"[{status}] {message}"
        print(formatted)  # Console output
        
        if self.callback:
            self.callback(formatted)
    
    def success(self, message: str):
        self._emit(self.STATUS_SUCCESS, message, "success")
    
    def warning(self, message: str):
        self._emit(self.STATUS_WARNING, message, "warning")
    
    def progress(self, message: str):
        self._emit(self.STATUS_PROGRESS, message, "progress")
    
    def error(self, message: str):
        self._emit(self.STATUS_ERROR, message, "error")
    
    def info(self, message: str):
        self._emit(self.STATUS_INFO, message, "info")
    
    def get_logs(self) -> List[Dict[str, Any]]:
        return self.logs.copy()
    
    def get_formatted_logs(self) -> List[str]:
        return [f"[{log['status']}] {log['message']}" for log in self.logs]


# ============================================================================
# TEXT PROCESSOR
# ============================================================================

class TextProcessor:
    """Process and normalize text input"""
    
    def __init__(self, logger: AgentLogger):
        self.logger = logger
    
    def process(self, text: Optional[str]) -> tuple[Optional[str], bool]:
        """
        Process text input.
        Returns: (normalized_text, is_sparse)
        """
        if text is None or text.strip() == "":
            self.logger.warning("No text input provided → tagging text_sparse=true")
            return None, True
        
        self.logger.progress("Processing text input...")
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', text.strip())
        
        # Check length threshold
        is_sparse = len(normalized) < Config.TEXT_SPARSE_THRESHOLD
        
        if is_sparse:
            self.logger.warning(f"Text length ({len(normalized)} chars) below threshold → tagging text_sparse=true")
        else:
            self.logger.success(f"Text processed successfully ({len(normalized)} chars)")
        
        return normalized, is_sparse


# ============================================================================
# AUDIO PROCESSOR
# ============================================================================

class AudioProcessor:
    """Process audio files using Whisper for transcription"""
    
    def __init__(self, logger: AgentLogger):
        self.logger = logger
        self.model = None
    
    def _load_model(self):
        """Lazy load Whisper model"""
        if self.model is None and WHISPER_AVAILABLE:
            self.logger.progress("Loading Whisper model...")
            try:
                self.model = whisper.load_model("base")
                self.logger.success("Whisper model loaded")
            except Exception as e:
                self.logger.error(f"Failed to load Whisper model: {e}")
                self.model = None
    
    def process(self, audio_path: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[float], bool]:
        """
        Process audio file.
        Returns: (transcript, language, confidence, is_noisy)
        """
        if audio_path is None:
            self.logger.info("No audio input provided")
            return None, None, None, False
        
        if not os.path.exists(audio_path):
            self.logger.error(f"Audio file not found: {audio_path}")
            return None, None, None, True
        
        if not WHISPER_AVAILABLE:
            self.logger.warning("Whisper not available → cannot transcribe audio → tagging audio_noisy=true")
            return None, None, None, True
        
        self.logger.progress("Transcribing audio...")
        
        try:
            self._load_model()
            
            if self.model is None:
                self.logger.error("Whisper model not loaded → tagging audio_noisy=true")
                return None, None, None, True
            
            # Transcribe
            result = self.model.transcribe(audio_path)
            
            transcript = result.get("text", "").strip()
            language = result.get("language", "unknown")
            
            # Estimate confidence from segments
            segments = result.get("segments", [])
            if segments:
                avg_no_speech_prob = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
                confidence = 1.0 - avg_no_speech_prob
            else:
                confidence = 0.5  # Default medium confidence
            
            is_noisy = confidence < Config.AUDIO_CONFIDENCE_THRESHOLD
            
            if is_noisy:
                self.logger.warning(f"Audio confidence low ({confidence:.2f}) → tagging audio_noisy=true")
            else:
                self.logger.success(f"Audio transcribed: language={language}, confidence={confidence:.2f}")
            
            self.logger.success(f"Transcript: {transcript[:100]}..." if len(transcript) > 100 else f"Transcript: {transcript}")
            
            return transcript, language, round(confidence, 3), is_noisy
            
        except Exception as e:
            self.logger.error(f"Audio transcription failed: {e} → tagging audio_noisy=true")
            return None, None, None, True


# ============================================================================
# IMAGE PROCESSOR
# ============================================================================

class ImageProcessor:
    """Process images and assess quality"""
    
    def __init__(self, logger: AgentLogger):
        self.logger = logger
    
    def _calculate_blur_score(self, image_path: str) -> float:
        """Calculate blur score using Laplacian variance"""
        try:
            img = Image.open(image_path).convert('L')  # Grayscale
            img_array = np.array(img, dtype=np.float64)
            
            # Laplacian kernel
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
            
            from scipy import ndimage
            lap = ndimage.convolve(img_array, laplacian)
            variance = lap.var()
            
            # Normalize to 0-1 range (higher = sharper)
            normalized = min(variance / 500, 1.0)  # 500 is typical sharp image variance
            return normalized
            
        except ImportError:
            # Fallback if scipy not available - use simple edge detection
            img = Image.open(image_path).convert('L')
            img_array = np.array(img, dtype=np.float64)
            
            # Simple gradient-based sharpness
            gx = np.diff(img_array, axis=1)
            gy = np.diff(img_array, axis=0)
            gnorm = np.sqrt(gx[:, :-1]**2 + gy[:-1, :]**2)
            sharpness = np.average(gnorm)
            
            return min(sharpness / 50, 1.0)
        except Exception:
            return 0.5  # Default medium score on error
    
    def _check_resolution(self, image_path: str) -> tuple[int, int, bool]:
        """Check image resolution"""
        try:
            img = Image.open(image_path)
            width, height = img.size
            is_low_res = min(width, height) < Config.MIN_IMAGE_RESOLUTION
            return width, height, is_low_res
        except Exception:
            return 0, 0, True
    
    def process(self, image_paths: Optional[List[str]]) -> tuple[List[Dict[str, Any]], bool]:
        """
        Process image files.
        Returns: (image_quality_list, any_unclear)
        """
        if not image_paths:
            self.logger.info("No images provided")
            return [], False
        
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available → cannot assess image quality → tagging image_unclear=true")
            return [{"path": p, "score": 0.0} for p in image_paths], True
        
        results = []
        any_unclear = False
        
        for path in image_paths:
            self.logger.progress(f"Analyzing image: {os.path.basename(path)}...")
            
            if not os.path.exists(path):
                self.logger.error(f"Image not found: {path}")
                results.append({"path": path, "score": 0.0})
                any_unclear = True
                continue
            
            try:
                # Check resolution
                width, height, is_low_res = self._check_resolution(path)
                
                if is_low_res:
                    self.logger.warning(f"Image resolution low ({width}x{height})")
                
                # Calculate blur score
                blur_score = self._calculate_blur_score(path)
                
                # Combined quality score
                res_score = 1.0 if not is_low_res else 0.3
                quality_score = (blur_score * 0.7) + (res_score * 0.3)
                quality_score = round(quality_score, 3)
                
                results.append({"path": path, "score": quality_score})
                
                if quality_score < Config.IMAGE_QUALITY_THRESHOLD:
                    self.logger.warning(f"Image quality low ({quality_score:.2f}) → tagging image_unclear=true")
                    any_unclear = True
                else:
                    self.logger.success(f"Image quality acceptable ({quality_score:.2f})")
                    
            except Exception as e:
                self.logger.error(f"Image processing failed for {path}: {e}")
                results.append({"path": path, "score": 0.0})
                any_unclear = True
        
        return results, any_unclear


# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    """Process documents with OCR"""
    
    def __init__(self, logger: AgentLogger):
        self.logger = logger
    
    def process(self, document_paths: Optional[List[str]]) -> tuple[List[Dict[str, Any]], bool]:
        """
        Process document files with OCR.
        Returns: (documents_list, any_unclear)
        """
        if not document_paths:
            self.logger.info("No documents provided")
            return [], False
        
        results = []
        any_unclear = False
        
        for path in document_paths:
            self.logger.progress(f"Processing document: {os.path.basename(path)}...")
            
            if not os.path.exists(path):
                self.logger.error(f"Document not found: {path}")
                results.append({"path": path, "ocr_text": "", "ocr_confidence": 0.0})
                any_unclear = True
                continue
            
            ext = Path(path).suffix.lower()
            
            try:
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                    # Image-based document - run OCR
                    ocr_text, ocr_conf = self._run_ocr_on_image(path)
                elif ext == '.pdf':
                    # PDF document
                    ocr_text, ocr_conf = self._run_ocr_on_pdf(path)
                elif ext == '.txt':
                    # Plain text - just read
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        ocr_text = f.read()
                    ocr_conf = 100.0
                    self.logger.success(f"Text document read successfully")
                else:
                    self.logger.warning(f"Unsupported document format: {ext}")
                    ocr_text = ""
                    ocr_conf = 0.0
                
                results.append({
                    "path": path,
                    "ocr_text": ocr_text,
                    "ocr_confidence": round(ocr_conf, 2)
                })
                
                if ocr_conf < Config.OCR_CONFIDENCE_THRESHOLD:
                    self.logger.warning(f"OCR confidence low ({ocr_conf:.1f}%) → tagging document_unclear=true")
                    any_unclear = True
                else:
                    self.logger.success(f"Document processed, OCR confidence: {ocr_conf:.1f}%")
                    
            except Exception as e:
                self.logger.error(f"Document processing failed for {path}: {e}")
                results.append({"path": path, "ocr_text": "", "ocr_confidence": 0.0})
                any_unclear = True
        
        return results, any_unclear
    
    def _run_ocr_on_image(self, image_path: str) -> tuple[str, float]:
        """Run OCR on an image file"""
        if not TESSERACT_AVAILABLE:
            self.logger.warning("Tesseract not available → OCR skipped")
            return "", 0.0
        
        if not PIL_AVAILABLE:
            self.logger.warning("PIL not available → OCR skipped")
            return "", 0.0
        
        try:
            img = Image.open(image_path)
            
            # Get OCR data with confidence
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            # Extract text and calculate average confidence
            texts = []
            confidences = []
            
            for i, text in enumerate(data['text']):
                conf = int(data['conf'][i])
                if conf > 0 and text.strip():
                    texts.append(text)
                    confidences.append(conf)
            
            ocr_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return ocr_text, avg_confidence
            
        except Exception as e:
            self.logger.error(f"OCR failed: {e}")
            return "", 0.0
    
    def _run_ocr_on_pdf(self, pdf_path: str) -> tuple[str, float]:
        """Run OCR on a PDF file"""
        try:
            # PyMuPDF changed import name in version 1.24+
            try:
                import pymupdf as fitz  # New import (v1.24+)
            except ImportError:
                import fitz  # Old import (v1.23 and earlier)
            
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page in doc:
                full_text += page.get_text()
            
            doc.close()
            
            if full_text.strip():
                self.logger.success("PDF text extracted directly")
                return full_text.strip(), 95.0  # High confidence for direct text
            else:
                self.logger.info("PDF has no text layer → attempting image OCR")
                # Would need to render pages to images and OCR - simplified for now
                return "", 50.0
                
        except ImportError:
            self.logger.warning("PyMuPDF not available → PDF processing limited")
            return "", 0.0
        except Exception as e:
            self.logger.error(f"PDF processing failed: {e}")
            return "", 0.0


# ============================================================================
# FIELD INGESTION AGENT
# ============================================================================

class FieldIngestionAgent:
    """
    Main Field Ingestion Agent.
    Processes multimodal patient data and creates canonical JSON.
    
    This agent stabilizes reality. It does not understand patients.
    It prepares data so later agents can reason safely.
    """
    
    def __init__(self, log_callback: Optional[Callable[[str], None]] = None):
        self.logger = AgentLogger(callback=log_callback)
        self.text_processor = TextProcessor(self.logger)
        self.audio_processor = AudioProcessor(self.logger)
        self.image_processor = ImageProcessor(self.logger)
        self.document_processor = DocumentProcessor(self.logger)
    
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
        Process patient data and create canonical JSON.
        
        This is the main entry point for the agent.
        Can later be wrapped as a LangGraph node:
            def field_ingestion_agent(state):
                return {"patient_data": agent.process(**state)}
        
        Args:
            patient_hash: Anonymized patient identifier
            raw_text: Free-text clinical notes
            audio_path: Path to audio recording
            image_paths: List of paths to images
            document_paths: List of paths to documents
            device_id: Capture device identifier
            offline: Whether captured offline
            timestamp: Unix timestamp (defaults to now)
        
        Returns:
            Canonical patient_data JSON dict
        """
        # Generate interaction ID
        interaction_id = str(uuid.uuid4())
        
        self.logger.info("=" * 50)
        self.logger.info(f"Starting Field Ingestion Agent")
        self.logger.info(f"Patient Hash: {patient_hash}")
        self.logger.info(f"Interaction ID: {interaction_id}")
        self.logger.info("=" * 50)
        
        # Initialize uncertainty flags
        text_sparse = False
        audio_noisy = False
        image_unclear = False
        document_unclear = False
        
        # ===== A. TEXT PROCESSING =====
        self.logger.info("--- Step A: Text Processing ---")
        processed_text, text_sparse = self.text_processor.process(raw_text)
        
        # ===== B. AUDIO PROCESSING =====
        self.logger.info("--- Step B: Audio Processing ---")
        audio_transcript, audio_language, audio_confidence, audio_noisy = \
            self.audio_processor.process(audio_path)
        
        # ===== C. IMAGE PROCESSING =====
        self.logger.info("--- Step C: Image Processing ---")
        image_quality, image_unclear = self.image_processor.process(image_paths)
        
        # ===== D. DOCUMENT PROCESSING =====
        self.logger.info("--- Step D: Document Processing ---")
        documents, document_unclear = self.document_processor.process(document_paths)
        
        # ===== E. UNCERTAINTY FLAGS =====
        self.logger.info("--- Step E: Uncertainty Assessment ---")
        
        # Check for history_partial (multiple modalities missing)
        modalities_present = sum([
            processed_text is not None,
            audio_transcript is not None,
            len(image_quality) > 0,
            len(documents) > 0
        ])
        history_partial = modalities_present < 2
        
        if history_partial:
            self.logger.warning(f"Only {modalities_present} modality(ies) present → tagging history_partial=true")
        
        uncertainty_flags = {
            "text_sparse": text_sparse,
            "audio_noisy": audio_noisy,
            "image_unclear": image_unclear,
            "document_unclear": document_unclear,
            "history_partial": history_partial
        }
        
        self.logger.success(f"Uncertainty flags: {uncertainty_flags}")
        
        # ===== F. ASSEMBLE CANONICAL JSON =====
        self.logger.info("--- Step F: Creating patient_data JSON ---")
        
        capture_metadata = {
            "device_id": device_id,
            "offline": offline,
            "timestamp": timestamp or int(time.time())
        }
        
        patient_data = PatientData(
            patient_hash=patient_hash,
            interaction_id=interaction_id,
            processed_text=processed_text,
            audio_transcript=audio_transcript,
            audio_language=audio_language,
            audio_confidence=audio_confidence,
            image_quality=image_quality,
            documents=documents,
            uncertainty_flags=uncertainty_flags,
            capture_metadata=capture_metadata
        )
        
        result = patient_data.to_dict()
        
        self.logger.success("patient_data JSON created successfully")
        self.logger.info("=" * 50)
        self.logger.success("Field Ingestion Agent completed")
        self.logger.info("=" * 50)
        
        return result
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all processing logs"""
        return self.logger.get_logs()
    
    def get_formatted_logs(self) -> List[str]:
        """Get formatted log strings for display"""
        return self.logger.get_formatted_logs()


# ============================================================================
# STANDALONE ENTRY POINT
# ============================================================================

def field_ingestion_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph-compatible entry point.
    
    Usage:
        result = field_ingestion_agent(state)
        patient_data = result["patient_data"]
    """
    agent = FieldIngestionAgent()
    
    patient_data = agent.process(
        patient_hash=state.get("patient_hash", "unknown"),
        raw_text=state.get("raw_text"),
        audio_path=state.get("audio_path"),
        image_paths=state.get("image_paths"),
        document_paths=state.get("document_paths"),
        device_id=state.get("device_id", "device_001"),
        offline=state.get("offline", False),
        timestamp=state.get("timestamp")
    )
    
    return {"patient_data": patient_data, "logs": agent.get_formatted_logs()}


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FIELD INGESTION AGENT - DEMO")
    print("=" * 70 + "\n")
    
    # Create agent
    agent = FieldIngestionAgent()
    
    # Process sample data (text only for demo)
    result = agent.process(
        patient_hash="patient_abc123",
        raw_text="Patient presents with skin rash on forearm. Mild itching reported. No fever.",
        device_id="demo_device",
        offline=False
    )
    
    print("\n" + "=" * 70)
    print("FINAL OUTPUT: patient_data JSON")
    print("=" * 70)
    print(json.dumps(result, indent=2))
