"""
Media Processor Tool
====================
Handles raw data processing:
- Text Normalization
- Audio Transcription (Whisper)
- Image Quality Assessment
- Document OCR (Tesseract/PDF)
"""

import os
import re
import math
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import logging

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Configure logging
logger = logging.getLogger("media_processor")

class Config:
    TEXT_SPARSE_THRESHOLD = 20
    AUDIO_CONFIDENCE_THRESHOLD = 0.7
    IMAGE_QUALITY_THRESHOLD = 0.5
    MIN_IMAGE_RESOLUTION = 100
    OCR_CONFIDENCE_THRESHOLD = 60

# ============================================================================
# TEXT PROCESSOR
# ============================================================================

class TextProcessor:
    @staticmethod
    def process(text: Optional[str]) -> Tuple[Optional[str], bool]:
        if not text or not text.strip():
            return None, True
        
        normalized = re.sub(r'\s+', ' ', text.strip())
        is_sparse = len(normalized) < Config.TEXT_SPARSE_THRESHOLD
        
        return normalized, is_sparse

# ============================================================================
# AUDIO PROCESSOR
# ============================================================================

class AudioProcessor:
    def __init__(self):
        self.model = None

    def _load_model(self):
        if self.model is None and WHISPER_AVAILABLE:
            try:
                self.model = whisper.load_model("base")
            except Exception as e:
                logger.error(f"Whisper load failed: {e}")
                
    def process(self, audio_path: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[float], bool]:
        if not audio_path or not os.path.exists(audio_path):
            return None, None, None, False
            
        if not WHISPER_AVAILABLE:
            return None, None, None, True
            
        try:
            self._load_model()
            if not self.model:
                return None, None, None, True
                
            result = self.model.transcribe(audio_path)
            transcript = result.get("text", "").strip()
            language = result.get("language", "unknown")
            
            segments = result.get("segments", [])
            confidence = 1.0 - (sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)) if segments else 0.5
            
            is_noisy = confidence < Config.AUDIO_CONFIDENCE_THRESHOLD
            return transcript, language, round(confidence, 3), is_noisy
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return None, None, None, True

# ============================================================================
# IMAGE PROCESSOR
# ============================================================================

class ImageProcessor:
    def process(self, image_paths: Optional[List[str]]) -> Tuple[List[Dict[str, Any]], bool]:
        if not image_paths:
            return [], False
            
        results = []
        any_unclear = False
        
        for path in image_paths:
            if not os.path.exists(path):
                results.append({"path": path, "score": 0.0})
                any_unclear = True
                continue
                
            score = self._assess_quality(path)
            results.append({"path": path, "score": score})
            
            if score < Config.IMAGE_QUALITY_THRESHOLD:
                any_unclear = True
                
        return results, any_unclear
    
    def _assess_quality(self, path: str) -> float:
        if not PIL_AVAILABLE:
            return 0.0
        try:
            img = Image.open(path).convert('L')
            img_array = np.array(img, dtype=np.float64)
            # Simple gradient-based sharpness
            gx = np.diff(img_array, axis=1)
            gy = np.diff(img_array, axis=0)
            gnorm = np.sqrt(gx[:, :-1]**2 + gy[:-1, :]**2)
            sharpness = np.average(gnorm)
            return min(sharpness / 50, 1.0)
        except Exception:
            return 0.0

# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    def process(self, doc_paths: Optional[List[str]]) -> Tuple[List[Dict[str, Any]], bool]:
        if not doc_paths:
            return [], False
            
        results = []
        any_unclear = False
        
        for path in doc_paths:
            if not os.path.exists(path):
                results.append({"path": path, "ocr_text": "", "ocr_confidence": 0.0})
                any_unclear = True
                continue
                
            text, conf = self._extract_text(path)
            results.append({"path": path, "ocr_text": text, "ocr_confidence": conf})
            
            if conf < Config.OCR_CONFIDENCE_THRESHOLD:
                any_unclear = True
                
        return results, any_unclear
        
    def _extract_text(self, path: str) -> Tuple[str, float]:
        ext = Path(path).suffix.lower()
        
        if ext == '.txt':
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read(), 100.0
            except:
                return "", 0.0
        
        # Placeholder for PDF/Image OCR using Tesseract/PyMuPDF
        # Simplified for now to avoid dependency hell if not installed
        return "", 0.0
