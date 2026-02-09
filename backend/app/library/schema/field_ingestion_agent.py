from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ImageQuality(BaseModel):
    path: str
    score: float

class DocumentResult(BaseModel):
    path: str
    ocr_text: str
    ocr_confidence: float

class CaptureMetadata(BaseModel):
    device_id: str
    offline: bool
    timestamp: int

class UncertaintyFlags(BaseModel):
    text_sparse: bool = False
    audio_noisy: bool = False
    image_unclear: bool = False
    document_unclear: bool = False
    history_partial: bool = False

class PatientData(BaseModel):
    """Canonical patient data JSON structure"""
    patient_hash: str
    interaction_id: str
    processed_text: Optional[str] = None
    audio_transcript: Optional[str] = None
    audio_language: Optional[str] = None
    audio_confidence: Optional[float] = None
    image_quality: List[ImageQuality] = Field(default_factory=list)
    documents: List[DocumentResult] = Field(default_factory=list)
    uncertainty_flags: UncertaintyFlags = Field(default_factory=UncertaintyFlags)
    capture_metadata: CaptureMetadata

class FieldIngestionInput(BaseModel):
    patient_hash: str
    raw_text: Optional[str] = None
    audio_path: Optional[str] = None
    image_paths: Optional[List[str]] = None
    document_paths: Optional[List[str]] = None
    device_id: str = "device_001"
    offline: bool = False
    timestamp: Optional[int] = None

class FieldIngestionOutput(BaseModel):
    patient_data: PatientData
    logs: List[str]
