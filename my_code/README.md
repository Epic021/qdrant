# Field Ingestion Agent

A multimodal data ingestion system for rural healthcare. Processes text, audio, images, and documents with quality assessment and uncertainty tagging.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd my_code
pip install -r requirements.txt
```

**Optional dependencies for full features:**
```bash
# Audio transcription (requires ~1GB model download)
pip install openai-whisper

# OCR (requires Tesseract installation)
pip install pytesseract

# PDF text extraction
pip install PyMuPDF
```

### 2. Run the Application

```bash
python api.py
```

### 3. Open Browser

Navigate to: **http://localhost:8000**

---

## 📁 Project Structure

```
my_code/
├── agents.py          # Core Field Ingestion Agent logic
├── db.py              # SQLite database layer
├── api.py             # FastAPI backend
├── requirements.txt   # Python dependencies
├── static/
│   └── index.html     # Web frontend
├── uploads/           # Uploaded files storage
│   ├── audio/
│   ├── images/
│   └── documents/
└── ingestion.db       # SQLite database (created on first run)
```

---

## 🔧 Architecture

### Backend (FastAPI)
- **POST /api/process**: Process patient data
- **GET /api/interactions**: List processed interactions
- **GET /api/interactions/{id}**: Get specific interaction
- **GET /api/stats**: Database statistics

### Agent Processing Pipeline

```
A. Text Processing
   └─ Normalize whitespace
   └─ Check length → tag text_sparse if short

B. Audio Processing (if Whisper available)
   └─ Transcribe with Whisper
   └─ Detect language
   └─ Estimate confidence → tag audio_noisy if low

C. Image Processing
   └─ Check resolution
   └─ Calculate blur score
   └─ Quality score → tag image_unclear if low

D. Document Processing (if Tesseract available)
   └─ Run OCR
   └─ Calculate confidence → tag document_unclear if low

E. Uncertainty Assessment
   └─ Set history_partial if < 2 modalities present
   └─ Assemble all flags

F. Create patient_data JSON
```

### Database Tables

**raw_interactions**
| Column | Type | Description |
|--------|------|-------------|
| interaction_id | TEXT | Primary key |
| patient_hash | TEXT | Anonymized patient ID |
| raw_text | TEXT | Original text input |
| raw_audio_path | TEXT | Path to audio file |
| raw_image_paths | TEXT | JSON array of image paths |
| raw_document_paths | TEXT | JSON array of document paths |
| timestamp | INTEGER | Unix timestamp |

**processed_interactions**
| Column | Type | Description |
|--------|------|-------------|
| interaction_id | TEXT | Primary key |
| patient_hash | TEXT | Anonymized patient ID |
| patient_data_json | TEXT | Full canonical JSON |
| uncertainty_flags | TEXT | Flags JSON |
| processing_timestamp | INTEGER | When processed |

---

## 📋 Output Schema

```json
{
  "patient_hash": "string",
  "interaction_id": "uuid",
  "processed_text": "string | null",
  "audio_transcript": "string | null",
  "audio_language": "string | null",
  "audio_confidence": "float | null",
  "image_quality": [
    { "path": "string", "score": "float" }
  ],
  "documents": [
    {
      "path": "string",
      "ocr_text": "string",
      "ocr_confidence": "float"
    }
  ],
  "uncertainty_flags": {
    "text_sparse": "boolean",
    "audio_noisy": "boolean",
    "image_unclear": "boolean",
    "document_unclear": "boolean",
    "history_partial": "boolean"
  },
  "capture_metadata": {
    "device_id": "string",
    "offline": "boolean",
    "timestamp": "unix_epoch"
  }
}
```

---

## 🧪 Testing

### Run Agent Standalone

```bash
python agents.py
```

### Test Database Layer

```bash
python db.py
```

### Example API Call

```bash
curl -X POST http://localhost:8000/api/process \
  -F "patient_hash=patient_123" \
  -F "text_notes=Patient has skin rash" \
  -F "images=@skin_image.jpg"
```

---

## 🔮 LangGraph Integration

The agent is designed for future LangGraph integration:

```python
from agents import field_ingestion_agent

def field_ingestion_node(state):
    return field_ingestion_agent(state)

# Returns: {"patient_data": {...}, "logs": [...]}
```

---

## ⚠️ Design Philosophy

> **This agent stabilizes reality.**  
> It does not understand patients.  
> It prepares data so later agents can reason safely.

- ❌ No LLM
- ❌ No medical reasoning
- ❌ No skipping steps
- ❌ No silent failures
- ✅ Log everything
- ✅ Tag uncertainty
- ✅ Continue on error
