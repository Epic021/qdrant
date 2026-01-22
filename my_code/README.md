# Rural Healthcare Field Ingestion System

## Overview
An AI-powered system for rural healthcare workers to process patient data through images, audio, and text, retrieving similar historical cases for informed decision-making.

## Project Structure

```
my_code/
├── api.py                              # Main FastAPI server & workflow orchestration
├── agents.py                           # Agent 1: Field Ingestion Agent
├── patient_memory_agent.py             # Agent 2: Patient Memory Agent (Qdrant)
├── context_builder_agent.py            # Agent 3: Context Builder Agent
├── similar_case_retrieval_agent.py     # Agent 4: Similar Case Retrieval
├── explanation_and_referral_agents.py  # Agents 5 & 6: Explanation & Referral
├── db.py                               # SQLite Database Layer
├── qdrant_manager.py                   # Qdrant Cloud Client Wrapper
├── embeddings.py                       # Multi-modal Embedding Generators
├── llm_client.py                       # Gemini LLM Wrapper
├── static/
│   └── index_new.html                  # Web UI
└── uploads/                            # File storage (audio, images, docs)
```

## System Architecture

### 6-Agent Workflow
1. **Field Ingestion Agent** - Processes multi-modal input (text/audio/images/documents)
2. **Patient Memory Agent** - Stores events in Qdrant vector DB
3. **Context Builder** - Creates retrieval plan based on patient state
4. **Similar Case Retrieval** - Hybrid search (dense + sparse) for similar cases
5. **Explanation Agent** - Generates user-facing explanations (Gemini)
6. **Referral & Follow-up** - Detects gaps and generates nudges

### Data Flow
```
Input (Text/Audio/Image/Docs) 
  → Agent 1: Extract & QA
  → Agent 2: Store in Qdrant
  → Agent 3: Build retrieval plan
  → Agent 4: Find similar cases
  → Agent 5: Generate explanation
  → Agent 6: Follow-up detection
  → Output (Case summary + Recommendations + Chat)
```

## Technology Stack

- **Backend**: FastAPI + LangGraph
- **Database**: SQLite (structured data) + Qdrant Cloud (vector embeddings)
- **LLM**: Google Gemini 2.0 Flash
- **Embeddings**: 
  - Text: BioBERT (pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb)
  - Images: CLIP (openai/clip-vit-base-patch32)
  - Audio: Whisper (openai/whisper-base)
- **Frontend**: Vanilla HTML/CSS/JavaScript

## Setup

### 1. Environment Variables
Create `.env` file:
```bash
GEMINI_API_KEY=your_gemini_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_cloud_url
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Server
```bash
python api.py
```

Open `http://localhost:8000` in browser.

## Key Features

### Multi-Modal Processing
- Text notes (clinical observations)
- Audio recordings (transcribed via Whisper)
- Images (quality assessment + CLIP embeddings)
- Documents (OCR via Tesseract)

### Vector Search
- **Dense Search**: Semantic similarity using embeddings
- **Sparse Search**: BM25 keyword matching
- **Hybrid Re-ranking**: Combines both methods

### Safety Constraints
- No medical diagnosis
- No invented facts
- Uncertainty flagging (sparse data, noisy audio, unclear images)
- Geography-based filtering (same village/block)
- Verified cases only

### Chat Interface
- Context-aware Q&A about patient cases
- References similar historical cases
- Simple, actionable recommendations

## Database Schema

### SQLite Tables
1. `raw_interactions` - Uploaded files and metadata
2. `processed_interactions` - Extracted patient_data JSON
3. `patient_states` - Patient memory snapshots
4. `retrieval_plans` - Context builder outputs
5. `retrieval_results` - Similar cases found
6. `final_case_outputs` - Explanations and follow-ups

### Qdrant Collection
- **patient_context_events_v1**
  - Named vectors: `text_event` (768d), `image_event` (512d), `audio_event` (512d)
  - Payload: patient_hash, program, pregnancy_status, timestamp, etc.

## API Endpoints

- `POST /api/process` - Process new patient interaction
- `GET /api/final_output/{interaction_id}` - Get complete analysis
- `POST /api/chat` - Chat about specific interaction
- `POST /api/search` - Search patient history

## Workflow Configuration

LangGraph manages the agent pipeline:
```
Agent1 → Agent2 → Agent3 → Agent4 → Agent5 → Agent6 → Finalize
```

State is preserved across nodes, enabling rich context passing.

## Development Notes

- All embeddings are cached to avoid redundant processing
- Qdrant filters enforce safety (verified cases, geography scope)
- Uncertainty flags propagate through the pipeline
- File storage uses patient_hash for HIPAA-like anonymization

## License
Convolve Hackathon Submission 2025
