# Sahayak - Patient Data Processing System

**Sahayak** is an advanced **Agentic AI System** designed to process multi-modal patient data (Text, Audio, Images, Documents) for healthcare applications. It utilizes a layered architecture of specialized agents to ensure data integrity, safe historical retrieval (RAG), and hallucination-free recommendations.

## 🚀 Key Features

-   **Multi-Modal Ingestion**: Seamlessly processes raw text, voice recordings (with transcription), medical images, and PDF documents.
-   **Agentic Architecture**: A pipeline of 6 specialized agents (Ingestion, Context, Memory, Retriever, Recommender, Reviewer) working in concert.
-   **Hybrid RAG**: Combines Vector Search (Qdrant) with Deterministic SQL History for grounded retrieval.
-   **Safety First**: Includes a dedicated **Reviewer Agent** to validate all AI outputs against evidence, preventing hallucinations and unsafe medical advice.
-   **Real-time Feedback**: WebSocket integration provides live updates on the agent workflow steps to the frontend.

## 🏗 Architecture

The system follows a linear graph workflow managed by an **Orchestrator**:

1.  **Field Ingestion Agent**: Normalizes inputs, runs OCR/STT, and flags quality issues (blur, noise).
2.  **Context Builder Agent**: Analyzes data to create safe retrieval constraints (e.g., "Must be pregnant", "Adults only").
3.  **Patient Memory Agent**: Generates embeddings and persists data to Long-Term Memory (Vector + SQL).
4.  **Retriever Agent**: Executes hybrid search to find relevant past cases based on the plan.
5.  **Recommender Agent**: Synthesizes current state and past cases to generate insights and referral signals.
6.  **Reviewer Agent**: audits the analysis for safety and factual grounding before output.

For a detailed workflow breakdown, see [workflow_description.txt](./workflow_description.txt).

## 🛠 Tech Stack

-   **Backend**: Python 3.10+, FastAPI
-   **Frontend**: React 19, Vite
-   **AI/LLM**: Google Gemini 1.5 Pro, Sentence-Transformers
-   **Database**: Qdrant (Vector), SQLite (Relational)
-   **Tools**: Pillow (Images), Whisper (Audio - optional)

## 📦 Installation

### Prerequisites
-   Python 3.10 or higher
-   Node.js 18+
-   A running Qdrant instance (Docker or Cloud)

### 1. Backend Setup

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Configuration**: Create a `.env` file in the `backend` directory with your API keys:
```env
GOOGLE_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
```

### 2. Frontend Setup

```bash
python check_db.py
```

For Qdrant, ensure your instance is running and accessible.

### 6. (Optional) Load Sample Patient Data

```bash
python bulk_load_data.py
```

This loads synthetic patient cases into Qdrant for testing.

---

## 🚀 Running the System

### Option 1: Full FastAPI Backend
Start the complete system with web interface:

```bash
uvicorn backend.app.main:app --reload --port 8000
```
The API will be available at `http://localhost:8000`.
API Documentation: `http://localhost:8000/docs`.

### Option 2: Run 3-Agent Pipeline (Demo)
Test the first 3 agents (Ingestion + Memory + Context Builder):

```bash
python three_agent_pipeline.py
```

### Option 3: Run Individual Agents
For development and testing:

```python
from agents import field_ingestion_agent
from patient_memory_agent import patient_memory_agent

# Agent 1
state = {
    "patient_hash": "demo_patient",
    "raw_text": "Patient has skin rash on arms. Itching reported."
}
result = field_ingestion_agent(state)

# Agent 2
result_2 = patient_memory_agent({"patient_data": result["patient_data"]})
```

## 💻 Technology Stack

### Core Framework
- **FastAPI** - RESTful API backend
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation

### AI & Machine Learning
- **Sentence Transformers** - Text embeddings (all-MiniLM-L6-v2)
- **CLIP** - Image embeddings (ViT-B/32)
- **Whisper** - Audio transcription (OpenAI)
- **Google Gemini** - LLM for explanations
- **PyTorch** - Deep learning framework

### Vector Database
- **Qdrant** - Vector similarity search

### Data Processing
- **Pillow** - Image processing
- **NumPy, SciPy** - Numerical operations
- **Tesseract** - OCR (optional)
- **PyMuPDF** - PDF processing (optional)

### Storage
- **SQLite** - Relational database for structured data

---

## 📊 Usage Examples

### Example 1: Process Patient Data via API

```bash
curl -X POST http://localhost:8000/process \
  -F "patient_hash=patient_123" \
  -F "text_notes=Patient reports itching and rash on arms." \
  -F "images=@lesion_photo.jpg"
```

### Example 2: Run 3-Agent Pipeline Programmatically

```python
from my_code.three_agent_pipeline import run_three_agent_pipeline

result = run_three_agent_pipeline(
    patient_hash="demo_patient_001",
    raw_text="Patient has skin rash on arms. Itching reported. No fever.",
    image_paths=["./uploads/lesion1.jpg"]
)

print(f"Event ID: {result['event_id']}")
print(f"Total Visits: {result['patient_current_state']['total_visits']}")
print(f"Retrieval Plan: {result['retrieval_plan']}")
```

### Example 3: Query Patient Memory Directly

```python
from my_code.qdrant_manager import QdrantManager

qm = QdrantManager()
history = qm.get_patient_history("patient_123", limit=5)

for event in history:
    print(f"Visit on {event['timestamp']}: {event['text_summary']}")
```

---

## 🔐 Security & Privacy

- **Patient anonymization**: All patient identifiers are hashed
- **Offline capability**: System works without internet (except for LLM calls)
- **Data isolation**: SQLite and Qdrant ensure proper data separation
- **Safety constraints**: Agents never provide medical diagnosis or advice

---

## 🧪 Testing

Run the health check:
```bash
curl http://localhost:8000/health
```

Test with sample data:
```bash
cd my_code
python three_agent_pipeline.py
```

---

## 📝 License

See [LICENSE](file:///d:/convolve/Convolve-4.0-Submission-2Slow2Serious/LICENSE) file for details.

---

## 🤝 Contributing

This is a healthcare AI research project. For questions or contributions, please contact the development team.

---

## 🙏 Acknowledgments

Built for **Convolve 4.0 Hackathon** by **Team 2Slow2Serious**.

Designed for rural healthcare workers to leverage AI-powered collective memory for better patient care.
