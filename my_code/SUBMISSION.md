# Convolve 4.0 Hackathon Submission

## Team: 2Slow2Serious

## Project: Rural Healthcare Field Ingestion System

### Problem Statement
Rural healthcare workers need quick access to historical case data when treating patients with limited diagnostic tools.

### Solution
An AI-powered multi-modal system that:
- Processes text notes, audio, images, and documents
- Stores patient events in a vector database
- Retrieves similar historical cases
- Provides actionable recommendations via chat interface

### Technical Highlights
- **6-Agent LangGraph Pipeline** - Modular, scalable workflow
- **Hybrid Vector Search** - Dense (CLIP/BioBERT) + Sparse (BM25)
- **Multi-Modal Embeddings** - Text, Image, Audio processing
- **Safety Constraints** - No diagnosis, only experience recall
- **Uncertainty Handling** - Flags sparse/noisy data

### Tech Stack
- FastAPI + LangGraph
- Qdrant Cloud (Vector DB)
- Google Gemini 2.0 Flash
- SQLite + File Storage

### Key Features
1. **Multi-modal Input** - Text, audio, images, PDFs
2. **Vector Search** - Find similar cases from 1000s of historical records
3. **Chat Interface** - Ask questions about specific patients
4. **Follow-up Detection** - Automatic gap identification

### Demo Flow
1. Healthcare worker uploads patient photo + notes
2. System extracts features and searches similar cases
3. Returns 3-5 most relevant historical cases
4. Worker can chat to ask "What should I do?"
5. System provides recommendations based on past outcomes

### Code Structure
See `README.md` for complete architecture and setup.

### Run Instructions
```bash
# Setup
cp .env.example .env  # Add API keys
pip install -r requirements.txt

# Run
python api.py

# Open
http://localhost:8000
```

### Future Enhancements
- Mobile app for offline usage
- Voice-first interface for low-literacy users
- Integration with government health platforms
- Multi-language support (Hindi, regional languages)

## Contact
Team 2Slow2Serious
Convolve 4.0 - 2025
