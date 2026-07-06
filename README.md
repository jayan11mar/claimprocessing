# Claims Processing & Settlement Assistant

This repository contains a claims support assistant for insurance workflows. It combines a FastAPI backend, a Streamlit frontend, LangChain-based orchestration, custom domain tools, and a retrieval-augmented knowledge base for policy and claims guidance.

## Project updates

The project now includes:

- A conversational claims assistant with FAQ handling and tool-based workflows
- Policy status, claim status, claims intake, fraud scoring, and settlement calculation tools
- Retrieval-augmented generation (RAG) over a knowledge base with hybrid retrieval and citations
- Multiple vector-store backends including FAISS, Chroma, and Pinecone support
- SQLite-backed conversation memory for multi-turn sessions
- LangSmith tracing, JSON logging, guardrails, and correlation-aware request handling
- Evaluation and regression scripts for end-to-end behavior and golden datasets

## Architecture at a glance

- Backend API: FastAPI service in app/api/server.py
- Application entrypoint: app/main.py
- Orchestration layer: app/chains/
- RAG pipeline: app/rag/
- Domain tools: app/tools/
- Frontend UI: frontend/streamlit_app.py
- Data and evaluation assets: data/, docs/, eval/, scripts/, tests/

## Prerequisites

- Python 3.9+
- OpenAI API access for LLM and embedding usage

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Copy the example environment file:

```bash
cp .env.example .env
```

4. Update .env with your configuration, including at minimum:

```env
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL_NAME=gpt-4o-mini
VECTOR_BACKEND=faiss
```

## Run the application

### Start the API server

```bash
uvicorn app.main:app --reload
```

The API will be available at http://127.0.0.1:8000.

### Start the Streamlit frontend

```bash
streamlit run frontend/streamlit_app.py
```

## API endpoints

The backend exposes:

- GET /health
- POST /chat
- POST /reset
- POST /retrieve
- POST /ingest
- GET /sources
- GET /history/{session_id}
- POST /evaluate

## Evaluation and validation

Run the test suite:

```bash
PYTHONPATH=. pytest -q
```

Run the end-to-end evaluation harness:

```bash
python scripts/evaluate.py
```

Run golden-dataset regression validation:

```bash
python scripts/validate_golden_dataset.py
```

Additional validation helpers are available in the scripts/ folder for memory continuity, ingestion verification, and RAG evaluation.

## Project structure

- app/: application modules and business logic
- frontend/: Streamlit chat UI
- data/: knowledge base, fixtures, and local indexes
- docs/: reference and evaluation documents
- eval/: evaluation utilities and reporting
- scripts/: operational and validation scripts
- tests/: automated test coverage

## Notes

- The default vector backend is FAISS, but the configuration supports Chroma or Pinecone.
- SQLite is used for session memory and chat history.
- Optional LangSmith tracing can be enabled through environment variables.
