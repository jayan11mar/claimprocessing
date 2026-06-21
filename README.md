# Claims Processing & Settlement Automation Assistant

This repository contains an insurance claims FAQ assistant with LangChain-based automation, FastAPI backend, Streamlit frontend, and custom domain tools.

## Setup

1. Create a Python virtual environment and activate it.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Copy the example environment file:

```bash
cp .env.example .env
```

4. Update `.env` with your OpenAI API key and environment settings.

## Run the Streamlit app

```bash
streamlit run frontend/streamlit_app.py
```

## Run the backend API

```bash
uvicorn main:app --reload
```

The backend exposes:

- `GET /health`
- `POST /chat`
- `POST /reset`

## Run the frontend client

```bash
streamlit run frontend/streamlit_app.py
```

## Run tests

Use the repository root on `PYTHONPATH` so imports from `app/` resolve correctly:

```bash
PYTHONPATH=. pytest -q
```

## Evaluate end-to-end behavior

An evaluation harness is included at `scripts/evaluate.py`.
It sends 20 sample queries covering FAQ, tool usage, and multi-turn conversation behavior to the running `/chat` endpoint and saves the results to `scripts/results.json`.

```bash
python scripts/evaluate.py
```

## Architecture

The system is built as a modular FastAPI + Streamlit application with LangChain orchestration, custom domain tools, SQLite memory, Pydantic validation, JSON logging, and optional LangSmith tracing.

```
User Browser / Streamlit
          |
          v
    Streamlit client
          |
          v
     FastAPI backend
      /    |    \
     v     v     v
   Memory  Agent  LangSmith
   (SQLite) Chain  Tracing
      |      |
      v      v
   Tools    LLM
 (claims,    +
 fraud,    prompt
 settlement) templates
```

## Documentation

- `docs/evaluation_report.md`: evaluation summary of the 20 queries.

## Project layout

- `app/`: application modules
- `frontend/streamlit_app.py`: chat UI client
- `scripts/evaluate.py`: evaluation harness
- `docs/evaluation_report.md`: evaluation report
- `.env.example`: environment variable template
- `requirements.txt`: dependencies
