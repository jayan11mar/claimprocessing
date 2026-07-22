# Deployment Guide

## Purpose
This document describes how to set up, configure, and run the Claims Processing & Settlement system locally, including the FastAPI backend, Streamlit frontend, MCP servers, test suite, and Docker deployment.

## Local Environment Setup

### Prerequisites
- Python 3.9+
- OpenAI API key (for LLM and embeddings)
- Git

### Step 1: Clone and Navigate

```bash
git clone <repository-url> /path/to/claimprocessing
cd /path/to/claimprocessing
```

### Step 2: Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Note:** `requirements.txt` currently lists packages without pinned versions. All packages install at the latest compatible versions at time of installation. For reproducible builds, consider adding version pins (e.g., `fastapi==0.115.0`).

### Step 4: Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with at minimum:
```env
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL_NAME=gpt-4o-mini
VECTOR_BACKEND=faiss
JWT_SECRET=your_jwt_secret_here
```

Optional feature flags (set to `true` to enable):
```env
ENABLE_LCEL=true
ENABLE_MULTI_AGENT=true
ENABLE_MCP=true
ENABLE_HITL=true
ENABLE_RBAC=true
ENABLE_AGENTS=true
ENABLE_DRIFT=true
ENABLE_PROMPT_MANAGER=true
```

### Step 5: Initialize Data (if needed)

The system creates SQLite databases and FAISS indexes automatically on first run:
- `data/claims.db` — Claims data
- `data/hitl_tasks.db` — HITL task store (created by `HITLTaskStore.__init__()`)
- `data/faiss_index` — FAISS vector index
- `data/faiss_index.meta.json` — FAISS metadata

## Running the Application

### Start the FastAPI Backend

```bash
cd /path/to/claimprocessing
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is available at `http://127.0.0.1:8000`.

Key endpoints:
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (model, temperature, DB status, uptime) |
| POST | `/chat` | Chat completion |
| GET | `/hitl/pending` | List pending HITL tasks |
| POST | `/hitl/review/{task_id}` | Approve/reject HITL task |
| GET | `/roles` | List RBAC roles |
| GET | `/auth/context` | Get current auth context |
| POST | `/auth/token` | Create JWT token |
| GET | `/prompts` | List prompt templates |
| POST | `/eval/regression` | Run regression evaluation |

### Start the Streamlit Frontend

```bash
cd /path/to/claimprocessing
source venv/bin/activate
streamlit run app/frontend/streamlit_app.py --server.port 8501
```

The UI is available at `http://127.0.0.1:8501`.

The Streamlit app has four tabs:
1. **💬 Chat** — Conversational claims assistant
2. **🛂 HITL Review** — Approve/reject pending HITL tasks
3. **📝 Prompt Versions** — Browse and activate prompt template versions
4. **📊 Evaluation Dashboard** — View regression metrics and trends

### Configure Backend URL in Streamlit

In the Streamlit sidebar, set the "Backend URL" to `http://127.0.0.1:8000` (default).

## Running Tests

### Run All Tests

```bash
cd /path/to/claimprocessing
source venv/bin/activate
PYTHONPATH=. python -m pytest -v
```

### Run Tests by Deliverable

| Deliverable | Test Command |
|-------------|-------------|
| MCP Integration | `PYTHONPATH=. python -m pytest tests/test_mcp_integration.py -v` |
| HITL Workflow | `PYTHONPATH=. python -m pytest tests/test_hitl_workflow.py -v` |
| RBAC / Zero Leakage | `PYTHONPATH=. python -m pytest tests/test_role_based_rag.py -v` |
| Prompt Management | `PYTHONPATH=. python -m pytest tests/test_prompts_loader.py tests/test_prompt_versioning.py -v` |
| LCEL Chains | `PYTHONPATH=. python -m pytest tests/test_chains.py tests/test_base_chain.py -v` |
| Evaluation | `PYTHONPATH=. python -m pytest tests/test_eval_package.py tests/test_rag_evaluation_harness.py -v` |
| RAG Pipeline | `PYTHONPATH=. python -m pytest tests/test_rag_pipeline.py tests/test_rag_integration.py -v` |

### Run Evaluation Scripts

```bash
# Full evaluation
PYTHONPATH=. python scripts/evaluate.py

# Golden dataset validation
PYTHONPATH=. python scripts/validate_golden_dataset.py

# RAG evaluation
PYTHONPATH=. python scripts/evaluate_rag.py

# End-to-end RAG validation
PYTHONPATH=. python scripts/end_to_end_rag_validation.py
```

## MCP Server Startup

The MCP module expects external HTTP servers for tool execution. These are defined in `config/mcp_servers.yaml` but **not provided** in this repository — they are external services.

| Server | URL | Tools | Auth |
|--------|-----|-------|------|
| Hospital Network | `http://127.0.0.1:9001` | check_hospital_network, get_hospital_details | None |
| Policy Admin | `http://127.0.0.1:9002` | get_policy_details, check_claim_eligibility | API Key |
| Fraud Detection | `http://127.0.0.1:9003` | score_fraud_risk, get_fraud_signals | Bearer |
| IRDAI Compliance | `http://127.0.0.1:9004` | check_compliance_status, get_reporting_requirements | Basic |

If these servers are unavailable, MCP tool calls will fail gracefully with timeout/retry errors. The system will continue functioning for non-MCP operations.

## Docker Deployment

The project includes Docker support for containerised deployment.

### Docker Compose

```bash
docker compose up --build
```

This starts:
- FastAPI backend (port 8000)
- Streamlit frontend (port 8501)

### Dockerfile

The Dockerfile is at `docker/Dockerfile` and builds a container image with all dependencies.

### Configuration Files
- `docker-compose.yml` — Multi-service orchestration
- `docker/Dockerfile` — Container image definition
- `.dockerignore` — Docker context exclusions

### Build Single Image

```bash
docker build -f docker/Dockerfile -t claims-processing:latest .
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key (required) |
| `OPENAI_MODEL_NAME` | `gpt-4o-mini` | Model for LLM calls |
| `VECTOR_BACKEND` | `faiss` | Vector store backend |
| `JWT_SECRET` | — | Secret for JWT token signing |
| `LANGSMITH_TRACING` | `false` | Enable LangSmith tracing |
| `LANGSMITH_API_KEY` | — | LangSmith API key |
| `LANGSMITH_PROJECT` | — | LangSmith project name |
| `ENABLE_LCEL` | `false` | Enable LCEL chains |
| `ENABLE_MCP` | `false` | Enable MCP integration |
| `ENABLE_HITL` | `false` | Enable HITL workflow |
| `ENABLE_RBAC` | `false` | Enable RBAC |
| `ENABLE_AGENTS` | `false` | Enable agent subsystem |
| `ENABLE_DRIFT` | `false` | Enable drift detection |
| `ENABLE_PROMPT_MANAGER` | `false` | Enable versioned prompt management |
| `HITL_STORE_PATH` | `data/hitl_tasks.db` | HITL SQLite database path |
| `MCP_SERVERS_PATH` | `config/mcp_servers.yaml` | MCP server config path |
| `HITL_RULES_PATH` | `config/hitl_rules.yaml` | HITL rules config path |

## VM / Disk Space Considerations

- **FAISS index:** `data/faiss_index` — ~12 KB (grows with knowledge base size)
- **SQLite databases:** `data/claims.db`, `data/hitl_tasks.db` — typically < 100 MB each
- **Virtual environment:** `venv/` — typically 200-500 MB
- **Embedding models:** `sentence-transformers` models download to `~/.cache/huggingface/` — typically 200-300 MB
- **Evaluation reports:** `reports/` — typically < 10 MB

For production deployments:
- Ensure sufficient disk space for the vector index and knowledge base documents
- Consider using a dedicated vector database (Pinecone, Chroma) instead of FAISS for scalability
- Use environment-specific configuration files rather than `.env`
- Enable LangSmith tracing for production monitoring

## Troubleshooting

### Backend Unavailable
If the Streamlit UI shows "Backend health unavailable":
1. Verify FastAPI is running: `curl http://127.0.0.1:8000/health`
2. Check the backend URL in the Streamlit sidebar
3. Ensure `OPENAI_API_KEY` is set in `.env`

### MCP Tool Failures
If MCP tool calls fail:
1. Verify MCP servers are running on their respective ports
2. Check `config/mcp_servers.yaml` for correct URLs
3. Set `ENABLE_MCP=true` in `.env`

### HITL Not Showing Tasks
1. Set `ENABLE_HITL=true` in `.env`
2. Verify `data/hitl_tasks.db` exists
3. Check FastAPI startup logs for HITL initialization

### Test Failures
1. Ensure virtual environment is activated
2. Install test dependencies: `python -m pip install pytest pytest-cov`
3. Set `PYTHONPATH=.` before running tests
4. Some tests require `OPENAI_API_KEY` to be set