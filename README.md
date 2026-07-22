# Claims Processing & Settlement Assistant

This repository contains a claims support assistant for insurance workflows. It combines a FastAPI backend, a Streamlit frontend, LangChain-based orchestration, custom domain tools, and a retrieval-augmented knowledge base for policy and claims guidance.

## Project features

- **Conversational claims assistant** with FAQ handling and tool-based workflows
- **Domain tools** — policy status, claim status, claims intake, fraud scoring, and settlement calculation
- **Retrieval-Augmented Generation (RAG)** over a knowledge base with hybrid retrieval, reranking, and citations
- **Multi-vector-store backends** — FAISS, Chroma, and Pinecone support
- **SQLite-backed conversation memory** for multi-turn sessions
- **LangSmith tracing**, JSON logging, guardrails, and correlation-aware request handling
- **Evaluation and regression scripts** for end-to-end behavior and golden datasets
- **Multi-agent orchestration** — task decomposer, dispatcher, aggregator, and agent registry
- **Human-in-the-Loop (HITL)** — configurable rules, task store, review triggers, and escalation management
- **Role-Based Access Control (RBAC)** — role models, authentication, filters, validators, and audit logging
- **Drift detection** — embedding drift, prompt drift, baseline computation, drift alerts, and detector
- **Model Context Protocol (MCP)** — MCP client, server registry, tool adapters, and authentication
- **Prompt management** — versioned prompt registry, loader, validator, and template support
- **LCEL (LangChain Expression Language)** chain support for composable pipelines

## Architecture at a glance

| Layer | Location |
|-------|----------|
| Backend API | `app/api/server.py` |
| Application entrypoint | `app/main.py` |
| Orchestration | `app/chains/` |
| RAG pipeline | `app/rag/` |
| Domain tools | `app/tools/` |
| Multi-agent system | `app/agents/` |
| Human-in-the-Loop | `app/hitl/` |
| RBAC & audit | `app/rbac/` |
| Drift detection | `app/drift/` |
| MCP integration | `app/mcp/` |
| Prompt manager | `app/prompt_manager/` |
| Configuration | `config/` |
| Frontend UI | `app/frontend/streamlit_app.py` |
| Data & evaluation | `data/`, `docs/`, `eval/`, `scripts/`, `tests/` |

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

4. Update `.env` with your configuration, including at minimum:

```env
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL_NAME=gpt-4o-mini
VECTOR_BACKEND=faiss
```

### Optional feature flags (Week-8+)

The following environment variables enable optional subsystems:

```env
ENABLE_LCEL=true           # LangChain Expression Language chains
ENABLE_MULTI_AGENT=true    # Multi-agent orchestration
ENABLE_MCP=true            # Model Context Protocol integration
ENABLE_HITL=true           # Human-in-the-Loop workflows
ENABLE_RBAC=true           # Role-based access control
ENABLE_AGENTS=true         # Agent subsystem
ENABLE_DRIFT=true          # Drift detection monitoring
ENABLE_PROMPT_MANAGER=true # Versioned prompt management
```

## Configuration

The `config/` directory contains YAML configuration files for each optional subsystem:

| File | Purpose |
|------|---------|
| `agents.yaml` | Agent definitions and routing rules |
| `drift_thresholds.yaml` | Thresholds for embedding/prompt drift alerts |
| `hitl_rules.yaml` | HITL trigger rules and escalation policies |
| `mcp_servers.yaml` | MCP server definitions and connection settings |
| `roles.yaml` | RBAC role definitions and permission mappings |
| `prompts/` | Versioned prompt templates |

## Run the application

### Start the API server

```bash
uvicorn app.main:app --reload
```

The API will be available at http://127.0.0.1:8000.

### Start the Streamlit frontend

```bash
streamlit run app/frontend/streamlit_app.py
```

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/chat` | Chat completion (supports RAG, HITL, agents) |
| POST | `/reset` | Reset conversation session |
| POST | `/retrieve` | Retrieve documents from the knowledge base |
| POST | `/ingest` | Ingest documents into the knowledge base |
| GET | `/sources` | List available knowledge base sources |
| GET | `/history/{session_id}` | Retrieve conversation history |
| POST | `/evaluate` | Run evaluation against golden datasets |

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

Additional validation helpers:

| Script | Purpose |
|--------|---------|
| `scripts/validate_memory_continuity.py` | Validate multi-turn memory continuity |
| `scripts/verify_ingestion.py` | Verify knowledge base ingestion |
| `scripts/diagnose_retrieval_path.py` | Diagnose RAG retrieval issues |
| `scripts/evaluate_rag.py` | RAG-specific evaluation |
| `scripts/end_to_end_rag_validation.py` | Full pipeline RAG validation |
| `scripts/verify_langsmith_traces.py` | Verify LangSmith trace output |
| `scripts/verify_rag_pipeline_with_langsmith.py` | RAG pipeline trace verification |
| `scripts/verify_prompt_migration.py` | Prompt version migration validation |
| `scripts/generate_eval_golden_sets.py` | Generate golden evaluation datasets |
| `scripts/validate_generated_datasets.py` | Validate generated datasets |
| `scripts/validate_sqlite_context.py` | Validate SQLite context persistence |
| `scripts/add_roles_to_golden_set.py` | Add RBAC roles to golden datasets |
| `scripts/demo_policy_status.py` | Demo policy status tool |

## Project structure

```
├── app/                    # Application modules
│   ├── agents/             # Multi-agent orchestration (decomposer, dispatcher, aggregator, registry)
│   ├── api/                # FastAPI server
│   ├── chains/             # LangChain orchestration (RAG, FAQ, HITL, tool, LCEL, router)
│   ├── drift/              # Drift detection (embedding, prompt, alerts, baseline)
│   ├── frontend/           # Streamlit chat UI
│   ├── hitl/               # Human-in-the-Loop (manager, store, triggers)
│   ├── mcp/                # Model Context Protocol (client, registry, adapters, auth)
│   ├── memory/             # SQLite-backed conversation memory
│   ├── prompt_manager/     # Versioned prompt management (loader, registry, validator)
│   ├── rbac/               # Role-based access control (auth, filter, validator, audit)
│   ├── rag/                # RAG pipeline (retrievers, rerankers, chunking)
│   ├── tools/              # Domain tools (policy, claims, fraud, settlement)
│   ├── callbacks/          # Logging, metrics, and tracing callbacks
│   ├── logging/            # Structured JSON logging
│   └── models/             # Domain models and schemas
├── config/                 # Configuration files (agents, drift, HITL, MCP, roles, prompts)
├── data/                   # Knowledge base, fixtures, indexes, and databases
├── docs/                   # Reference and evaluation documentation
├── eval/                   # Evaluation utilities, golden datasets, reporting
├── scripts/                # Operational, validation, and demo scripts
└── tests/                  # Automated test coverage
```

## Deployment

The project includes Docker support for containerised deployment:

```bash
docker compose up --build
```

This starts both the API server and the Streamlit frontend. See `docker/README.md` for detailed build/run instructions.

- `docker/Dockerfile` — container image definition
- `docker-compose.yml` — multi-service orchestration
- `.dockerignore` — Docker context exclusions

## Notes

- The default vector backend is **FAISS**, but configuration supports Chroma or Pinecone.
- SQLite is used for session memory, chat history, and HITL task persistence.
- Optional LangSmith tracing can be enabled through environment variables.
- The MCP subsystem allows connecting to external tool servers via the Model Context Protocol.
- Drift detection helps monitor embedding distribution shifts and prompt changes over time.
- RBAC enforces role-based access policies with full audit logging.