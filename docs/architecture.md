# System Architecture

## Purpose
This document describes the high-level architecture of the Claims Processing & Settlement system, including major modules, request flow, and folder responsibilities.

## Major Modules

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app/frontend/)              │
│  Chat | HITL Review | Prompt Versions | Evaluation Dashboard │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / REST
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 FastAPI Server (app/api/server.py)           │
│  /chat  /hitl/*  /prompts/*  /eval/*  /auth/*  /retrieve    │
└──────────┬──────────┬──────────┬──────────┬─────────────────┘
           │          │          │          │
           ▼          ▼          ▼          ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
   │ LCEL     │ │ HITL     │ │ Prompt   │ │ RBAC         │
   │ Router   │ │ Manager  │ │ Registry │ │ Permission   │
   │ chains/  │ │ hitl/    │ │ prompts/ │ │ Matrix       │
   └────┬─────┘ └──────────┘ └──────────┘ └──────┬───────┘
        │                                         │
        ▼                                         ▼
   ┌──────────┐                          ┌──────────────┐
   │ MCP      │                          │ RAG Pipeline │
   │ Client   │                          │ rag/         │
   │ Pool     │                          └──────┬───────┘
   └──────────┘                                 │
                                                ▼
                                         ┌──────────────┐
                                         │ Vector Store  │
                                         │ (FAISS)      │
                                         └──────────────┘
```

## Key Components

### 1. API Layer (`app/api/server.py`)
- FastAPI application with endpoints: `/health`, `/chat`, `/reset`, `/history`, `/hitl/*`, `/prompts/*`, `/eval/*`, `/auth/*`, `/retrieve`, `/roles`, `/sources`
- Integrates with LCEL router, HITL manager, prompt registry, RBAC, and RAG pipeline
- Initializes MCP client pool, prompt registry, and vector store on startup

### 2. LCEL Chain Orchestration (`app/chains/`)
- **`router.py`** — Top-level `lcel_router` (Runnable) with intent-based `RunnableBranch`
- **`rag_chain_lcel.py`** — Wraps knowledge retrieval as a Runnable
- **`tool_chain_lcel.py`** — Tool dispatch chain
- **`hitl_chain.py`** — HITL pause/resume chain
- **`base_lcel.py`** — Shared helpers: `make_retryable()`, `make_fallback_chain()`, `build_run_dict()`
- All chains use `RunnableLambda`, `RunnablePassthrough`, `RunnableBranch`

### 3. MCP Integration (`app/mcp/`)
- **`registry.py`** — `MCPServerRegistry` loads from `config/mcp_servers.yaml`
- **`client.py`** — `MCPClient` with HTTP transport, health checks, retry-with-backoff
- **`auth.py`** — Auth header builders (none, api_key, bearer, basic)
- **`tool_adapter.py`** — Discovers and creates LangChain tools from MCP servers
- **Pool** — `MCPClientPool` singleton manages one client per server

### 4. HITL Workflow (`app/hitl/`)
- **`triggers.py`** — Rule evaluation engine loading from `config/hitl_rules.yaml`
- **`manager.py`** — `HITLManager` with pause/resume lifecycle
- **`store.py`** — SQLite-backed `HITLTaskStore` for persistent task storage
- **`models.py`** — `HITLTask`, `HITLTriggerResult` dataclasses

### 5. RBAC (`app/rbac/`)
- **`auth.py`** — JWT token creation/decoding (HS256)
- **`models.py`** — `PermissionMatrix` singleton loading from `config/roles.yaml`
- **`filter.py`** — Pre-retrieval metadata filter, top-k clamp
- **`validator.py`** — Post-retrieval validation (0% leakage guarantee)
- **`audit.py`** — Audit logging for every retrieval

### 6. Prompt Management (`app/prompts/`, `app/prompt_manager/`)
- **YAML templates** in `config/prompts/` (6 files, versioned)
- **`registry.py`** — `PromptRegistry` with versioned prompt storage
- **`loader.py`** — YAML file reader
- **`prompts/loader.py`** — Backward-compatible loader delegating to registry

### 7. RAG Pipeline (`app/rag/`)
- **`embeddings.py`** — Embedding function
- **`vectorstores/`** — Vector store implementations (FAISS)
- **`retriever_hybrid.py`** — Hybrid retrieval (dense + sparse)
- **`reranker.py`** — Cross-encoder reranking
- **`qa_chain.py`** — QA chain with citation support

### 8. Evaluation (`eval/`)
- **`regression_suite.py`** — Regression runner against golden set
- **`custom_metrics.py`** — 5 custom metrics (golden set pass rate, answer stability, regulatory compliance, role appropriateness, HITL trigger precision)
- **`dashboard.py`** — Trend data preparation for Streamlit dashboard
- **`intrinsic.py`** — Intrinsic retrieval metrics
- **`extrinsic.py`** — Extrinsic answer quality metrics
- **`llm_judge.py`** — LLM-as-judge evaluation

### 9. Callbacks (`app/callbacks/`)
- **`logging_cb.py`** — `LoggingCallbackHandler` for structured JSON logging
- **`metrics_cb.py`** — `MetricsCallbackHandler` for latency/count metrics
- **`tracing_cb.py`** — LangSmith trace callback

### 10. Streamlit UI (`app/frontend/streamlit_app.py`)
- Tabbed interface: Chat, HITL Review, Prompt Versions, Evaluation Dashboard
- Role selector with JWT token management
- Citation-aware chat bubbles

## End-to-End Request Flow

```
User Message
    │
    ▼
FastAPI /chat endpoint
    │
    ├── LoggingCallbackHandler  (structured logging)
    ├── MetricsCallbackHandler  (latency/count metrics)
    ├── TracingCallbackHandler  (LangSmith trace)
    │
    ▼
lcel_router.invoke()
    │
    ├── classify_node (RunnableLambda — intent detection via FAQChain)
    │   └── Attaches _resolved_intent, _faq_confidence
    │
    ├── RunnableBranch
    │   ├── rag → rag_lcel_chain (knowledge retrieval → answer)
    │   ├── hitl → hitl_lcel_chain (evaluate triggers → pause if matched)
    │   └── tool → tool_lcel_chain (tool dispatch with MCP)
    │       └── ─── hitl_lcel_chain (post-tool HITL check)
    │
    └── _post_process (strips internal keys)
    │
    ▼
API response to caller
```

## Key Folder Responsibilities

| Folder | Responsibility |
|--------|---------------|
| `app/api/` | FastAPI server, REST endpoints |
| `app/chains/` | LCEL chain orchestration |
| `app/mcp/` | MCP client and server registry |
| `app/hitl/` | Human-in-the-loop workflow |
| `app/rbac/` | Role-based access control |
| `app/prompts/` | Prompt loading (backward-compat) |
| `app/prompt_manager/` | Versioned prompt registry |
| `app/callbacks/` | LangChain callback handlers |
| `app/rag/` | RAG pipeline (retrieval, reranking, QA) |
| `app/agents/` | Agent orchestration |
| `app/frontend/` | Streamlit UI |
| `app/memory/` | SQLite-backed conversation memory |
| `config/` | YAML configuration files |
| `eval/` | Evaluation framework |
| `tests/` | Pytest test suite |
| `data/` | SQLite databases, FAISS index |
| `docs/` | Documentation |
| `reports/` | Evaluation reports |
| `scripts/` | Utility and validation scripts |

## Configuration Files

| File | Purpose |
|------|---------|
| `config/mcp_servers.yaml` | MCP server definitions (4 servers, 8 tools) |
| `config/hitl_rules.yaml` | HITL trigger rules (5 rules) |
| `config/roles.yaml` | RBAC role definitions (4 roles) |
| `config/agents.yaml` | Agent descriptors |
| `config/prompts/*.yaml` | Prompt templates (6 files, versioned) |
| `config/drift_thresholds.yaml` | Drift detection thresholds |
| `.env` | Environment variables (JWT_SECRET, API keys) |
| `requirements.txt` | Python dependencies |
| `docker-compose.yml` | Docker composition |