# Gen AI Capstone Plan - Claims Processing & Settlement

> **Usage with GitHub Copilot**  
> Implement this project **phase-by-phase**. For each phase:
> - Work only within that phase’s section until its acceptance criteria are met.  
> - Select the relevant checklist items and ask Copilot to generate or update code for those items only.  
> - Do not skip phases or merge their tasks; each phase is meant to be independently runnable.

---

## Project overview

This project implements an Insurance **Claims Processing & Settlement** solution with an FAQ chatbot and agentic tools for claims intake, fraud signals, and settlement calculations.

Architecture stack:
- **LLM**: OpenAI-compatible ChatCompletion API.
- **Orchestration**: LangChain (chains, tools, memory).
- **Memory**: SQLite-based conversational memory (at least 10 turns).
- **Backend**: FastAPI with `/chat`, `/reset`, `/health`.
- **Frontend**: Streamlit chat UI calling FastAPI.
- **Validation & models**: Pydantic.
- **Observability**: Structured JSON logging and LangSmith traces.
- **RAG**: Hybrid retrieval (BM25 + dense) with re-ranking, multi-format document loaders, and evaluation harness.

This plan covers **both Week 2 and core assistant deliverables**:
- Week 2: Insurance FAQ chatbot with prompt engineering, few-shot learning, CoT, structured JSON outputs, guardrails, and simple UI.
- core assistant: Full LangChain-based app with memory, tools, FastAPI backend, Streamlit frontend, logging, tests, and evaluation.
- RAG extension: Document-backed knowledge layer with hybrid retrieval, QA chain with citations, and evaluation harness.

---

## Requirement → Phase mapping

| Requirement                                                                                       | Phase(s)                |
|---------------------------------------------------------------------------------------------------|-------------------------|
| Week 2: Insurance FAQ Chatbot (prompt engg, few-shot, CoT, JSON, guardrails, simple UI)          | 1, 2                    |
| LangChain-based architecture with clean chain composition                                         | 3, 4, 5                 |
| Conversational memory (SQLite, ≥10 turns)                                                         | 3, 4                    |
| At least 3 custom tools (claims intake, fraud, settlement)                                       | 5                       |
| Few-shot prompt templates with semantic example selection                                        | 3, 5                    |
| Structured output parsing with Pydantic validation                                               | 2, 3, 5                 |
| FastAPI backend with /chat, /reset, /health                                                      | 4                       |
| Streamlit frontend with chat interface (calling FastAPI)                                         | 1, 2, 4                 |
| Error handling with retry logic and fallback chains + LangSmith traces                           | 4, 6                    |
| Structured JSON logging for all interactions                                                     | 4, 6                    |
| Response latency targets (<3s simple, <8s tool-augmented)                                       | 4, 6                    |
| Modular codebase (tools, chains, prompts, API layers separated)                                  | 0, 3, 4, 5              |
| Environment-based configuration via .env                                                         | 0                       |
| Unit tests for each custom tool with ≥80% coverage                                               | 6                       |
| Prompt template library (JSON)                                                                   | 2, 3                    |
| Evaluation report: 20 queries (FAQ, tools, multi-turn)                                           | 7                       |
| Golden dataset and regression validation for future RAG/phase/tool changes                      | 7A                      |
| SQL domain persistence for policies/claims                                                      | 8, 9                    |
| Fraud-gated claim registration                                                                  | 9                       |
| RAG: Manifest-driven loaders, chunking, embeddings, basic retriever                             | 10                      |
| RAG: Vector backends and metadata schema selection                                              | 11                      |
| RAG: Hybrid retrieval and QA chain                                                              | 12                      |
| RAG: Evaluation harness and acceptance thresholds                                               | 13                      |
| RAG: Agent integration, new endpoints, and tool routing                                         | 14                      |
| RAG: Acceptance criteria validation and sign-off                                                | 15                      |
| Reviewer remediation (persistent vector, hybrid, cross-encoder rerank, grounded citations)      | R1–R6                   |
| Week 8: Bootstrap scaffolding (packages, config/*.yaml, .env flags, deps)                       | W8-0                    |
| Week 8: LangChain orchestration → LCEL chains, callbacks, chain registry/router                 | W8-1                    |
| Week 8: Prompt version control via YAML + /prompts* endpoints                                   | W8-2                    |
| Week 8: MCP integration (client, registry, auth, adapter) + /mcp/* endpoints                    | W8-3                    |
| Week 8: HITL workflows (manager, triggers, persistent store) + /hitl/* endpoints                | W8-4                    |
| Week 8: Role-Based RAG (JWT auth, pre+post filter, audit) + /roles, /auth/context               | W8-5                    |
| Week 8: Expanded evaluation (regression suite, custom metrics, comparator) + /eval/*            | W8-6                    |
| Week 8: Streamlit UI (HITL panel, role selector, prompt viewer, eval dashboard)                 | W8-7                    |
| Week 8: Containerisation (Dockerfile + docker-compose) [optional-advantage]                     | W8-8                    |
| Week 8: A2A / multi-agent orchestration [optional/bonus]                                        | W8-9                    |
| Week 8: Prompt & embedding drift detection [optional/bonus]                                     | W8-10                   |
| Week 8: Extensive validation — full regression + Section-4 acceptance thresholds                | W8-V                    |
| Week 8: requirements.txt verification + deployment cleanup (diagnostics/plan removal)           | W8-C                    |

---

## High-level review of current phases (post-update)

This review reflects the current implementation state after all phases have been completed.

- Phase 0-2: scaffolding, FAQ workflow, guardrails, and prompt library are in place and runnable.
- Phase 3: LangChain, SQLite memory, and the initial agent flow are present with working storage layer.
- Phase 3A: Multi-turn memory reliability hardening added with regression tests and validation scripts.
- Phase 4: FastAPI chat/reset/health endpoints are implemented and remain the integration surface for the app.
- Phase 5: Domain tools (claims intake, fraud detector, settlement calculator) are implemented and integrated.
- Phase 6: Observability, LangSmith tracing, error handling, and unit tests with ≥80% coverage are in place.
- Phase 7: Evaluation against 20 test queries completed with documentation.
- Phase 7A: Golden dataset and regression validation harness created.
- Phase 8-9: SQLite schema for policy/claim domain data and fraud-gated claim registration implemented.
- Phase 10-15: Full RAG extension implemented including manifest-driven loaders, hybrid retrieval, QA chain with citations, evaluation harness, agent integration, and acceptance sign-off.

---

## Phase execution note

Each phase below should remain independently runnable. Before accepting a phase, validate it with the relevant command(s) and confirm that existing working flows were not regressed.

---

## Global project structure (for all phases)

> **Copilot note:** Keep this structure from Phase 0 onwards. Each phase will populate or refine specific modules.

Actual project layout (as implemented):

```text
.
├── .env.example
├── .gitignore
├── diagnostics.md
├── END_TO_END_RAG_VALIDATION.json
├── END_TO_END_RAG_VALIDATION.md
├── extended_plan.md
├── INGESTION_FIX_SUMMARY.md
├── METADATA_FILTER_FIX_SUMMARY.md
├── plan.md
├── README.md
├── requirements.txt
├── RETRIEVAL_FALLBACK_FIX_SUMMARY.md
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── langsmith_integration.py
│   ├── main.py
│   ├── api/
│   │   └── server.py
│   ├── chains/
│   │   ├── agent_chain.py
│   │   ├── base_chain.py
│   │   ├── faq_chain.py
│   │   └── simple_faq_llm.py
│   ├── logging/
│   │   └── json_logger.py
│   ├── memory/
│   │   └── sqlite_memory.py
│   ├── models/
│   │   ├── domain.py
│   │   └── faq.py
│   ├── prompts/
│   │   ├── examples_store.py
│   │   ├── faq_examples.py
│   │   ├── loader.py
│   │   └── templates.json
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── acceptance_validation.py
│   │   ├── chunkers.py
│   │   ├── embeddings.py
│   │   ├── evaluation_harness.py
│   │   ├── ingest_basic.py
│   │   ├── list_ids.py
│   │   ├── loaders.py
│   │   ├── metadata.py
│   │   ├── qa_chain.py
│   │   ├── qa_demo.py
│   │   ├── query_transform.py
│   │   ├── reranker.py
│   │   ├── retriever_basic.py
│   │   ├── retriever_bm25.py
│   │   ├── retriever_hybrid.py
│   │   ├── benchmarks/
│   │   │   ├── __init__.py
│   │   │   └── vector_backend_bench.py
│   │   └── vectorstores/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── chroma_store.py
│   │       ├── faiss_store.py
│   │       └── pinecone_store.py
│   └── tools/
│       ├── claim_status_checker.py
│       ├── claims_intake.py
│       ├── fraud_detector.py
│       ├── guardrails.py
│       ├── knowledge_retrieval.py
│       ├── policy_checker.py
│       └── settlement_calculator.py
├── data/
│   ├── claims.db
│   ├── faiss_index/                 # persisted FAISS vector index (Phase R2)
│   ├── faiss_index.backup
│   ├── faiss_index.meta.json
│   ├── faiss_index.meta.json.backup
│   ├── golden_dataset/
│   │   ├── claims.json
│   │   ├── faq.json
│   │   ├── fraud.json
│   │   ├── guardrails.json
│   │   ├── memory.json
│   │   ├── rag_aml_fraud.json
│   │   ├── rag_claims_insurance.json
│   │   ├── rag_customer_svc.json
│   │   ├── rag_failure_cases.json
│   │   ├── rag_knowledge_base_golden.json
│   │   ├── rag_loan_underwriting.json
│   │   ├── settlement.json
│   │   └── test_queries.json
│   └── knowledge_base/
│       ├── manifest.yaml
│       ├── adjudication_memos/
│       │   └── prior_adjudication_memos.csv
│       ├── exclusions/
│       │   └── health_exclusions_summary.pdf
│       ├── network/
│       │   ├── hospital_network_agreement_bopartitemodel.docx
│       │   └── hospital_network_agreement.pdf
│       ├── policies/
│       │   ├── health_hdfcergo_wording.pdf
│       │   ├── health_kotakmahindra_wording.pdf
│       │   ├── health_sbihealth_wording.pdf
│       │   ├── motor_sbi_private_wording.pdf
│       │   └── motor_sbi_wording.pdf
│       └── regulations/
│           └── irDAI_health_regulations_2016.docx
├── docs/
│   ├── api_rag_testing_guide.md
│   ├── env_configuration_checklist.md
│   ├── evaluation_report.md
│   ├── knowledge_base_eval_golden_sets.md
│   ├── project_acceptance_mapping.md
│   ├── project_signoff_report.md
│   ├── rag_integration.md
│   ├── test_user_queries.md
│   └── vector_backend_choice.md
├── eval/
│   ├── __init__.py
│   ├── eval_set.json
│   ├── extrinsic.py
│   ├── failure_analysis.py
│   ├── golden_set.json
│   ├── intrinsic.py
│   ├── llm_judge.py
│   ├── run_eval.py
│   └── run_failure_eval.py
├── frontend/
│   └── streamlit_app.py
├── notebooks/
│   └── 01_chunking_comparison.ipynb
├── reports/
│   ├── acceptance_evidence.json
│   ├── eval_baseline.md
│   ├── eval_final.md
│   ├── failure_analysis.json
│   ├── langsmith_trace_verifiation_report.md
│   ├── langsmith_trace_verification_kb.json
│   ├── langsmith_trace_verification.json
│   ├── rag_pipeline_langsmith_verification_report.md
│   ├── rag_pipeline_langsmith_verification.json
│   ├── remediation_baseline.md
│   ├── report.md
│   └── summary.json
├── Screenshots/
│   └── Screenshots.docx
├── scripts/
│   ├── demo_policy_status.py
│   ├── diagnose_retrieval_path.py
│   ├── end_to_end_rag_validation.py
│   ├── evaluate_rag.py
│   ├── evaluate.py
│   ├── generate_eval_golden_sets.py
│   ├── rag_evaluation_results.json
│   ├── results.json
│   ├── validate_generated_datasets.py
│   ├── validate_golden_dataset.py
│   ├── validate_memory_continuity.py
│   ├── validate_sqlite_context.py
│   ├── verify_ingestion.py
│   ├── verify_langsmith_traces.py
│   └── verify_rag_pipeline_with_langsmith.py
└── tests/
    ├── conftest.py
    ├── test_api_caching.py
    ├── test_api_chat.py
    ├── test_api_health_reset.py
    ├── test_api_langsmith_integration_enabled.py
    ├── test_api_langsmith.py
    ├── test_api_rag_and_retrieval.py
    ├── test_api_rag_endpoints.py
    ├── test_api_server_coverage.py
    ├── test_base_chain.py
    ├── test_claims_intake.py
    ├── test_eval_package.py
    ├── test_faq_chain_coverage.py
    ├── test_fraud_detector.py
    ├── test_guardrails.py
    ├── test_json_logger.py
    ├── test_knowledge_retrieval_integration.py
    ├── test_langsmith_integration_coverage.py
    ├── test_langsmith_integration_unit.py
    ├── test_metadata_filter_inference.py
    ├── test_multi_turn_context.py
    ├── test_policy_checker.py
    ├── test_policy_status_integration.py
    ├── test_prompts_loader.py
    ├── test_queries_validation.py
    ├── test_rag_evaluation_harness.py
    ├── test_rag_golden_dataset.py
    ├── test_rag_hybrid_reranking_streaming.py
    ├── test_rag_integration.py
    ├── test_rag_pipeline.py
    ├── test_rag_retriever_and_config.py
    ├── test_rag_simple.py
    ├── test_retrieval_filter_fallback.py
    ├── test_settlement_calculator.py
    ├── test_simple_acknowledgments.py
    ├── test_sqlite_memory_coverage.py
    ├── test_sqlite_persistence.py
    └── test_vector_backend.py
```

---

## Phase 0 - Repo scaffolding and environment

**Goal**  
Create a clean, reproducible project skeleton suitable for incremental Copilot-driven development.

**Tasks**

- [x] Create the directory structure shown in **Global project structure**.
- [x] Add `requirements.txt` with (minimum):
  - `fastapi`, `uvicorn[standard]`
  - `streamlit`
  - `langchain`
  - `openai`
  - `pydantic`
  - `python-dotenv`
  - `pytest`, `pytest-cov`
  - `langsmith` (or latest client)
- [x] Implement `app/config.py`:
  - [x] Use `BaseSettings` (Pydantic v1 or v2) for:
    - `OPENAI_API_KEY`
    - `OPENAI_MODEL_NAME`
    - `LOG_LEVEL`
    - `SQLITE_DB_PATH`
    - `ENVIRONMENT` (e.g., `dev` / `prod`)
  - [x] Add a helper `get_settings()` singleton.
- [x] Create `.env.example` with all required keys, documented.
- [x] Implement basic JSON logging setup in `app/logging/json_logger.py`:
  - [x] Function `get_logger(name: str)` that returns a logger emitting JSON lines (`timestamp`, `level`, `message`, plus extra dict).
- [x] Create placeholder files for all modules (empty classes/functions) so imports will resolve in later phases.
- [x] Add a minimal `frontend/streamlit_app.py` that renders a placeholder page ("Claims Processing & Settlement - WIP").
- [x] Add a minimal `README.md` describing setup steps (install, env, run placeholder Streamlit).

**Acceptance criteria**

- [x] `pip install -r requirements.txt` succeeds.
- [x] `streamlit run frontend/streamlit_app.py` opens a placeholder app without errors.
- [x] `python -m app` imports resolve (no missing modules).

---

## Phase 1 - Week 2 baseline FAQ chatbot (Streamlit + direct LLM)

**Goal**  
Deliver an initial Insurance FAQ chatbot (Week 2 scope) using direct OpenAI calls from Streamlit, focusing on prompt engineering, few-shot examples, and basic JSON-shaped responses (no LangChain yet).

**Tasks**

- [x] Implement simple FAQ examples in `app/prompts/faq_examples.py`:
  - [x] Include at least the 3 provided examples:
    - New claim processing.
    - Suspicious claim / fraud signals.
    - Claim status update.
  - [x] Add 2-3 more examples for policy coverage and documentation FAQs.
- [x] Implement a simple prompt builder in `app/chains/simple_faq_llm.py`:
  - [x] System prompt: role = Insurance FAQ assistant for claims, policy, and settlement questions.
  - [x] Ingest few-shot examples from `faq_examples.py` into a text prompt.
  - [x] Append user message.
- [x] Implement an LLM call helper using OpenAI ChatCompletion:
  - [x] Function `call_faq_llm(user_message: str) -> dict`.
  - [x] Return structure: `{ "answer_text": str, "raw_json": dict | None }`.
  - [x] Ask model to output an `answer` plus a simple JSON stub (`intent`, `category`, `confidence`).
- [x] Build initial Streamlit FAQ UI in `frontend/streamlit_app.py`:
  - [x] Text input box for user query.
  - [x] "Send" button.
  - [x] Display model's `answer_text` and a pretty-printed JSON block.
  - [x] Store chat history in `st.session_state` (no DB yet) to show a scrollable conversation.

**Acceptance criteria**

- [x] Running `streamlit run frontend/streamlit_app.py` yields a working FAQ chatbot.
- [x] At least 5 manual queries produce consistent text answers + JSON stubs.
- [x] No LangChain, FastAPI, or SQLite in use yet.

---

## Phase 2 - Guardrails, Pydantic parsing, and prompt template library (Week 2 hardening)

**Goal**  
Upgrade the FAQ chatbot with Pydantic-validated outputs, a prompt template library, and input guardrails (PII, off-topic, injection) so Week 2 deliverables are fully met.

**Tasks**

- [x] Define Pydantic models in `app/models/faq.py`:
  - [x] `FAQIntent` enum: `CLAIM_REGISTRATION`, `POLICY_STATUS`, `FRAUD_CHECK`, `SETTLEMENT_QUERY`, `DOCUMENTS_REQUIRED`, `OTHER`.
  - [x] `FAQResponse` model:
    - `intent: FAQIntent`
    - `category: str`
    - `confidence: float`
    - `answer_text: str`
    - `reasoning: Optional[str]`
    - `metadata: Dict[str, Any] = {}`
- [x] Create prompt template library `app/prompts/templates.json`:
  - [x] System templates for main assistant.
  - [x] Guardrail templates (off-topic, unsafe content, PII warning).
  - [x] A small loader `app/prompts/loader.py` with `load_templates()`.
- [x] Implement guardrails in `app/tools/guardrails.py`:
  - [x] `detect_pii(text) -> dict` - simple regex for emails, phone numbers, IDs.
  - [x] `is_off_topic(text) -> dict` - keyword-based off-topic detection.
  - [x] `detect_prompt_injection(text) -> dict` - look for "ignore previous instructions", etc.
  - [x] Return a structured result `{ "triggered": bool, "rule": str, "details": str }`.
- [x] Integrate guardrails into `frontend/streamlit_app.py`:
  - [x] Before calling `call_faq_llm`, run all guardrails.
  - [x] If any guardrail triggers, return a safe canned response using `templates.json`.
- [x] Replace ad-hoc JSON handling with Pydantic:
  - [x] Ask the LLM to return a JSON block for `FAQResponse`.
  - [x] Parse with `FAQResponse.parse_raw` or `parse_obj`.
  - [x] On parse failure, re-prompt once with a stricter instruction.

**Acceptance criteria**

- [x] All responses are validated instances of `FAQResponse` in the UI.
- [x] PII/off-topic/injection attempts are blocked with safe guardrail responses.
- [x] `templates.json` is the single source of truth for system and guardrail messages.

---

## Phase 3 - Introduce LangChain, SQLite memory, semantic example selection (core assistant start)

**Goal**  
Refactor the FAQ chatbot into a LangChain-based chain with SQLite-backed conversational memory and semantic few-shot selection while preserving Week 2 behavior.

**Tasks**

- [x] Set up base LangChain utilities in `app/chains/base_chain.py`:
  - [x] Helper to create the OpenAI chat model.
  - [x] Helper to create a `ChatPromptTemplate` from templates and examples.
- [x] Implement SQLite-backed memory in `app/memory/sqlite_memory.py`:
  - [x] Use `sqlite3` or LangChain's `SQLiteChatMessageHistory`.
  - [x] Provide functions:
    - `get_history(session_id: str) -> list[BaseMessage]`.
    - `append_message(session_id: str, role: str, content: str)`.
  - [x] Ensure at least 10 turns are preserved per session.
- [x] Implement semantic few-shot selection in `app/prompts/examples_store.py`:
  - [x] Store all FAQ examples with `embedding` (computed via OpenAI embeddings).
  - [x] Function `select_examples(query: str, k: int) -> list[Example]` using cosine similarity.
- [x] Build LangChain FAQ chain in `app/chains/faq_chain.py`:
  - [x] Combine guardrails (pre-filter) + semantic examples + `ChatPromptTemplate` + LLM.
  - [x] Use `RunnableWithMessageHistory` (or custom) for memory integration.
- [x] Update `frontend/streamlit_app.py` (temporarily still direct to Python, not API):
  - [x] Replace `call_faq_llm` with calls to `faq_chain.invoke()`.
  - [x] Pass a `session_id` (from `st.session_state`) to memory layer.

**Acceptance criteria**

- [x] Multi-turn conversations (≥10 turns) retain context via SQLite across messages.
- [x] Few-shot examples are selected based on semantic similarity.
- [x] Week 2 features (guardrails, structured FAQResponse) still work as before.

---

## Phase 3A - Multi-turn memory reliability hardening (non-breaking)

**Goal**  
Improve multi-turn context reliability for same-session conversations without changing the currently working FAQ, tool, or API flows.

**Tasks**

- [x] Keep the existing SQLite memory and agent/tool logic intact.
- [x] Add a single, explicit path in the agent layer to read the active session history before tool and follow-up decisions.
- [x] Ensure a follow-up turn in the same `session_id` can reuse previously supplied policy numbers, claim IDs, incident dates, and other extracted context.
- [x] Add a regression test for at least 3 consecutive turns in one session.
- [x] Add a simple validation command for memory continuity so the phase can be verified independently.

**Acceptance criteria**

- [x] The current FAQ, policy status, claim intake, fraud, and settlement flows still pass their existing test suite.
- [x] Running `PYTHONPATH=. pytest -q tests/test_claims_intake.py -k 'history or multiturn'` passes.
- [x] Running `PYTHONPATH=. python scripts/validate_sqlite_context.py` passes.
- [x] A new multi-turn regression test confirms that a later turn in the same `session_id` can reuse earlier context without asking the user to repeat it.
- [x] No existing endpoint, frontend, or tool behavior is removed; only context handling and validation are strengthened.

---

## Phase 4 - FastAPI backend with /chat, /reset, /health and Streamlit as client (core assistant core)

**Goal**  
Create a FastAPI backend exposing `/chat`, `/reset`, `/health`, and refactor Streamlit to call these endpoints. Add structured JSON logging and basic retry/fallback behavior.

**Tasks**

- [x] Implement FastAPI app in `app/api/server.py` and `app/main.py`:
  - [x] `/health` (GET): returns JSON with version, uptime, model name, DB status.
  - [x] `/chat` (POST):
    - Request model: `session_id: str`, `message: str`, optional `metadata: dict`.
    - Response model: `answer_text: str`, `structured: FAQResponse`, `chain_metadata: dict`.
  - [x] `/reset` (POST): clears conversation history for `session_id` in SQLite.
- [x] Wire `faq_chain` into FastAPI handlers:
  - [x] Use shared memory layer (`sqlite_memory.py`).
  - [x] Add a simple retry: on exception or parse error, run a fallback chain with a safe, short answer.
- [x] Add structured JSON logging via `json_logger.py`:
  - [x] Log each `/chat` call with: `timestamp`, `session_id`, `user_message`, `intent`, `latency_ms`, `guardrail_triggered`, `error_info`.
- [x] Refactor `frontend/streamlit_app.py` to use HTTP:
  - [x] Maintain `session_id` in `st.session_state`.
  - [x] Call FastAPI `/chat` via `requests` (or `httpx`).
  - [x] Use `/reset` endpoint for "Reset conversation" button.
  - [x] Optionally show `chain_metadata` in an expandable panel per message.

**Acceptance criteria**

- [x] `uvicorn app.main:app --reload` starts the backend and `/health` returns correct JSON.
- [x] Streamlit frontend no longer calls LangChain directly; all chat goes through FastAPI.
- [x] Logs for `/chat` requests appear as JSON lines with required fields.
- [x] Validation command: `PYTHONPATH=. pytest -q tests/test_api_chat.py tests/test_api_health_reset.py` passes.
- [x] Existing working chat and reset behavior remains intact after the memory-hardening change.

---

## Phase 5 - Domain tools: Claims Intake, Fraud Detector, Settlement Calculator (core assistant tools)

**Goal**  
Implement the three custom domain tools and integrate them as LangChain tools within the chat flow.

**Tasks**

- [x] Define domain models in `app/models/domain.py`:
  - [x] `Policy` with fields like `policy_number`, `sum_insured`, `deductible`, `copay_percent`, `sub_limits: Dict[str, float]`, `status`, etc.
  - [x] `Claim` with fields like `claim_id`, `policy_number`, `claim_amount`, `diagnosis_code`, `hospital_name`, `admission_date`, `discharge_date`, etc.
  - [x] `FraudSignal` and `FraudScoreResult` with fields such as `score`, `signals: List[str]`.
  - [x] `SettlementBreakdown` with `gross_amount`, `deductible`, `copay_amount`, `approved_amount`, `notes: List[str]`.
- [x] Create a small in-memory or SQLite-backed data store for demo policies/claims:
  - [x] Utility functions to fetch policies/claims by ID.
- [x] Tool 1 - Claims Intake & Validation Engine (`app/tools/claims_intake.py`):
  - [x] Function `register_and_validate_claim(policy_number: str, claim_amount: float, extra_info: dict) -> ClaimValidationResult`.
  - [x] Responsibilities:
    - Create a new claim ID.
    - Check basic policy status and constraints.
    - Produce flags for obvious coverage gaps/exclusions (generic rules only).
- [x] Tool 2 - Fraud Signal Detector (`app/tools/fraud_detector.py`):
  - [x] Function `compute_fraud_score(claim_id: str) -> FraudScoreResult`.
  - [x] Use simple fraud indicators:
    - Time between policy start and claim.
    - Count of recent claims.
    - Duplicate claim amounts.
- [x] Tool 3 - Settlement Calculator (`app/tools/settlement_calculator.py`):
  - [x] Function `calculate_settlement(claim_id: str) -> SettlementBreakdown`.
  - [x] Apply generic rules using policy fields: deductible, copay, simple depreciation factor, sub-limits.
- [x] Integrate tools into LangChain as `StructuredTool` or `@tool` wrappers.
- [x] Build an agent-style chain in `app/chains/agent_chain.py`:
  - [x] Use `FAQResponse.intent` to decide when to call which tools.
  - [x] Merge tool outputs into final `answer_text` and `FAQResponse.metadata`.
- [x] Update `/chat` endpoint to use `agent_chain` instead of FAQ-only chain.

**Acceptance criteria**

- [x] Queries like "Register a new claim...", "What is the fraud score for claim X?", "Calculate settlement for claim Y" invoke the appropriate tools.
- [x] Tool inputs/outputs are validated with Pydantic.
- [x] Tools are modular and decoupled from the API/front-end code.

---

## Phase 6 - Observability, LangSmith, error handling, and tests (core assistant hardening)

**Goal**  
Add robust error handling, LangSmith tracing, latency tracking, and unit tests with ≥80% coverage for the custom tools.

**Tasks**

- [x] Enhance error handling in FastAPI:
  - [x] Global exception handler that wraps unhandled errors into a safe JSON response.
  - [x] Attach a correlation ID to each `/chat` request and propagate in logs.
  - [x] Ensure fallback chain is used on LLM/tool errors.
- [x] Integrate LangSmith tracing:
  - [x] Configure LangSmith project name and API key via `.env`.
  - [x] Enable tracing for `agent_chain` and tool calls.
- [x] Add latency metrics:
  - [x] Measure total `/chat` latency, LLM call time, and tool execution time per request.
  - [x] Include these in `chain_metadata` and logs.
  - [x] Verify simple queries < 3 seconds and tool-augmented ones < 8 seconds under normal conditions.
- [x] Implement unit tests with `pytest` in `tests/`:
  - [x] `test_claims_intake.py`: normal and edge cases (inactive policy, large claim).
  - [x] `test_fraud_detector.py`: scenarios with 0, medium, and high fraud scores.
  - [x] `test_settlement_calculator.py`: varying deductibles and copays.
  - [x] Use `pytest-cov` to ensure ≥80% coverage for each `app/tools/*.py` module.

**Acceptance criteria**

- [x] FastAPI never returns raw stack traces to clients; errors are logged and user gets a helpful fallback message.
- [x] LangSmith (or equivalent) shows traces of chains with tool calls and memory.
- [x] `pytest --cov=app/tools` reports ≥80% coverage for tools.

---

## Phase 7 - Evaluation, Week 2 & core assistant validation, and documentation

**Goal**  
Validate the system against the 20 specified test queries, and finalize documentation showing that both Week 2 and core assistant requirements are met.

**Tasks**

- [x] Implement a simple evaluation harness (CLI script or notebook):
  - [x] Call `/chat` for each of the 20 queries (FAQ, tools, multi-turn).
  - [x] Save responses (answer, `FAQResponse`, `chain_metadata`) to a JSON or CSV file.
- [x] Manually verify behavior for each category:
  - [x] FAQ-only questions.
  - [x] Tool usage questions.
  - [x] Multi-turn scenarios leveraging memory.
- [x] Create `docs/evaluation_report.md`:
  - [x] For each query: input, expected behavior (1-2 lines), actual behavior (summary), and notes on accuracy, formatting, and guardrail effectiveness.
  - [x] Explicitly mark which queries exercised which tools.
- [x] Update `README.md` with:
  - [x] High-level architecture description and simple diagram (optional ASCII).
  - [x] Setup and run instructions (backend, frontend, env variables, DB initialization).
  - [x] How to run tests and evaluation harness.
  - [x] Note on Week 2 vs core assistant requirements and where each is satisfied.
- [x] Capture or link LangSmith traces/screenshots showing chain execution and tool usage.

**Acceptance criteria**

- [x] All 20 test queries run end-to-end through `/chat` and are documented in the evaluation report.
- [x] README and docs clearly explain how this project satisfies both Week 2 and core assistant requirements.
- [x] Repo is ready for review as a capstone project.

---

## Phase 7A - Golden dataset and regression validation for future RAG extension

**Goal**  
Prepare the project for a future RAG extension by creating a phase-specific golden dataset and regression validation process for each phase and tool.

**Tasks**

- [x] Define a golden dataset format that captures: user query, expected tool/intent, expected structured response, relevant context history, and evaluation labels.
- [x] Create a `data/golden_dataset/` folder pattern in the repo and add at least one seed golden dataset file for:
  - FAQ responses.
  - Claim intake workflow.
  - Fraud detection workflow.
  - Settlement calculation workflow.
  - Multi-turn memory/regression scenarios.
- [x] Implement a validation harness or script (e.g. `scripts/validate_golden_dataset.py`) that:
  - [x] Loads each golden dataset item.
  - [x] Replays it against the current system phase/tool chain.
  - [x] Compares actual outputs to expected structured outputs.
  - [x] Produces a regression summary with pass/fail status per item.
- [x] Add phase/tool-specific validation checks:
  - [x] FAQ phase: verify intent classification, JSON schema, and answer correctness against golden expected labels.
  - [x] Tool phase: verify correct tool selection and expected tool output structure.
  - [x] Memory phase: verify multi-turn context reuse and stateful references across turns.
- [x] Document the golden dataset schema and validation command in `README.md` or `docs/evaluation_report.md`.
- [x] Add or update tests to ensure the golden dataset validation harness works as a regression gate.

**Acceptance criteria**

- [x] A golden dataset schema is documented and stored in the repo.
- [x] A regression validation script can execute the dataset and report pass/fail results.
- [x] The phase/tool regression dataset covers FAQ, claim intake, fraud, settlement, and multi-turn memory.
- [x] Future RAG extension decisions can leverage the dataset and regression results as a clear baseline.

---

## Phase 8 - SQLite schema for policy and claim domain data

**Goal**  
Create a persistent SQL schema for demo policy and claim records while keeping the current in-memory domain store intact.

**Tasks**

- [x] Extend `app/memory/sqlite_memory.py` to create `policies` and `claims` tables alongside `chat_history`.
- [x] Define policy-specific columns for coverage, sub-limits, dates, and underwriting metadata.
- [x] Define claim-specific columns for claim details, optional enrichment, and fraud/settlement state.
- [x] Keep the existing `chat_history` conversation memory unchanged.
- [x] Optionally add a schema validation helper or startup check that confirms all three tables exist.

**Acceptance criteria**

- [x] `SQLiteMemory` initializes the `policies` and `claims` tables on startup.
- [x] The schema includes optional enrichments such as `status`, `approved_amount`, `fraud_score`, and `settlement_status`.
- [x] The SQLite database file contains `chat_history`, `policies`, and `claims` after app initialization.

---

## Phase 9 - Migrate domain data to SQLite and enforce fraud-gated claims

**Goal**  
Move policy and claim storage from in-memory dictionaries to SQL tables, and require fraud validation before claim persistence.

**Tasks**

- [x] Replace in-memory domain access functions in `app/models/domain.py` with SQL-backed implementations using `policies` and `claims`.
- [x] Use the SQL tables for `get_policy`, `get_claim`, `save_claim`, `get_claims_for_policy`, and `get_claims_for_policy_holder`.
- [x] Update `app/tools/claims_intake.py` so claim registration computes fraud score before saving a claim.
- [x] Persist `claim.status`, `claim.fraud_score`, and optional settlement status on saved claim records.
- [x] Add a demo data seeding helper that populates SQL tables from the current `_DEMO_POLICIES` and `_DEMO_CLAIMS` seed data.

**Acceptance criteria**

- [x] Claim registration only persists claim records when fraud score is within acceptable bounds.
- [x] Fraud detection can operate both against saved claims and against a pre-save claim evaluation.
- [x] Existing tool and API tests continue to pass with SQL-backed policy/claim persistence.

---

## Copilot execution guidelines

- Work **top-down**: complete Phase 0, then 1, ..., 7 in order.  
- For each phase:
  - Read the phase's **Goal**, then execute the **Tasks** using Copilot for code generation.
  - Use the **Acceptance criteria** as a regression checklist before moving forward.
- You may paste in or adapt prompt/chain patterns from classroom materials, but keep all code organized into the modules indicated above.


---

## RAG extension - RAG layer for Claims Processing

> These phases extend the existing Week 2/core Claims Processing & Settlement solution with a production-grade RAG layer,
> aligning with the RAG extension requirements document and the trainer's banking RAG reference project.
> Implement them only after core assistant phases are stable. Each phase should be independently runnable.

### RAG extension overall goals

- Add a document-backed knowledge layer over policy wordings, IRDAI regulations, exclusion lists,
  hospital network agreements, and prior adjudication memos.
- Expose a RAG-backed knowledge_retrieval tool to the existing LangChain agent so it can decide
  per query whether to call a deterministic tool, the RAG tool, or both.
- Implement hybrid retrieval (BM25 + dense vectors), re-ranking, answer + citations, and an
  evaluation harness measuring retrieval and answer quality against a golden set.

The repository layout for RAG modules follows this pattern:

```text
project-root/
├── app/
│   ├── rag/              # RAG pipeline
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── acceptance_validation.py
│   │   ├── chunkers.py   # recursive + semantic chunking strategies
│   │   ├── embeddings.py # embedding adapters (OpenAI + sentence-transformers)
│   │   ├── evaluation_harness.py
│   │   ├── ingest_basic.py
│   │   ├── list_ids.py
│   │   ├── loaders.py    # multi-format document loaders
│   │   ├── metadata.py
│   │   ├── qa_chain.py   # answer + citations QA chain
│   │   ├── qa_demo.py
│   │   ├── query_transform.py  # multi-query expansion / HyDE
│   │   ├── reranker.py
│   │   ├── retriever_basic.py  # dense-only retriever
│   │   ├── retriever_bm25.py   # BM25 retriever
│   │   ├── retriever_hybrid.py # BM25 + dense hybrid retriever with re-ranking
│   │   ├── benchmarks/
│   │   │   ├── __init__.py
│   │   │   └── vector_backend_bench.py
│   │   └── vectorstores/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── chroma_store.py
│   │       ├── faiss_store.py
│   │       └── pinecone_store.py
│   ├── tools/
│   │   ├── knowledge_retrieval.py  # RAG-backed tool for the agent
│   │   └── ... existing tools ...
│   └── ... existing modules ...
├── data/
│   ├── golden_dataset/
│   │   ├── rag_aml_fraud.json
│   │   ├── rag_claims_insurance.json
│   │   ├── rag_customer_svc.json
│   │   ├── rag_failure_cases.json
│   │   ├── rag_knowledge_base_golden.json
│   │   ├── rag_loan_underwriting.json
│   │   └── ... existing golden datasets ...
│   └── knowledge_base/
│       ├── manifest.yaml
│       ├── adjudication_memos/
│       ├── exclusions/
│       ├── network/
│       ├── policies/
│       └── regulations/
├── eval/                 # RAG eval harness
│   ├── __init__.py
│   ├── eval_set.json
│   ├── extrinsic.py      # answer faithfulness/answer correctness metrics
│   ├── failure_analysis.py
│   ├── golden_set.json   # RAG extension golden set
│   ├── intrinsic.py      # retrieval metrics (Hit@K, MRR, NDCG, context precision/recall)
│   ├── llm_judge.py      # LLM-as-judge scoring
│   ├── run_eval.py       # CLI entrypoint
│   └── run_failure_eval.py
├── docs/
│   ├── api_rag_testing_guide.md
│   ├── env_configuration_checklist.md
│   ├── evaluation_report.md
│   ├── knowledge_base_eval_golden_sets.md
│   ├── project_acceptance_mapping.md
│   ├── project_signoff_report.md
│   ├── rag_integration.md
│   ├── test_user_queries.md
│   └── vector_backend_choice.md  # justification for FAISS/Chroma/Pinecone
├── reports/
│   ├── acceptance_evidence.json
│   ├── eval_baseline.md
│   ├── eval_final.md
│   ├── failure_analysis.json
│   ├── langsmith_trace_verifiation_report.md
│   ├── langsmith_trace_verification_kb.json
│   ├── langsmith_trace_verification.json
│   ├── rag_pipeline_langsmith_verification_report.md
│   ├── rag_pipeline_langsmith_verification.json
│   ├── remediation_baseline.md
│   ├── report.md
│   └── summary.json
├── scripts/
│   ├── diagnose_retrieval_path.py
│   ├── end_to_end_rag_validation.py
│   ├── evaluate_rag.py
│   ├── generate_eval_golden_sets.py
│   ├── rag_evaluation_results.json
│   ├── validate_generated_datasets.py
│   ├── validate_golden_dataset.py
│   ├── verify_ingestion.py
│   ├── verify_langsmith_traces.py
│   └── verify_rag_pipeline_with_langsmith.py
├── tests/
│   ├── test_api_rag_and_retrieval.py
│   ├── test_api_rag_endpoints.py
│   ├── test_knowledge_retrieval_integration.py
│   ├── test_rag_evaluation_harness.py
│   ├── test_rag_golden_dataset.py
│   ├── test_rag_hybrid_reranking_streaming.py
│   ├── test_rag_integration.py
│   ├── test_rag_pipeline.py
│   ├── test_rag_retriever_and_config.py
│   ├── test_rag_simple.py
│   └── test_vector_backend.py
└── ... existing files ...
```

---

## Phase 10 - RAG foundations: manifest-driven loaders, chunking, embeddings, basic retriever

**Goal**  
Stand up an end-to-end retrieval pipeline over the claims knowledge base so the agent has a
knowledge layer to draw on. This phase focuses on a manifest-driven loader, chunking strategies,
embeddings, and a basic dense retriever.

**Tasks**

- [x] Create `data/knowledge_base/manifest.yaml` listing all KB sources:
  - Health policy wordings:  
    - `policies/health_hdfcergo_wording.pdf`  
    - `policies/health_kotakmahindra_wording.pdf`  
    - `policies/health_sbihealth_wording.pdf`
  - Motor policy wordings:  
    - `policies/motor_sbi_private_wording.pdf`  
    - `policies/motor_sbi_wording.pdf`
  - Regulations:  
    - `regulations/irDAI_health_regulations_2016.docx`
  - Network agreements:  
    - `network/hospital_network_agreement_bopartitemodel.docx`  
    - `network/hospital_network_agreement.pdf`
  - Exclusions:  
    - `exclusions/health_exclusions_summary.pdf`
  - Adjudication memos:  
    - `adjudication_memos/prior_adjudication_memos.csv`

  Each entry includes: `id`, `path`, `doc_type`, `insurance_type`, `product_code`,
  `product_name` (where applicable), `claim_type` (for memos), and `jurisdiction`.

- [x] Create `app/rag/loaders.py`:

  - [x] Implement `load_manifest()` that reads `manifest.yaml` from `KNOWLEDGE_BASE_DIR`
        (e.g., `data/knowledge_base/manifest.yaml`) using `pyyaml` and returns the parsed dict.
  - [x] Implement `iter_manifest_sources()` that yields a normalized structure for each source:
        `id`, full `path`, `doc_type`, `insurance_type`, `product_code`, `product_name`,
        `claim_type`, `jurisdiction`, and any extra `metadata`.
  - [x] Implement format-specific loaders for PDF, DOCX, Markdown, and JSON using tools
        specified in `requirements.txt` (`pypdf`, `python-docx`, `docx2txt`, `beautifulsoup4`,
        `lxml`, etc.).  
        - PDF → text via `pypdf`.  
        - DOCX → text via `python-docx` or `docx2txt`.  
        - Markdown → text via simple file read.  
        - CSV memos → flatten relevant fields into text blocks.
  - [x] Provide a function `load_documents_from_manifest() -> list[Document]` where `Document`
        includes `text`, `source_id`, `source_path`, `doc_type`, `insurance_type`, `product_code`,
        `claim_type`, and raw metadata.

- [x] Create `app/rag/chunkers.py`:

  - [x] Implement `recursive_chunk(text, config)` using `RecursiveCharacterTextSplitter` with
        Week-6 defaults (size ~800, overlap ~100).
  - [x] Implement a semantic chunker for policy/regulation/network docs that respects headings,
        clause numbering, and section breaks.
  - [x] Ensure chunk metadata preserves `doc_type`, `insurance_type`, `product_code`,
        `claim_type`, and `section/clause_id` where available.

- [x] Create `app/rag/embeddings.py`:

  - [x] Wrap OpenAI embeddings (e.g., `text-embedding-3-small`) and sentence-transformer models
        from `sentence-transformers`.
  - [x] Provide an adapter `get_embedding_fn(model_name: str)` that returns a callable used by
        vector stores, ensuring the same model is used for ingestion and querying (version-pinned
        via config).

- [x] Create `app/rag/vectorstores/base.py`:

  - [x] Define an abstract `VectorStore` interface (`add`, `search`, `delete`, `persist`,
        `as_retriever`).
  - [x] Implement concrete vector stores:
        - `faiss_store.py` using `faiss-cpu`.
        - `chroma_store.py` using `chromadb`.
        - `pinecone_store.py` (optional, if using managed Pinecone).
  - [x] Implement `get_vector_store(backend: str)` in `app/rag/vectorstores/__init__.py` that
        returns the appropriate store instance based on `VECTOR_BACKEND` (`faiss`, `chroma`,
        `pinecone`).

- [x] Create `app/rag/retriever_basic.py`:

  - [x] Implement `build_basic_retriever()` that:
        - Calls `load_documents_from_manifest()` to read all sources defined in `manifest.yaml`.
        - Applies recursive/semantic chunking to each document type.
        - Embeds chunks using the chosen embedding model from `get_embedding_fn`.
        - Upserts chunks into the configured vector store with metadata fields:
          `doc_type`, `insurance_type`, `product_code`, `claim_type`, `section`, `clause_id`.
        - Returns a LangChain `VectorStoreRetriever` (or equivalent `Runnable`) over this store.

- [x] Add a CLI script `python -m app.rag.ingest_basic`:

  - [x] Entry point that:
        - Loads manifest sources.  
        - Runs loaders → chunkers → embeddings → vector store upsert.  
        - Prints a summary per `doc_type` (`policy_wording`, `regulation`, `network`,
          `exclusion_summary`, `memo`, etc.) including document counts and total chunk counts.

**How to run this phase independently**

- [x] Ensure `.env` includes:
  - `KNOWLEDGE_BASE_DIR` (root for claims KB documents, e.g., `data/knowledge_base`).  
  - `VECTOR_BACKEND` (`faiss`, `chroma`, or `pinecone`).  
  - `EMBEDDING_MODEL` (e.g., `text-embedding-3-small` or the sentence-transformer you choose).
- [x] Run: `python -m app.rag.ingest_basic`.
- [x] Verify that the chosen vector store directory/collection contains the expected number of
      chunks and that per-type counts match the manifest (3 health policies, 2 motor policies,
      1 regulations document, 2 network agreements, 1 exclusions file, adjudication memos).

**Acceptance criteria**

- [x] Multi-format document ingestion (PDF, DOCX, CSV) succeeds for all sources listed
      in `manifest.yaml`.
- [x] Corpus is chunked with both recursive and semantic strategies (configurable via code or
      `.env`), and chunk metadata includes `doc_type`, `insurance_type`, `product_code`,
      `claim_type`.
- [x] Embeddings are stored in the selected vector backend and can be retrieved via a basic
      retriever built from `build_basic_retriever()`.
- [x] RAG retrieval can filter by `doc_type` and `product_code` using metadata, aligning with
      Week-6 requirements for clause/exclusion/regulatory lookups.

---

## Phase 11 - Vector backends and metadata schema selection

**Goal**  
Abstract the storage layer so you can choose FAISS, Chroma, or Pinecone per environment, and define
metadata schemas suitable for claims processing (policy wordings, regulations, network agreements,
prior memos).

**Tasks**

- [x] Implement a vector store factory in `app/rag/vectorstores/__init__.py`:
  - [x] Function `get_vector_store(backend: str)` that returns an instance of FAISS/Chroma/Pinecone
        store based on `VECTOR_BACKEND`.
- [x] Add a benchmark script `app/rag/benchmarks/vector_backend_bench.py`:
  - [x] Measure ingestion time, top-5 retrieval latency (p50/p95), recall@5, and storage footprint
        for each backend on a sample corpus.
- [x] Define per-chunk metadata schema in `app/rag/metadata.py`:
  - [x] Fields: `doc_type` (policy_wording | regulation | network | memo), `insurance_type`,
        `insurer`, `product_code`, `claim_type`, `section`, `clause_id`.
  - [x] Ensure metadata is attached to every chunk on upsert.
- [x] Create `docs/vector_backend_choice.md`:
  - [x] Summarize benchmark results and justify chosen backend for your local dev environment
        (e.g., FAISS for local experiments, Chroma for simple persistence).

**How to run this phase independently**

- [x] Run: `python -m app.rag.benchmarks.vector_backend_bench`.
- [x] Inspect the generated metrics and choose a default backend via `.env`.

**Acceptance criteria**

- [x] All three backends (FAISS, Chroma, Pinecone if configured) are pluggable without code changes
      beyond config.
- [x] Each chunk in the vector store has the required metadata fields for filtering.
- [x] `docs/vector_backend_choice.md` is committed and explains the selection.

---

## Phase 12 - Advanced RAG: hybrid retrieval and QA chain

**Goal**  
Raise retrieval quality by combining sparse (BM25) and dense signals, applying query transforms, and
building an answer + citations QA chain that the claims agent can call.

**Tasks**

- [x] Implement BM25 retriever in `app/rag/retriever_bm25.py` using `rank-bm25` over the same
      chunks.
- [x] Implement hybrid retriever in `app/rag/retriever_hybrid.py`:
  - [x] Fuse BM25 and dense scores via weighted sum or RRF (reciprocal rank fusion).
  - [x] Support HyDE / multi-query expansion in `app/rag/query_transform.py`.
- [x] Implement QA chain in `app/rag/qa_chain.py`:
  - [x] Input: user query + optional claim context (policy number, claim id).
  - [x] Steps: hybrid retrieval → re-ranking via cross-encoder (e.g., `Cohere rerank` or
        `sentence-transformers` cross-encoder) → answer generation → citations.
  - [x] Output: JSON with `answer_text`, `citations: list[ChunkCitation]`, `confidence`.
  - [x] Enforce citation rule: every factual claim in the answer must reference a `chunk_id`; the
        response JSON includes the exact chunk text for each citation.
- [x] Create `python -m app.rag.qa_demo` for local testing:
  - [x] Accept a query from stdin.
  - [x] Print answer + citations and show top-5 retrieved chunks.

**How to run this phase independently**

- [x] After Phase 10-11 ingestion, run: `python -m app.rag.qa_demo`.
- [x] Test queries drawn from RAG extension per-project RAG test table (coverage lookup, exclusion,
      regulatory reference, sub-limit, past cases, network, comparative, hard/ambiguous, refusal,
      cite-required rejection letter).

**Acceptance criteria**

- [x] Hybrid retrieval improves top-5 recall and answer quality versus dense-only retriever.
- [x] QA chain returns answers with at least one citation per factual claim.
- [x] Critical query patterns (policy clause interpretation, exclusion lookup, partial-rejection
      decisions, regulatory limits) perform well.

---

## Phase 13 - RAG evaluation harness and acceptance thresholds

**Goal**  
Make the RAG pipeline measurable and align it with RAG extension acceptance thresholds for claims.

**Tasks**

- [x] Build a golden set in `eval/golden_set.json` and `data/golden_dataset/`:
  - [x] Include the 20 core assistant queries plus 30 RAG/hybrid queries (10 per-project RAG queries
        enumerated in the RAG extension document, plus 20 hybrid queries combining tools + RAG).
- [x] Implement `eval/intrinsic.py`:
  - [x] Metrics: Hit@K, MRR, NDCG, context precision/recall.
- [x] Implement `eval/extrinsic.py`:
  - [x] Metrics: faithfulness (groundedness), answer correctness, answer relevance.
- [x] Implement `eval/llm_judge.py`:
  - [x] Read separate judge configuration from environment (e.g., `JUDGE_MODEL_NAME`,
        `JUDGE_OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`).
  - [x] Construct a judge LLM instance that is distinct from the main generation model
        (e.g., app uses `OPENAI_MODEL_NAME=gpt-4o-mini`, judge uses `JUDGE_MODEL_NAME=gpt-4o`
        or an Anthropic Claude model).
  - [x] LLM-as-judge scoring for correctness, completeness, citation quality, and clarity
        (1-5 scale).
  - [x] Ensure eval code never falls back to the main app model if judge configuration is
        present; using the same model is allowed only as a temporary fallback.
- [x] Implement `eval/run_eval.py` CLI:
  - [x] For each golden set item, run RAG QA chain.
  - [x] Compute metrics and write JSON results to `reports/` directory.
- [x] Create `docs/eval_baseline.md` and `docs/eval_final.md` with
      baseline and tuned metrics.

**How to run this phase independently**

- [x] Run: `python -m eval.run_eval`.
- [x] Inspect metrics and compare against RAG extension acceptance thresholds for Claims (Hit@5, MRR,
      faithfulness, answer correctness, LLM-judge average, citation coverage).

**Acceptance criteria**

- [x] Metrics meet or approach the thresholds listed for "Claims / Insurance" in the RAG extension
      document.
- [x] Citation coverage is 100% for factual claims in the golden set.
- [x] Evaluation results and methodology documents are committed.

---

## Phase 14 - Agent integration, new endpoints, and tool routing

**Goal**  
Expose the RAG retrieval tool to the existing claims agent, add RAG extension endpoints, and ensure the
agent can route between deterministic tools and RAG based on intent.

**Tasks**

- [x] Extend `app/tools/knowledge_retrieval.py` with a `knowledge_retrieval` tool
      that calls `qa_chain`.
- [x] Update the agent in `app/chains/agent_chain.py`:
  - [x] Include deterministic tools (claims intake, fraud, settlement) + `knowledge_retrieval`.
  - [x] Use classifier/intent router to decide when to call which tool(s).
  - [x] Allow hybrid responses (tool output + RAG-backed explanation/citations).
- [x] Extend FastAPI endpoints in `app/api/server.py`:
  - [x] `/chat` (POST): now returns retrieval trace + citations when RAG is invoked.
  - [x] `/ingest` (POST): upload docs, run loaders → chunkers → embeddings → upsert; return job ID.
  - [x] `/ingest/status/{job_id}` (GET): poll ingestion job state.
  - [x] `/retrieve` (POST): pure retrieval (no LLM), top-k chunks with scores.
  - [x] `/evaluate` (POST): run evaluation suite against stored golden set.
  - [x] `/sources` (GET): list indexed documents with metadata and chunk count.
  - [x] `/sources/{doc_id}` (DELETE): remove a document and its chunks.
- [x] Update Streamlit frontend to:
  - [x] Show clickable citations that surface the exact chunk.
  - [x] Provide basic document management (list sources, delete source, run RAG-only queries).

**How to run this phase independently**

- [x] Run backend: `uvicorn app.main:app --reload`.
- [x] Test new endpoints via Swagger UI (`/docs`) and via HTTPie/curl.
- [x] Confirm `/chat` uses RAG tool when required and includes citations.

**Acceptance criteria**

- [x] All core assistant endpoints continue to work; new RAG extension endpoints are added without regression.
- [x] Agent can route between tools and RAG based on intent/classification.
- [x] RAG-backed answers in `/chat` include traceable citations (doc name/page/section).

---

## Phase 15 - Acceptance criteria validation and sign-off

**Goal**  
Systematically validate the project requirements and acceptance thresholds for the Claims Processing
& Settlement Automation assistant.

**Tasks**

- [x] Create `docs/project_acceptance_mapping.md` that lists each acceptance criterion and maps it to:
  - [x] Implemented module(s).
  - [x] Test(s) or evaluation metric(s).
  - [x] Evidence (LangSmith trace ID, screenshot, eval JSON path).
- [x] Add tests in `tests/test_rag_pipeline.py` and `tests/test_api_rag_endpoints.py`:
  - [x] Cover ingestion endpoints, retrieval-only endpoint, and RAG-backed `/chat` behavior.
- [x] Run `python -m eval.run_eval` and capture metrics vs thresholds.
- [x] Capture LangSmith traces for at least 30 sample RAG + hybrid queries showing run trees.
- [x] Generate a sign-off document `docs/project_signoff_report.md` summarizing:
  - [x] Which thresholds are met.
  - [x] Any gaps and mitigation notes.

**How to run this phase independently**

- [x] Run: `PYTHONPATH=. pytest -q tests/test_rag_pipeline.py tests/test_api_rag_endpoints.py`.
- [x] Run: `python -m eval.run_eval`.
- [x] Review `docs/project_signoff_report.md` prior to submission.

**Acceptance criteria**

- [x] Every project requirement and acceptance threshold from the requirements PDF has a clear,
      implemented check and evidence.
- [x] Test suite and evaluation harness pass without critical failures.
- [x] Existing functionality remains intact (no regressions), and the RAG layer is demonstrably
      production-grade for the training context.


---

## Reviewer remediation phases (R1–R6)

> These phases fix the Week 6 reviewer findings and adjacent gaps discovered during code review:
> (1) hybrid search with reranking not active on the live path, and (2) vector store is in-memory
> instead of persistent. Root cause (confirmed): `app/tools/knowledge_retrieval.py` calls
> `run_qa_chain(...)` without an `embedding_fn`, so dense retrieval is disabled (BM25-only), the
> cross-encoder `rerank_score` silently falls back to the fusion score, and `app/rag/qa_chain.py`
> re-chunks the manifest per request instead of loading the persisted FAISS index. Each phase is
> independently runnable and must not regress existing flows.

---

## Phase R1 - Diagnose and capture baseline evidence (no functional code changes)

**Goal**
Prove exactly what the live retrieval path does today so fixes are targeted and reviewer-verifiable.

**Tasks**

- [ ] Add `scripts/diagnose_retrieval_path.py` that runs one KB query end-to-end and prints:
  - [ ] Which retriever is effectively used (dense+BM25 vs BM25-only) based on whether `embedding_fn` is not None.
  - [ ] Whether a persisted FAISS index was loaded vs rebuilt in memory.
  - [ ] Whether `rerank_score` is present on results (i.e., cross-encoder actually ran) vs fallback to `combined_score`.
- [ ] Inspect `app/rag/retriever_hybrid.py` and record its behavior when `embedding_fn=None` and whether it invokes the cross-encoder.
- [ ] Capture a baseline LangSmith trace and terminal output into `reports/remediation_baseline.md`.

**How to run this phase independently**

- [ ] Run: `PYTHONPATH=. python scripts/diagnose_retrieval_path.py`.

**Acceptance criteria**

- [ ] The script clearly reports the active retriever mode, persistence mode, and rerank status.
- [ ] `reports/remediation_baseline.md` documents the "before" state as evidence for the reviewer.
- [ ] No behavior changes are introduced in this phase.

---

## Phase R2 - Persistent vector store on the live path (fixes "in-memory" finding)

**Goal**
Load a persisted FAISS index at query time instead of re-chunking and re-embedding the manifest on
every request; ensure the index survives process/container restarts.

**Tasks**

- [ ] Add `VECTOR_PERSIST_PATH` to `app/config.py` and `.env.example` (default `data/faiss_index`).
- [ ] In `app/rag/vectorstores/faiss_store.py`, implement/verify `persist(path)` and `load(path)` plus a
      "load-if-exists, else build-then-persist" guard.
- [ ] Refactor `app/rag/qa_chain.py` so `_build_qa_payload` retrieves against the persisted store:
  - [ ] Remove per-request `_load_chunks_from_manifest()` from the default query path.
  - [ ] Load chunks/index once (cached module-level or via the vector store) and reuse across queries.
- [ ] Ensure ingestion (`python -m app.rag.ingest_basic`) writes to `VECTOR_PERSIST_PATH`.
- [ ] Keep the `chunks=` override argument for tests, but the live default must use the persisted store.

**How to run this phase independently**

- [ ] Run ingestion once: `python -m app.rag.ingest_basic`.
- [ ] Restart the process, then run: `PYTHONPATH=. python scripts/diagnose_retrieval_path.py`.

**Acceptance criteria**

- [ ] After a restart, retrieval loads the persisted index (no re-embedding), confirmed by the diagnostic script.
- [ ] `PYTHONPATH=. pytest -q tests/test_vector_backend.py tests/test_sqlite_persistence.py` passes.
- [ ] "Vector store persists across container restarts (volume-mounted)" NFR is demonstrably met.

---

## Phase R3 - Restore true hybrid retrieval end-to-end (fixes "hybrid not implemented", part 1)

**Goal**
Ensure dense retrieval is actually active by threading an embedding function through the agent tool
and QA chain, so BM25 + dense fusion runs on the live path.

**Tasks**

- [ ] Update `app/tools/knowledge_retrieval.py` to obtain the embedding function via
      `app/rag/embeddings.py::get_embedding_fn(EMBEDDING_MODEL)` and pass it into `run_qa_chain(...)`.
- [ ] Verify `app/rag/qa_chain.py` forwards `embedding_fn` into `hybrid_retrieve(...)` (it already accepts it).
- [ ] In `app/rag/retriever_hybrid.py`, ensure dense scoring runs when `embedding_fn` is provided and that
      BM25 is built over the SAME persisted chunks (no separate in-memory corpus).
- [ ] Add a `RETRIEVER_MODE` config (`hybrid` default) so the mode is explicit and testable.

**How to run this phase independently**

- [ ] Run: `PYTHONPATH=. python scripts/diagnose_retrieval_path.py` and confirm mode = hybrid (dense+BM25).
- [ ] Run: `python -m app.rag.qa_demo` and confirm fused BM25+dense candidates appear in the trace.

**Acceptance criteria**

- [ ] The diagnostic script reports dense+BM25 (not BM25-only) on the live path.
- [ ] Hybrid retrieval improves top-5 recall versus BM25-only on the golden set.

---

## Phase R4 - Activate cross-encoder reranking (fixes "hybrid not implemented", part 2)

**Goal**
Guarantee a real reranking stage reorders fused candidates before answer generation, with no silent
fallback to the fusion score.

**Tasks**

- [ ] Wire `app/rag/reranker.py` to use a `sentence-transformers` cross-encoder
      (`cross-encoder/ms-marco-MiniLM-L-6-v2`); add `RERANKER_MODEL` to config/`.env.example`.
- [ ] Ensure `hybrid_retrieve(...)` (or `qa_chain`) invokes the reranker and sets `rerank_score` on every result.
- [ ] Replace the silent fallback in `qa_chain.py` (`result.get("rerank_score", ...combined_score)`) so that a
      missing `rerank_score` raises or logs a warning rather than passing silently.
- [ ] Log top-k ordering before vs after rerank for verifiability.

**How to run this phase independently**

- [ ] Run: `python -m app.rag.qa_demo` and confirm each candidate has a `rerank_score` and order changes post-rerank.

**Acceptance criteria**

- [ ] `rerank_score` is present on all results; the diagnostic script reports rerank = active.
- [ ] Candidate ordering demonstrably changes after reranking on at least one sample query.

---

## Phase R5 - LLM-grounded answer generation with per-claim citations (adjacent gap)

**Goal**
Replace the raw-excerpt "answer" with a real LLM-synthesized answer where every factual claim is
grounded to a `chunk_id`, satisfying the Phase 12/13 citation rule.

**Tasks**

- [ ] In `app/rag/qa_chain.py`, add an answer-generation step that sends reranked top-k chunks to the LLM
      with a citation-enforcing prompt (each claim must reference a `chunk_id`).
- [ ] Return `answer_text` synthesized by the LLM (not a truncated excerpt), preserving the existing
      `citations` list and `confidence` fields.
- [ ] Keep `stream_qa_chain` behavior working over the new generated answer.
- [ ] Enforce refusal/"insufficient context" behavior when retrieval returns nothing relevant.

**How to run this phase independently**

- [ ] Run: `python -m app.rag.qa_demo` on clause/exclusion/regulatory queries and confirm synthesized answers with inline citations.

**Acceptance criteria**

- [ ] Every factual claim in the answer references a `chunk_id`; citation coverage is 100% on the golden set.
- [ ] Answers are LLM-synthesized (not verbatim chunk excerpts) and respect the latency NFR (< 8s tool-augmented).

---

## Phase R6 - Prove the fix and lock against regression (reviewer-facing evidence)

**Goal**
Turn the remediation into reviewer-ready before/after evidence and guard it with tests.

**Tasks**

- [ ] Re-run `python -m eval.run_eval`; record before/after Hit@5 and MRR in `docs/eval_final.md`.
- [ ] Update `tests/test_rag_hybrid_reranking_streaming.py` to assert the LIVE path (via `knowledge_retrieval`)
      uses dense+BM25 and sets `rerank_score` (not a standalone import).
- [ ] Add a persistence regression test asserting the index is loaded (not rebuilt) on a second call.
- [ ] Capture LangSmith traces + a screenshot for submission; note improvements in `docs/project_signoff_report.md`.

**How to run this phase independently**

- [ ] Run: `PYTHONPATH=. pytest -q tests/test_rag_hybrid_reranking_streaming.py tests/test_rag_pipeline.py`.
- [ ] Run: `python -m eval.run_eval`.

**Acceptance criteria**

- [ ] Tests assert hybrid + rerank on the live path and vector-store persistence across calls.
- [ ] `docs/eval_final.md` shows measurable retrieval improvement over the BM25-only baseline.
- [ ] Reviewer findings (hybrid+rerank, persistence) are closed with committed evidence.
---

## Week 8 extension - Production-grade capabilities (MCP · LCEL · HITL · Prompt Versioning · Role-Based RAG · Advanced Eval)

These phases extend the **corrected Week 6 baseline** (RAG foundations + reviewer remediation R1-R6)
with the Week 8 production-hardening layer. They move the assistant from prototype to deployment-ready:
external tool interoperability via MCP, LCEL-based LangChain orchestration, Human-in-the-Loop (HITL)
approval gates, YAML prompt version control, role-based RAG, and an expanded evaluation/regression suite.

**Design principles for the whole Week 8 block (read before starting):**

- **Non-regression is mandatory.** Every Week 2 / Week 4 / Week 6 / R1-R6 test must continue to pass.
  This is the final evaluation project; a regression is a failure.
- **All new modules live under the existing `app.*` package** (e.g. `app/mcp/`, `app/hitl/`,
  `app/rbac/`, `app/prompt_manager/`, `app/callbacks/`, `app/agents/`, `app/drift/`), so imports stay
  consistent with the current codebase. YAML configs live in `config/`; prompt templates in `prompts/`.
- **Each phase is independently runnable** with its own validation command and acceptance criteria.
  Do not merge phases or skip acceptance gates.
- **Feature flags default OFF** (`ENABLE_MCP`, `ENABLE_HITL`, `ENABLE_RBAC`, etc.) so a half-built
  phase can never break the live Week 6 path.
- **Models:** generation stays on `gpt-4o-mini` (trainer constraint); the LLM-as-judge uses an
  **independent Claude model** for eval, with a **Python semantic-similarity scorer** as an offline
  fallback when the judge key is unavailable. Never let eval silently fall back to the generation model.
- **Secrets** are injected via environment only; `.env` is never committed and never baked into images.
- You may adapt prompt/chain patterns from classroom training material, but keep all code organised
  into the modules indicated below.

### Week 8 requirement -> Phase mapping

| Week 8 Requirement (spec section) | Phase |
| --- | --- |
| Week 8 bootstrap: packages, `config/*.yaml`, `.env` keys, deps (non-breaking) | W8-0 |
| LangChain orchestration -> LCEL chains, callbacks, chain registry/router (3.2) | W8-1 |
| Prompt version control via YAML + `/prompts*` endpoints (3.4) | W8-2 |
| MCP integration: client/registry/auth/adapter + `/mcp/*` endpoints (3.1) | W8-3 |
| HITL workflows: manager/triggers/persistent store + `/hitl/*` endpoints (3.3) | W8-4 |
| Role-Based RAG: JWT auth, pre+post-retrieval filter, audit + `/roles`,`/auth/context` (3.6) | W8-5 |
| Expanded evaluation: regression suite, custom metrics, comparator, export + `/eval/*` (3.5) | W8-6 |
| Streamlit UI: HITL panel, role selector, prompt viewer, eval dashboard (5.2) | W8-7 |
| Containerisation: Dockerfile + docker-compose full stack (5.2) [optional-advantage] | W8-8 |
| [Optional] A2A / multi-agent orchestration (3.7) | W8-9 |
| [Optional] Prompt & embedding drift detection (3.8) | W8-10 |
| Extensive validation: full regression + Week 8 acceptance thresholds (Section 4) | W8-V |
| requirements.txt verification + deployment cleanup (diagnostics/plan removal) | W8-C |

### Week 8 new modules (all under `app/`, configs under `config/`, prompts under `prompts/`)

```
app/
├── mcp/                     # W8-3  MCP integration
│   ├── __init__.py  client.py  registry.py  auth.py  tool_adapter.py
├── chains/                  # W8-1  LCEL orchestration (extends existing app/chains)
│   ├── base_lcel.py  router.py  rag_chain_lcel.py  tool_chain_lcel.py  hitl_chain.py
├── callbacks/               # W8-1  custom LangChain callbacks
│   ├── __init__.py  logging_cb.py  tracing_cb.py  metrics_cb.py
├── hitl/                    # W8-4  Human-in-the-Loop
│   ├── __init__.py  manager.py  models.py  triggers.py  store.py
├── prompt_manager/          # W8-2  Prompt version control
│   ├── __init__.py  loader.py  registry.py  validator.py  models.py
├── rbac/                    # W8-5  Role-based access control
│   ├── __init__.py  auth.py  models.py  filter.py  validator.py  audit.py
├── agents/                  # W8-9  [optional] multi-agent
│   ├── __init__.py  registry.py  decomposer.py  dispatcher.py  aggregator.py
└── drift/                   # W8-10 [optional] drift detection
    ├── __init__.py  baseline.py  prompt_drift.py  embedding_drift.py  detector.py  alerts.py
prompts/                     # W8-2  versioned YAML prompt templates
├── rag_qa.yaml  agent_system.yaml  hitl_review.yaml  faq.yaml  guardrail.yaml
config/                      # W8-0  all YAML configs
├── mcp_servers.yaml  hitl_rules.yaml  roles.yaml  drift_thresholds.yaml  agents.yaml
eval/                        # W8-6  extended evaluation (extends existing eval/)
├── regression_suite.py  custom_metrics.py  comparator.py  export.py  dashboard.py
docker/                      # W8-8  [optional-advantage] containerisation
├── Dockerfile
docker-compose.yml  .dockerignore
```

---

## Phase W8-0 - Week 8 bootstrap and non-breaking scaffolding

**Goal**
Create empty module packages, `config/*.yaml` stubs, new `.env` keys (feature flags OFF), and dependency
placeholders so all later Week 8 phases import cleanly with zero behaviour change to the Week 6 path.

**Requirement traceability**
Prerequisite for spec Sections 3.1-3.8, 6 (repository layout), and 7 (tech stack).

**Files to create/modify**

- Create packages: `app/mcp/__init__.py`, `app/callbacks/__init__.py`, `app/hitl/__init__.py`,
  `app/prompt_manager/__init__.py`, `app/rbac/__init__.py`, `app/agents/__init__.py`, `app/drift/__init__.py`.
- Create `config/` with valid-but-empty stubs: `mcp_servers.yaml`, `hitl_rules.yaml`, `roles.yaml`,
  `drift_thresholds.yaml`, `agents.yaml`.
- Create `prompts/` directory (empty for now; populated in W8-2).
- Update `app/config.py` and `.env.example` with the new keys below.
- Update `requirements.txt` header comment reserving Week 8 deps (added incrementally per phase).

**Tasks**

- [ ] Add feature-flag + path settings to `app/config.py` (Pydantic `BaseSettings`) with safe defaults:
      `ENABLE_MCP=false`, `ENABLE_HITL=false`, `ENABLE_RBAC=false`, `ENABLE_DRIFT=false`,
      `ENABLE_MULTI_AGENT=false`.
- [ ] Add path/model keys: `PROMPTS_DIR=prompts`, `MCP_REGISTRY_PATH=config/mcp_servers.yaml`,
      `HITL_RULES_PATH=config/hitl_rules.yaml`, `HITL_STORE_PATH=data/hitl_tasks.db`,
      `ROLES_CONFIG_PATH=config/roles.yaml`, `DRIFT_THRESHOLDS_PATH=config/drift_thresholds.yaml`,
      `JWT_SECRET`, `JWT_ALGORITHM=HS256`, `JUDGE_MODEL_NAME`, `JUDGE_ANTHROPIC_API_KEY`.
- [ ] Mirror every new key in `.env.example` with an inline comment; keep `.env` gitignored.
- [ ] Ensure each new package exposes a no-op importable surface so `import app.<pkg>` resolves.
- [ ] Do NOT modify any existing module logic, endpoint, or test.

**How to run this phase independently**

- [ ] `PYTHONPATH=. python -c "import app.mcp, app.callbacks, app.hitl, app.prompt_manager, app.rbac, app.agents, app.drift; print('imports-ok')"`
- [ ] `PYTHONPATH=. python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('config/*.yaml')]; print('yaml-ok')"`
- [ ] `PYTHONPATH=. pytest -q` (full existing suite must stay green).

**Acceptance criteria**

- [ ] All new packages import without error.
- [ ] All `config/*.yaml` stubs parse as valid YAML.
- [ ] New settings load via `get_settings()` with defaults; feature flags default to OFF.
- [ ] Zero change to existing endpoints, tools, or test outcomes.

**No-regression guardrail**
No existing file behaviour is altered; only additive scaffolding and OFF-by-default flags.

**Cline task prompt**

> Create Week 8 scaffolding only, no functional logic. Add empty importable packages `app/mcp`,
> `app/callbacks`, `app/hitl`, `app/prompt_manager`, `app/rbac`, `app/agents`, `app/drift`. Create
> `config/` with valid empty YAML stubs (`mcp_servers.yaml`, `hitl_rules.yaml`, `roles.yaml`,
> `drift_thresholds.yaml`, `agents.yaml`) and an empty `prompts/` dir. Extend `app/config.py` and
> `.env.example` with the feature flags and paths listed in this phase (all flags default false). Do not
> modify existing module logic, endpoints, or tests. Validate by running the import check, the YAML-parse
> check, and the full `PYTHONPATH=. pytest -q` suite. Stop when all three pass.

---

## Phase W8-1 - LCEL orchestration and custom callbacks

**Goal**
Restructure the agent orchestration around LangChain Expression Language (LCEL) with a chain registry/router
mapping intent -> chain, chain-level retry/fallback, streaming, and custom callbacks for structured logging,
LangSmith tracing, and metric emission - while wrapping (not replacing) the existing agent/RAG/tool logic.

**Requirement traceability**
Spec 3.2 (LangChain Orchestration Restructuring); deliverables `chains/{base,rag_chain,tool_chain,hitl_chain,router}.py`,
`callbacks/{logging,tracing,metrics}.py`, `tests/test_chains.py`.

**Files to create/modify**

- Create `app/chains/base_lcel.py`, `app/chains/router.py`, `app/chains/rag_chain_lcel.py`,
  `app/chains/tool_chain_lcel.py`, `app/chains/hitl_chain.py` (placeholder wiring for W8-4).
- Create `app/callbacks/{logging_cb,tracing_cb,metrics_cb}.py`.
- Light touch: `app/chains/agent_chain.py` (delegate to router; keep original callable intact),
  `app/api/server.py` (route `/chat` through the router behind a flag).
- Create `tests/test_chains.py`.

**Tasks**

- [ ] Implement `base_lcel.py` with composable primitives
      (`RunnablePassthrough | RunnableLambda | RunnableParallel`) and shared model/prompt builders.
- [ ] Implement `router.py`: a chain registry mapping `FAQIntent` categories to specific chains, with
      `.with_retry()` and `.with_fallbacks()` for graceful degradation.
- [ ] Wrap existing FAQ/RAG/tool flows as `rag_chain_lcel.py` and `tool_chain_lcel.py` (reuse current logic;
      do not rewrite tool internals).
- [ ] Implement custom callbacks emitting structured JSON (via existing `json_logger`) and LangSmith spans;
      propagate request-scoped metadata (session id, trace id, and later role) via `RunnableConfig`.
- [ ] Add streaming support (`.stream()`/`.astream()`).
- [ ] Route `/chat` through the router only when a flag is set; keep the legacy path as fallback.

**How to run this phase independently**

- [ ] `PYTHONPATH=. pytest -q tests/test_chains.py`
- [ ] `PYTHONPATH=. python -m app.chains.router --smoke` (prints the selected chain per sample intent).
- [ ] `PYTHONPATH=. pytest -q` (full regression).

**Acceptance criteria**

- [ ] Router selects the correct chain per intent; retry/fallback demonstrably engages on a forced error.
- [ ] Callbacks emit structured JSON lines and LangSmith spans for a `/chat` call.
- [ ] `/chat` request/response contract is byte-compatible with the Week 6 behaviour.
- [ ] All Week 2/4/6 + R1-R6 tests still pass.

**No-regression guardrail**
The original `agent_chain` callable remains available; the LCEL path is additive and flag-gated with the
legacy path as fallback.

**Cline task prompt**

> Introduce LCEL orchestration without breaking existing flows. Build `app/chains/base_lcel.py` (shared
> runnables), `router.py` (intent->chain registry with `.with_retry()` and `.with_fallbacks()`),
> `rag_chain_lcel.py` and `tool_chain_lcel.py` that WRAP the current RAG/tool logic, and a placeholder
> `hitl_chain.py`. Add `app/callbacks/{logging_cb,tracing_cb,metrics_cb}.py` attached via RunnableConfig.
> Route `/chat` through the router behind a flag, keeping the legacy path as fallback. Preserve the `/chat`
> I/O contract exactly. Validate with `pytest -q tests/test_chains.py`, the `--smoke` router check, and the
> full existing suite. Do not delete or rewrite `agent_chain.py` internals.

---

## Phase W8-2 - Prompt version control via YAML

**Goal**
Externalise every prompt into versioned YAML files with metadata (version, author, changelog,
`model_compatibility`, `input_variables`), load at startup with hot-reload, and expose `/prompts`,
`/prompts/{name}/history`, `/prompts/{name}/activate` (rollback in <30s). No hardcoded prompt strings remain.

**Requirement traceability**
Spec 3.4 (Prompt Version Control via YAML); deliverables `prompts/{*.yaml}`,
`prompt_manager/{loader,registry,validator,models}.py`, `tests/test_prompt_versioning.py`;
NFR: rollback <30s; pitfall: no hardcoded prompt survives migration.

**Files to create/modify**

- Create `prompts/{rag_qa,agent_system,hitl_review,faq,guardrail}.yaml` (>=5 templates).
- Create `app/prompt_manager/{loader,registry,validator,models}.py`.
- Modify `app/chains/*`, `app/tools/guardrails.py`, `app/prompts/*` consumers to fetch from the registry.
- Add `/prompts` endpoints in `app/api/server.py`.
- Create `tests/test_prompt_versioning.py`.

**Tasks**

- [ ] Define a Pydantic `PromptTemplate` model (name, version, author, changelog[], model_compatibility[],
      input_variables[], template).
- [ ] `loader.py`: parse all `prompts/*.yaml`; `watchdog`-based hot-reload without restart.
- [ ] `registry.py`: in-memory registry keyed by name+version with an "active version" pointer;
      `activate(name, version)` for rollback.
- [ ] `validator.py`: assert required metadata present and that `input_variables` match placeholders.
- [ ] Replace all inline prompt strings in `app/` with `registry.get(name).render(**vars)`.
- [ ] Endpoints: `GET /prompts`, `GET /prompts/{name}/history` (with diffs), `POST /prompts/{name}/activate`.

**How to run this phase independently**

- [ ] `PYTHONPATH=. pytest -q tests/test_prompt_versioning.py`
- [ ] Hardcode guard (must return nothing): `! grep -rnE 'template\s*=\s*"""' app/ --include=*.py`
- [ ] Rollback timing check: `PYTHONPATH=. python -m app.prompt_manager.registry --activate rag_qa_chain 1.0.0`

**Acceptance criteria**

- [ ] No hardcoded prompt strings remain in `app/*.py` (grep guard clean).
- [ ] `/prompts/{name}/activate` performs rollback in <30s and takes effect at runtime.
- [ ] Each YAML prompt version is covered by >=1 golden-set test case.
- [ ] Hot-reload picks up a YAML edit without process restart.

**No-regression guardrail**
Chains resolve prompts through the registry with the migrated templates producing identical rendered text
to the previous inline versions (verified against golden set).

**Cline task prompt**

> Migrate all prompts to versioned YAML under `prompts/` (>=5 templates: rag_qa, agent_system, hitl_review,
> faq, guardrail). Build `app/prompt_manager/{loader,registry,validator,models}.py` with startup load plus
> `watchdog` hot-reload and an active-version pointer supporting rollback. Add FastAPI `/prompts`,
> `/prompts/{name}/history`, `/prompts/{name}/activate`. Replace every inline prompt string in `app/` with a
> registry lookup. Validate with `pytest -q tests/test_prompt_versioning.py`, a CI grep proving zero inline
> prompts, and a rollback that completes <30s. Do not change rendered prompt text versus the current inline
> versions.

---

## Phase W8-3 - MCP (Model Context Protocol) integration

**Goal**
Enable the agent to discover and invoke external tools exposed as MCP servers defined in
`config/mcp_servers.yaml`, translating MCP tool schemas into LangChain tools, handling per-server auth and
connection lifecycle (health checks, retries, timeouts). Expose `/mcp/tools` and `/mcp/invoke`.

**Requirement traceability**
Spec 3.1 (MCP Integration); Claims MCP servers: hospital network directory, policy administration system,
fraud detection scoring, IRDAI compliance API; deliverables `mcp/{client,registry,auth,tool_adapter}.py`,
`config/mcp_servers.yaml`, `tests/test_mcp_integration.py`; NFR: invocation <3s; threshold: success rate >=95%.

**Files to create/modify**

- Create `app/mcp/{client,registry,auth,tool_adapter}.py`.
- Populate `config/mcp_servers.yaml` with the four Claims servers per spec (with local stub MCP servers so
  the graded run is reproducible offline; each server implemented to the MCP spec).
- Create local MCP server implementations under `app/mcp/servers/` (hospital_network, policy_admin,
  fraud_scoring, irdai_compliance) that speak the protocol per spec.
- Register MCP tools with the LCEL agent (W8-1) behind `ENABLE_MCP`.
- Create `tests/test_mcp_integration.py`.

**Tasks**

- [ ] `registry.py`: read `config/mcp_servers.yaml`; hold server descriptors (name, url, capabilities, auth).
- [ ] `auth.py`: per-server credential resolution (API key / OAuth token) from env.
- [ ] `client.py`: connection lifecycle - discover tools, health check, retry with backoff, timeout (<3s).
- [ ] `tool_adapter.py`: translate MCP tool schema -> LangChain `StructuredTool` with typed I/O.
- [ ] Implement the four Claims MCP servers to spec so `/mcp/tools` lists them and `/mcp/invoke` works.
- [ ] Endpoints: `GET /mcp/tools` (with connection status), `POST /mcp/invoke` (by name + params).
- [ ] Agent treats MCP tools identically to local tools; selection driven by clear tool descriptions.

**How to run this phase independently**

- [ ] Start stub servers: `PYTHONPATH=. python -m app.mcp.servers --all &`
- [ ] `PYTHONPATH=. pytest -q tests/test_mcp_integration.py`
- [ ] `curl -s localhost:8000/mcp/tools | jq` and a sample `POST /mcp/invoke`.

**Acceptance criteria**

- [ ] >=2 MCP servers (target: all 4) discoverable via `/mcp/tools` with connection status.
- [ ] `/mcp/invoke` returns valid tool output or a structured error.
- [ ] MCP tool invocation success rate >=95%; per-call latency <3s (excluding third-party time).
- [ ] Agent selects an MCP tool for an appropriate query without hardcoded routing.

**No-regression guardrail**
MCP is flag-gated (`ENABLE_MCP`); with the flag off, the Week 6 tool set behaves exactly as before.

**Cline task prompt**

> Build MCP integration per spec. Implement `app/mcp/{registry,auth,client,tool_adapter}.py` reading
> `config/mcp_servers.yaml`, exposing discovered tools as LangChain StructuredTools with health-check,
> retry-with-backoff, and <3s timeout. Implement the four Claims MCP servers per spec under
> `app/mcp/servers/` (hospital network, policy admin, fraud scoring, IRDAI compliance) so the run is
> reproducible offline. Add `/mcp/tools` and `/mcp/invoke`. Gate everything behind `ENABLE_MCP`. Validate by
> starting the stub servers, running `pytest -q tests/test_mcp_integration.py`, and a `/mcp/invoke` smoke
> call. Require success rate >=95% and latency <3s.

---

## Phase W8-4 - Human-in-the-Loop (HITL) workflows

**Goal**
Introduce approval gates at critical decision points. The agent pauses on a trigger, serialises its
recommendation with full context (retrieved chunks, reasoning trace, confidence) into a persistent task
store, exposes it via `/hitl/pending`, and resumes only on `/hitl/review/{task_id}` (approve/reject + comments).

**Requirement traceability**
Spec 3.3 (HITL Workflows); Claims triggers: claim amount > Rs 5L, claim rejection, partial settlement,
fraud flag raised, policy exclusion applied; deliverables `hitl/{manager,models,triggers,store}.py`,
`config/hitl_rules.yaml`, `tests/test_hitl_workflow.py`; NFR: state persists across restarts;
thresholds: trigger precision >=85%, recall >=95%.

**Files to create/modify**

- Create `app/hitl/{manager,models,triggers,store}.py`.
- Populate `config/hitl_rules.yaml` with the 5 Claims trigger rules.
- Wire `hitl_chain.py` (from W8-1) to pause/resume via the manager.
- Add `/hitl/pending` and `/hitl/review/{task_id}` in `app/api/server.py`.
- Create `tests/test_hitl_workflow.py`.

**Tasks**

- [ ] `models.py`: `HITLTask` (id, status, recommendation, context, confidence, created_at, decision, comments).
- [ ] `store.py`: SQLite-backed persistent store at `HITL_STORE_PATH` (survives restart).
- [ ] `triggers.py`: evaluate configurable rules from `config/hitl_rules.yaml`.
- [ ] `manager.py`: intercept matching actions, serialise context, pause the chain, resume on decision.
- [ ] Endpoints: `GET /hitl/pending`, `POST /hitl/review/{task_id}` with `{decision, comments}`.
- [ ] Ensure `/reset` does not clear pending HITL tasks (they are operational state, not chat memory).

**How to run this phase independently**

- [ ] `PYTHONPATH=. pytest -q tests/test_hitl_workflow.py`
- [ ] End-to-end: submit a > Rs 5L claim -> confirm task in `GET /hitl/pending` -> `POST /hitl/review` approve
      -> confirm agent resumes.
- [ ] Restart-persistence: create a task, restart process, confirm it still appears in `/hitl/pending`.

**Acceptance criteria**

- [ ] Trigger fires for each of the 5 configured Claims conditions; task appears in `/hitl/pending`.
- [ ] Approval/rejection via `/hitl/review` correctly resumes or halts execution.
- [ ] HITL state persists across a container/process restart.
- [ ] Trigger precision >=85% and recall >=95% on the labelled HITL test set.

**No-regression guardrail**
HITL is flag-gated (`ENABLE_HITL`); with the flag off, actions execute exactly as in Week 6.

**Cline task prompt**

> Implement HITL per spec. Build `app/hitl/{models,store,triggers,manager}.py` with a SQLite-backed
> persistent task store at `HITL_STORE_PATH` and trigger rules from `config/hitl_rules.yaml` (Claims:
> amount > Rs 5L, claim rejection, partial settlement, fraud flag raised, policy exclusion applied). The
> chain must pause on a trigger, serialise recommendation + retrieved chunks + reasoning + confidence, and
> resume on decision. Add `/hitl/pending` and `/hitl/review/{task_id}` (body: decision, comments). Ensure
> `/reset` does not delete pending tasks. Gate behind `ENABLE_HITL`. Validate with
> `pytest -q tests/test_hitl_workflow.py`, an end-to-end pause/approve/resume, and a restart test proving
> persistence. Require precision >=85%, recall >=95%.

---

## Phase W8-5 - Role-Based RAG with JWT auth

**Goal**
Restrict document retrieval by the authenticated user's role. Implement JWT-based auth middleware (per spec),
a role-permission matrix, a pre-retrieval metadata filter injected before hybrid retrieval, and a
post-retrieval validator that guarantees no restricted chunk leaks through the re-ranker/cache. Audit every
role-filtered retrieval. Expose `/roles` and `/auth/context`.

**Requirement traceability**
Spec 3.6 (Role-Based RAG); Claims roles: `claims_processor` (policy wordings + SOPs), `senior_adjuster`,
`claims_manager` (+ prior memos), `fraud_investigator` (+ investigation files); deliverables
`rbac/{models,filter,validator,audit}.py`, `config/roles.yaml`, `tests/test_role_based_rag.py`;
NFR: filtering overhead <200ms; threshold: 0% leakage (mandatory); pitfall: role leakage through re-ranker.

**Files to create/modify**

- Create `app/rbac/{auth,models,filter,validator,audit}.py`.
- Populate `config/roles.yaml` with the 4 Claims roles and allowed `doc_type` sets.
- Add JWT auth middleware in `app/api/server.py`; add `/roles` and `/auth/context`.
- Hook filter into `app/rag/retriever_hybrid.py` (pre) and validator after re-rank in `app/rag/qa_chain.py`.
- Create `tests/test_role_based_rag.py`.

**Tasks**

- [ ] `auth.py`: JWT middleware (HS256 via `JWT_SECRET`); decode token -> role + permissions.
- [ ] `models.py`: role-permission matrix (role -> allowed doc_type / metadata filters).
- [ ] `filter.py`: inject role constraints into the vector-store query BEFORE hybrid retrieval.
- [ ] `validator.py`: post-retrieval (after re-rank) check that drops any restricted chunk - defence in depth.
- [ ] `audit.py`: log every role-filtered retrieval (role, query, allowed doc_types, returned chunk ids).
- [ ] Endpoints: `GET /roles`, `GET /auth/context` (current role, permissions, accessible doc types).
- [ ] Propagate role via `RunnableConfig` (from W8-1) into retrieval.

**How to run this phase independently**

- [ ] `PYTHONPATH=. pytest -q tests/test_role_based_rag.py`
- [ ] Leakage test: query restricted docs as `claims_processor`; assert zero restricted chunks returned.
- [ ] Overhead check: measure retrieval latency with vs without the filter (<200ms delta).

**Acceptance criteria**

- [ ] JWT auth resolves role for each request; `/auth/context` returns correct permissions.
- [ ] Pre-retrieval filter + post-retrieval validator yield **0% leakage** across all 4 roles.
- [ ] Role filtering adds <200ms to retrieval latency.
- [ ] All role-filtered retrievals are audit-logged.

**No-regression guardrail**
RBAC is flag-gated (`ENABLE_RBAC`); with the flag off, retrieval behaves as in Week 6. A default
"full-access" service role preserves existing tests.

**Cline task prompt**

> Implement Role-Based RAG per spec with JWT auth. Build `app/rbac/{auth,models,filter,validator,audit}.py`
> driven by `config/roles.yaml` (Claims roles: claims_processor, senior_adjuster, claims_manager,
> fraud_investigator with the doc_type permissions from the spec). Add JWT middleware (HS256) and `/roles`,
> `/auth/context`. Inject role metadata filters BEFORE hybrid retrieval and validate AFTER re-rank
> (defence-in-depth against re-ranker/cache leakage). Audit every filtered retrieval. Propagate role via
> RunnableConfig. Gate behind `ENABLE_RBAC` with a default full-access service role for existing tests.
> Validate with `pytest -q tests/test_role_based_rag.py` proving 0% leakage and <200ms overhead.

---

## Phase W8-6 - Expanded RAG evaluation and regression suite

**Goal**
Extend the Week 6 evaluation harness into a production-grade regression + monitoring suite: automated
regression against the golden set, custom evaluation dimensions, A/B comparator (prompt/model versions),
JSON/CSV export for CI, and `/eval/regression` + `/eval/drift` endpoints. The LLM-as-judge uses an
**independent Claude model**, with a **Python semantic-similarity scorer** as offline fallback.

**Requirement traceability**
Spec 3.5 (RAG Evaluation - Expanded); deliverables `eval/{regression_suite,custom_metrics,comparator,export,dashboard}.py`,
`reports/eval_week8_baseline.md`, `reports/eval_week8_final.md`; new metrics: Golden Set Pass Rate >=95%,
Answer Stability >=0.90, Regulatory Compliance >=0.90, Role Appropriateness 100%, HITL Trigger Precision >=0.85.

**Files to create/modify**

- Create `eval/regression_suite.py`, `eval/custom_metrics.py`, `eval/comparator.py`, `eval/export.py`.
- Extend `eval/llm_judge.py` to require an independent Claude judge (`JUDGE_MODEL_NAME` +
  `JUDGE_ANTHROPIC_API_KEY`) and add a Python semantic-similarity fallback scorer (sentence-transformers
  cosine) when the judge key is absent - never fall back to the generation model.
- Add `/eval/regression` and `/eval/drift` endpoints (drift detail lands in W8-10; endpoint stub here).
- Create `reports/eval_week8_baseline.md` (capture BEFORE other Week 8 changes) and `eval_week8_final.md`.
- Extend `tests/test_rag_evaluation_harness.py`.

**Tasks**

- [ ] `custom_metrics.py`: Regulatory Compliance Score, Role Appropriateness, HITL Trigger Precision,
      Answer Stability (semantic similarity across re-runs), Golden Set Pass Rate.
- [ ] `regression_suite.py`: run the full golden set, compare against all Week 6 + Week 8 thresholds,
      emit pass/fail per item and a summary.
- [ ] `comparator.py`: A/B two prompt or model versions with randomised label order (reduce position bias).
- [ ] `export.py`: JSON/CSV export for CI integration.
- [ ] Judge: independent Claude via `JUDGE_MODEL_NAME`; **fallback = Python cosine-similarity scorer**,
      logged clearly as a fallback; never reuse `gpt-4o-mini`.
- [ ] Endpoints: `POST /eval/regression`, `POST /eval/drift` (stub -> full in W8-10).

**How to run this phase independently**

- [ ] `PYTHONPATH=. python -m eval.regression_suite`
- [ ] `PYTHONPATH=. pytest -q tests/test_rag_evaluation_harness.py`
- [ ] `curl -s -X POST localhost:8000/eval/regression | jq '.summary'`

**Acceptance criteria**

- [ ] Golden Set Pass Rate >=95%; Answer Stability >=0.90.
- [ ] All Week 6 thresholds still met (Hit@5 >=0.85, MRR >=0.65, Faithfulness >=0.90,
      Answer Correctness >=0.80, LLM-Judge >=4.0, Citation 100%).
- [ ] Judge is a different model family from the generator; fallback scorer engages only when judge key is
      absent and is clearly logged.
- [ ] `reports/eval_week8_baseline.md` and `eval_week8_final.md` show before/after metrics.

**No-regression guardrail**
The Week 6 `run_eval` remains functional; the regression suite is additive and reuses the same golden set.

**Cline task prompt**

> Extend evaluation into a regression + monitoring suite. Build `eval/{regression_suite,custom_metrics,
> comparator,export}.py` and add `/eval/regression` plus an `/eval/drift` stub. Custom metrics: Regulatory
> Compliance, Role Appropriateness, HITL Trigger Precision, Answer Stability, Golden Set Pass Rate. Update
> `eval/llm_judge.py` to REQUIRE an independent Claude judge (`JUDGE_MODEL_NAME` + `JUDGE_ANTHROPIC_API_KEY`)
> and add a Python sentence-transformers cosine-similarity fallback scorer used ONLY when the judge key is
> absent - never fall back to `gpt-4o-mini`. Capture `reports/eval_week8_baseline.md` before other Week 8
> changes and `eval_week8_final.md` after. Validate with `python -m eval.regression_suite`
> (pass rate >=95%, stability >=0.90) and `pytest -q tests/test_rag_evaluation_harness.py`.

---

## Phase W8-7 - Streamlit UI extension

**Goal**
Extend the Streamlit frontend with a HITL approval panel (full context view), a role selector, a prompt-version
viewer, and an evaluation trend dashboard - all calling the new FastAPI endpoints (no direct chain calls).

**Requirement traceability**
Spec 5.2 (Runtime & Evaluation): Streamlit UI extended with HITL approval panel, role selector, prompt
version viewer, and eval dashboard.

**Files to create/modify**

- Modify `frontend/streamlit_app.py`.
- Create `eval/dashboard.py` (trend graphs consumed by the Streamlit eval page).

**Tasks**

- [ ] HITL panel: list `/hitl/pending`, show retrieved chunks + reasoning trace + confidence + proposed
      action, with approve/reject + comments -> `/hitl/review/{task_id}`.
- [ ] Role selector: obtain a JWT for the chosen role; show `/auth/context`; pass token on chat calls.
- [ ] Prompt viewer: list `/prompts`, show `/prompts/{name}/history` diffs, trigger `activate` (rollback).
- [ ] Eval dashboard: render week-over-week metric trends from `/eval/regression` results.
- [ ] Keep clickable citations from Week 6 working; UI must only call FastAPI.

**How to run this phase independently**

- [ ] `streamlit run frontend/streamlit_app.py` (backend running) and exercise each panel.
- [ ] Confirm approve/reject round-trips against `/hitl/*` and role switching changes retrievable docs.

**Acceptance criteria**

- [ ] All four panels function against the live API.
- [ ] HITL approval panel shows full context (chunks, reasoning, confidence) - not a bare button.
- [ ] Role selector demonstrably changes what documents are retrievable.
- [ ] No direct chain/tool imports in the frontend.

**No-regression guardrail**
Existing chat + citation UI remains intact; new panels are additive tabs/sections.

**Cline task prompt**

> Extend `frontend/streamlit_app.py` with four additions, all calling FastAPI only: (1) HITL approval panel
> listing `/hitl/pending`, showing chunks/reasoning/confidence/proposed-action with approve/reject+comments
> to `/hitl/review/{task_id}`; (2) role selector that fetches a JWT per role, shows `/auth/context`, and
> passes the token on chat calls; (3) prompt-version viewer using `/prompts`, `/prompts/{name}/history`,
> `/prompts/{name}/activate`; (4) eval dashboard (`eval/dashboard.py`) rendering trend graphs from
> `/eval/regression`. Keep Week 6 clickable citations working. Validate by launching Streamlit and
> exercising each panel end-to-end.

---

## Phase W8-8 - Containerisation (optional, evaluation advantage)

**Goal**
Author Docker artefacts so `docker-compose up` brings up API + UI + vector store + new services from a clean
machine, with volume-mounted persistence and env-injected secrets. Marked optional (shared VM has disk
constraints) but adds evaluation advantage; must be runnable locally.

**Requirement traceability**
Spec 5.2 / 10: `docker-compose up` brings up the complete Week 8 system; secrets via env only (pitfall:
secrets in images). NFR: vector store + HITL state persist via volume mounts.

**Files to create/modify**

- Create `docker/Dockerfile`, `docker-compose.yml`, `.dockerignore`.

**Tasks**

- [ ] `Dockerfile`: Python 3.11 base; install pinned `requirements.txt`; non-root user; no secrets baked.
- [ ] `docker-compose.yml`: services for FastAPI (uvicorn) and Streamlit; volume-mount `data/faiss_index`
      and `HITL_STORE_PATH`; inject `.env` via `env_file`.
- [ ] `.dockerignore`: exclude `.env`, `venv/`, `__pycache__`, diagnostics, `*_SUMMARY.md`, plan files,
      `notebooks/`, `Screenshots/`.
- [ ] Health check hitting `/health`.

**How to run this phase independently**

- [ ] `docker-compose config` (validates compose syntax).
- [ ] `docker-compose up --build` then `curl -s localhost:8000/health`.
- [ ] Confirm FAISS index + HITL DB survive `docker-compose down && up` (volume persistence).

**Acceptance criteria**

- [ ] `docker-compose up` yields a working API + UI on a clean machine.
- [ ] No secrets in the image; `.env` injected at runtime only.
- [ ] Persistent volumes retain vector index and HITL tasks across restarts.

**No-regression guardrail**
Purely additive deployment artefacts; application code is unchanged.

**Cline task prompt**

> Containerise the app (optional but scored-advantageous). Add `docker/Dockerfile` (Python 3.11, pinned
> requirements, non-root, no baked secrets), `docker-compose.yml` (FastAPI + Streamlit services,
> volume-mounted `data/faiss_index` and the HITL store, `.env` via env_file), and a `.dockerignore`
> excluding `.env`, `venv`, caches, diagnostics, `*_SUMMARY.md`, plan files, notebooks, and screenshots.
> Add a `/health` healthcheck. Validate with `docker-compose config`, `docker-compose up --build` + a
> `/health` curl, and a down/up cycle proving volume persistence. Do not modify application logic.

---

## Phase W8-9 - [OPTIONAL / BONUS] A2A multi-agent orchestration

**Goal**
Enable the primary agent to delegate sub-tasks to specialised child agents, each with their own tools,
prompts, and retrieval scope. Bonus credit; not required for sign-off.

**Requirement traceability**
Spec 3.7 (Optional A2A / Multi-Agent); Claims pattern: Orchestrator -> CoverageCheckAgent, FraudScreeningAgent,
SettlementAgent; deliverables `agents/{registry,decomposer,dispatcher,aggregator}.py`, `config/agents.yaml`,
`tests/test_multi_agent.py`.

**Files to create/modify**

- Create `app/agents/{registry,decomposer,dispatcher,aggregator}.py`.
- Populate `config/agents.yaml` with the three Claims child agents.
- Create `tests/test_multi_agent.py`.

**Tasks**

- [ ] `registry.py`: typed capability descriptors for each child agent.
- [ ] `decomposer.py`: split a complex request into sub-tasks.
- [ ] `dispatcher.py`: route sub-tasks to the right child agent (start with <=3 agents).
- [ ] `aggregator.py`: merge results + resolve conflicts; share context between agents.
- [ ] Gate behind `ENABLE_MULTI_AGENT`.

**How to run this phase independently**

- [ ] `PYTHONPATH=. pytest -q tests/test_multi_agent.py`
- [ ] Run a composite claim query and confirm delegation to Coverage/Fraud/Settlement agents.

**Acceptance criteria**

- [ ] Orchestrator decomposes and delegates to >=2 (target 3) child agents and aggregates a coherent result.
- [ ] Flag-gated; disabled by default; no impact on single-agent flows.

**No-regression guardrail**
`ENABLE_MULTI_AGENT` defaults OFF; the single-agent path is the production default.

**Cline task prompt**

> [Optional/bonus] Implement A2A multi-agent orchestration per spec. Build
> `app/agents/{registry,decomposer,dispatcher,aggregator}.py` driven by `config/agents.yaml` with three
> Claims child agents (CoverageCheckAgent, FraudScreeningAgent, SettlementAgent). Decompose complex queries,
> dispatch to <=3 agents, aggregate with conflict resolution, and share context. Gate behind
> `ENABLE_MULTI_AGENT` (default off). Validate with `pytest -q tests/test_multi_agent.py` and a composite
> claim query showing delegation. Do not affect single-agent flows.

---

## Phase W8-10 - [OPTIONAL / BONUS] Prompt & embedding drift detection

**Goal**
Detect silent degradation from prompt drift or embedding drift by comparing current outputs/embedding
distributions against a deployment baseline, with configurable alert thresholds and a `/eval/drift` endpoint.
Bonus credit; not required for sign-off.

**Requirement traceability**
Spec 3.8 (Optional Drift Detection); deliverables `drift/{baseline,prompt_drift,embedding_drift,detector,alerts}.py`,
`config/drift_thresholds.yaml`, `tests/test_drift_detection.py`; metrics: output semantic shift <0.85,
KL divergence >0.10, Spearman rank corr <0.80, NN stability <0.70.

**Files to create/modify**

- Create `app/drift/{baseline,prompt_drift,embedding_drift,detector,alerts}.py`.
- Populate `config/drift_thresholds.yaml`.
- Complete the `/eval/drift` endpoint (stubbed in W8-6).
- Create `tests/test_drift_detection.py`.

**Tasks**

- [ ] `baseline.py`: snapshot prompt outputs + embedding distributions at deployment.
- [ ] `prompt_drift.py`: cosine similarity of answer embeddings vs baseline; format-compliance check;
      citation-density change.
- [ ] `embedding_drift.py`: KL divergence of embedding vectors; Spearman rank correlation; NN stability
      (via `scipy`).
- [ ] `detector.py` + `alerts.py`: compare against `config/drift_thresholds.yaml`; raise alerts.
- [ ] `/eval/drift` returns drift scores; runnable as a scheduled cron job.

**How to run this phase independently**

- [ ] `PYTHONPATH=. pytest -q tests/test_drift_detection.py`
- [ ] `PYTHONPATH=. python -m app.drift.detector --baseline` then `--compare`.
- [ ] `curl -s -X POST localhost:8000/eval/drift | jq`.

**Acceptance criteria**

- [ ] Baseline captured; drift comparison produces scores against the configured thresholds.
- [ ] Alerts fire only when a threshold is breached (conservative defaults to avoid false alarms).
- [ ] `/eval/drift` returns valid drift scores and is cron-runnable.

**No-regression guardrail**
`ENABLE_DRIFT` defaults OFF; drift jobs are read-only and never alter the live retrieval path.

**Cline task prompt**

> [Optional/bonus] Implement prompt & embedding drift detection per spec. Build
> `app/drift/{baseline,prompt_drift,embedding_drift,detector,alerts}.py` with `config/drift_thresholds.yaml`.
> Prompt drift: answer-embedding cosine shift, format compliance, citation-density change. Embedding drift:
> KL divergence, Spearman rank correlation, NN stability (scipy). Complete `/eval/drift`. Use conservative
> thresholds to avoid false alarms; gate behind `ENABLE_DRIFT`. Validate with
> `pytest -q tests/test_drift_detection.py`, a baseline/compare CLI run, and an `/eval/drift` call. Drift
> jobs must be read-only.

---

## Phase W8-V - Extensive validation of all Week 8 changes

**Goal**
Run a comprehensive validation gate proving (a) zero regression across Weeks 2/4/6 + R1-R6, (b) every
mandatory Week 8 component works end-to-end, and (c) all Week 8 acceptance thresholds (Section 4) are met,
with >=50 LangSmith traces covering MCP calls, HITL gates, and role-filtered retrieval.

**Requirement traceability**
Spec Section 4 (Week 8 acceptance thresholds) + Section 10 (sign-off criteria).

**Files to create/modify**

- Create `scripts/validate_week8.py` (orchestrates the full gate).
- Create `docs/project_signoff_week8.md` (threshold -> evidence mapping).
- Create `tests/test_week8_endpoints_smoke.py` (every new endpoint).

**Tasks**

- [ ] Full regression: `PYTHONPATH=. pytest -q` (entire suite, all weeks).
- [ ] Regression suite: `PYTHONPATH=. python -m eval.regression_suite` -> pass rate >=95%.
- [ ] Endpoint smoke matrix: every Week 6 + Week 8 route (`/chat`, `/reset`, `/health`, `/ingest*`,
      `/retrieve`, `/evaluate`, `/sources*`, `/mcp/*`, `/hitl/*`, `/prompts*`, `/roles`, `/auth/context`,
      `/eval/regression`, `/eval/drift`).
- [ ] Threshold checks: MCP success >=95%; HITL precision >=85% / recall >=95%; prompt rollback <30s;
      role-filter leakage 0%; regression pass >=95%; answer stability >=0.90; all Week 6 thresholds held.
- [ ] Capture >=50 LangSmith traces (MCP, HITL, role-filtered retrieval) + screenshots.
- [ ] Write `docs/project_signoff_week8.md` mapping each threshold to concrete evidence.

**How to run this phase independently**

- [ ] `PYTHONPATH=. python scripts/validate_week8.py` (runs the whole gate and writes a summary).
- [ ] `PYTHONPATH=. pytest -q` and `PYTHONPATH=. python -m eval.regression_suite`.

**Acceptance criteria**

- [ ] Entire test suite green; zero regression in any prior-week test.
- [ ] Every Section-4 Week 8 threshold met or exceeded with recorded evidence.
- [ ] >=50 LangSmith traces captured across MCP / HITL / role-filtered retrieval.
- [ ] `docs/project_signoff_week8.md` completed with threshold->evidence traceability.

**No-regression guardrail**
This phase is the regression gate itself; it fails if any prior-week test breaks or any threshold misses.

**Cline task prompt**

> Run the full Week 8 validation gate. Create `scripts/validate_week8.py` that executes the entire
> `PYTHONPATH=. pytest -q` suite, `python -m eval.regression_suite`, and a smoke matrix hitting every Week 6
> + Week 8 endpoint. Verify all Section-4 thresholds (MCP >=95%, HITL precision >=85%/recall >=95%, rollback
> <30s, leakage 0%, regression >=95%, stability >=0.90) and confirm Week 6 thresholds still hold. Capture
> >=50 LangSmith traces covering MCP, HITL, and role-filtered retrieval. Produce `docs/project_signoff_week8.md`
> mapping each threshold to evidence. Fail the phase if any prior-week test regresses or any threshold misses.

---

## Phase W8-C - requirements.txt verification and deployment cleanup

**Goal**
Guarantee `requirements.txt` contains every package the project and all Week 8 phases need (pinned), remove
unused/duplicate entries, and strip diagnostics, fix-summaries, draft/plan files, and debug artefacts from
the deployment so the shipped artefact is clean and reproducible.

**Requirement traceability**
Original capstone instructions (verify requirements.txt completeness; remove diagnostics/plan files from
deployment) + spec Section 5.1 (pinned requirements.txt) + pitfall (secrets in images / stale artefacts).

**Files to create/modify**

- Update `requirements.txt` (pinned).
- Update `.gitignore` and `.dockerignore`.
- Create `scripts/verify_requirements.py` (import-vs-requirements audit).

**Tasks**

- [ ] Import-vs-requirements audit: parse all imports under `app/`, `eval/`, `frontend/`, `scripts/`,
      `tests/`; assert each third-party import has a pinned entry.
- [ ] Add missing pinned Week 8 deps: `mcp` (MCP SDK), `pyyaml`, `watchdog`, `semantic-version`, `scipy`,
      `python-jose[cryptography]` (JWT), `anthropic` (judge), plus confirm existing `rank-bm25`,
      `sentence-transformers`, `faiss-cpu`, `chromadb`, `fastapi`, `uvicorn[standard]`, `streamlit`,
      `langchain*`, `openai`, `langsmith`, `pydantic`, `python-dotenv`, `pytest`, `pytest-cov`,
      `pytest-asyncio`, `structlog`.
- [ ] Remove unused/duplicate packages only when provably unreferenced.
- [ ] Exclude from the deployment artefact (via `.dockerignore` / packaging): `diagnostics.md`,
      all `*_SUMMARY.md` (`INGESTION_FIX_SUMMARY.md`, `METADATA_FILTER_FIX_SUMMARY.md`,
      `RETRIEVAL_FALLBACK_FIX_SUMMARY.md`), `plan.md`, `extended_plan.md`,
      `scripts/diagnose_retrieval_path.py`, `notebooks/`, `Screenshots/`, and any `*.log`.
- [ ] Confirm `.env` is gitignored and never committed; `.env.example` retained as documentation only.
- [ ] `pip check` clean; fresh-venv install runs the app + full test suite.

**How to run this phase independently**

- [ ] `PYTHONPATH=. python scripts/verify_requirements.py` (fails on any missing/unpinned dep).
- [ ] `pip check`
- [ ] Fresh venv: `python -m venv /tmp/verifyenv && /tmp/verifyenv/bin/pip install -r requirements.txt`
      then `PYTHONPATH=. /tmp/verifyenv/bin/pytest -q`.

**Acceptance criteria**

- [ ] Every third-party import maps to a pinned `requirements.txt` entry; `pip check` is clean.
- [ ] Fresh-venv install runs the app and passes the test suite.
- [ ] Diagnostics, `*_SUMMARY.md`, and plan/draft files are absent from the built/deployed artefact.
- [ ] `.env` is not committed; `.env.example` remains as documentation.

**No-regression guardrail**
Cleanup only touches packaging/ignore rules and dependency pins; application source behaviour is unchanged
and re-verified by the fresh-venv test run.

**Cline task prompt**

> Finalise dependencies and clean deployment. Create `scripts/verify_requirements.py` that audits every
> import under `app/`, `eval/`, `frontend/`, `scripts/`, `tests/` against `requirements.txt` and fails on
> missing/unpinned deps. Add missing pinned Week 8 packages (mcp, pyyaml, watchdog, semantic-version, scipy,
> python-jose[cryptography], anthropic) and confirm all existing ones are pinned; remove only provably
> unused entries. Exclude from the deployment artefact via `.dockerignore`/packaging: `diagnostics.md`, all
> `*_SUMMARY.md`, `plan.md`, `extended_plan.md`, `scripts/diagnose_retrieval_path.py`, `notebooks/`,
> `Screenshots/`, and `*.log`. Confirm `.env` is gitignored. Validate with `scripts/verify_requirements.py`,
> `pip check`, and a fresh-venv install + `pytest -q`.

---

## Week 8 execution guidelines

- Work **in order**: W8-0 -> W8-1 -> ... -> W8-8, then optional W8-9/W8-10, then **W8-V** and **W8-C** last.
- For each phase: read the **Goal**, execute the **Tasks**, then use **Acceptance criteria** as the
  regression gate before moving on. Run the phase's **How to run** commands to validate in isolation.
- Keep all new code under the existing `app.*` package; configs in `config/`; prompts in `prompts/`.
- After every phase, re-run `PYTHONPATH=. pytest -q` to confirm no prior-week regression.
- Capture the pre-Week-8 baseline (`reports/eval_week8_baseline.md`) during W8-6 before other changes so
  improvement/no-regression can be proven at sign-off.
