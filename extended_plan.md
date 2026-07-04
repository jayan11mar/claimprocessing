# Gen AI Capstone Plan – Claims Processing & Settlement Automation Assistant

> **Usage with GitHub Copilot**  
> Implement this project **phase-by-phase**. For each phase:
> - Work only within that phase’s section until its acceptance criteria are met.  
> - Select the relevant checklist items and ask Copilot to generate or update code for those items only.  
> - Do not skip phases or merge their tasks; each phase is meant to be independently runnable.

---

## Project overview

This project implements an Insurance **Claims Processing & Settlement Automation Assistant** with an FAQ chatbot and agentic tools for claims intake, fraud signals, and settlement calculations.

Architecture stack:
- **LLM**: OpenAI-compatible ChatCompletion API.
- **Orchestration**: LangChain (chains, tools, memory).
- **Memory**: SQLite-based conversational memory (at least 10 turns).
- **Backend**: FastAPI with `/chat`, `/reset`, `/health`.
- **Frontend**: Streamlit chat UI calling FastAPI.
- **Validation & models**: Pydantic.
- **Observability**: Structured JSON logging and LangSmith traces.

This plan covers **both Week 2 and Week 4 deliverables**:
- Week 2: Insurance FAQ chatbot with prompt engineering, few-shot learning, CoT, structured JSON outputs, guardrails, and simple UI.
- Week 4: Full LangChain-based app with memory, tools, FastAPI backend, Streamlit frontend, logging, tests, and evaluation.

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

---

## High-level review of current phases (pre-update)

This review reflects the current implementation state and the feedback gap around multi-turn context handling.

- Phase 0–2: scaffolding, FAQ workflow, guardrails, and prompt library are in place and largely runnable.
- Phase 3: LangChain, SQLite memory, and the initial agent flow are present; the storage layer works, but the same-session context handoff should be hardened without altering already working logic.
- Phase 4: FastAPI chat/reset/health endpoints are implemented and remain the integration surface for the app.
- Phases 5–7: tool orchestration and evaluation are present, but they should remain intact while adding targeted validation around memory continuity.

Planned update: add a non-breaking phase focused on multi-turn context reliability and explicit validation so each phase remains independently runnable and reviewable.

---

## Phase execution note

Each phase below should remain independently runnable. Before accepting a phase, validate it with the relevant command(s) and confirm that existing working flows were not regressed.

---

## Global project structure (for all phases)

> **Copilot note:** Keep this structure from Phase 0 onwards. Each phase will populate or refine specific modules.

Proposed layout:

```text
.
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── faq_examples.py
│   │   ├── templates.json
│   │   └── loader.py
│   ├── chains/
│   │   ├── __init__.py
│   │   ├── simple_faq_llm.py
│   │   ├── base_chain.py
│   │   ├── faq_chain.py
│   │   └── agent_chain.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── guardrails.py
│   │   ├── claims_intake.py
│   │   ├── fraud_detector.py
│   │   └── settlement_calculator.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── faq.py
│   │   └── domain.py
│   ├── memory/
│   │   ├── __init__.py
│   │   └── sqlite_memory.py
│   ├── logging/
│   │   ├── __init__.py
│   │   └── json_logger.py
│   └── api/
│       ├── __init__.py
│       └── server.py
├── frontend/
│   └── streamlit_app.py
├── tests/
│   ├── __init__.py
│   ├── test_claims_intake.py
│   ├── test_fraud_detector.py
│   └── test_settlement_calculator.py
├── main.py
├── requirements.txt
├── README.md
└── .env.example
```

---

## Phase 0 – Repo scaffolding and environment

**Goal**  
Create a clean, reproducible project skeleton suitable for incremental Copilot-driven development.

**Tasks**

- [ ] Create the directory structure shown in **Global project structure**.
- [ ] Add `requirements.txt` with (minimum):
  - `fastapi`, `uvicorn[standard]`
  - `streamlit`
  - `langchain`
  - `openai`
  - `pydantic`
  - `python-dotenv`
  - `pytest`, `pytest-cov`
  - `langsmith` (or latest client)
- [ ] Implement `app/config.py`:
  - [ ] Use `BaseSettings` (Pydantic v1 or v2) for:
    - `OPENAI_API_KEY`
    - `OPENAI_MODEL_NAME`
    - `LOG_LEVEL`
    - `SQLITE_DB_PATH`
    - `ENVIRONMENT` (e.g., `dev` / `prod`)
  - [ ] Add a helper `get_settings()` singleton.
- [ ] Create `.env.example` with all required keys, documented.
- [ ] Implement basic JSON logging setup in `app/logging/json_logger.py`:
  - [ ] Function `get_logger(name: str)` that returns a logger emitting JSON lines (`timestamp`, `level`, `message`, plus extra dict).
- [ ] Create placeholder files for all modules (empty classes/functions) so imports will resolve in later phases.
- [ ] Add a minimal `frontend/streamlit_app.py` that renders a placeholder page (“Claims Assistant – WIP”).
- [ ] Add a minimal `README.md` describing setup steps (install, env, run placeholder Streamlit).

**Acceptance criteria**

- [ ] `pip install -r requirements.txt` succeeds.
- [ ] `streamlit run frontend/streamlit_app.py` opens a placeholder app without errors.
- [ ] `python -m app` imports resolve (no missing modules).

---

## Phase 1 – Week 2 baseline FAQ chatbot (Streamlit + direct LLM)

**Goal**  
Deliver an initial Insurance FAQ chatbot (Week 2 scope) using direct OpenAI calls from Streamlit, focusing on prompt engineering, few-shot examples, and basic JSON-shaped responses (no LangChain yet).

**Tasks**

- [ ] Implement simple FAQ examples in `app/prompts/faq_examples.py`:
  - [ ] Include at least the 3 provided examples:
    - New claim processing.
    - Suspicious claim / fraud signals.
    - Claim status update.
  - [ ] Add 2–3 more examples for policy coverage and documentation FAQs.
- [ ] Implement a simple prompt builder in `app/chains/simple_faq_llm.py`:
  - [ ] System prompt: role = Insurance FAQ assistant for claims, policy, and settlement questions.
  - [ ] Ingest few-shot examples from `faq_examples.py` into a text prompt.
  - [ ] Append user message.
- [ ] Implement an LLM call helper using OpenAI ChatCompletion:
  - [ ] Function `call_faq_llm(user_message: str) -> dict`.
  - [ ] Return structure: `{ "answer_text": str, "raw_json": dict | None }`.
  - [ ] Ask model to output an `answer` plus a simple JSON stub (`intent`, `category`, `confidence`).
- [ ] Build initial Streamlit FAQ UI in `frontend/streamlit_app.py`:
  - [ ] Text input box for user query.
  - [ ] “Send” button.
  - [ ] Display model’s `answer_text` and a pretty-printed JSON block.
  - [ ] Store chat history in `st.session_state` (no DB yet) to show a scrollable conversation.

**Acceptance criteria**

- [ ] Running `streamlit run frontend/streamlit_app.py` yields a working FAQ chatbot.
- [ ] At least 5 manual queries produce consistent text answers + JSON stubs.
- [ ] No LangChain, FastAPI, or SQLite in use yet.

---

## Phase 2 – Guardrails, Pydantic parsing, and prompt template library (Week 2 hardening)

**Goal**  
Upgrade the FAQ chatbot with Pydantic-validated outputs, a prompt template library, and input guardrails (PII, off-topic, injection) so Week 2 deliverables are fully met.

**Tasks**

- [ ] Define Pydantic models in `app/models/faq.py`:
  - [ ] `FAQIntent` enum: `CLAIM_REGISTRATION`, `POLICY_STATUS`, `FRAUD_CHECK`, `SETTLEMENT_QUERY`, `DOCUMENTS_REQUIRED`, `OTHER`.
  - [ ] `FAQResponse` model:
    - `intent: FAQIntent`
    - `category: str`
    - `confidence: float`
    - `answer_text: str`
    - `reasoning: Optional[str]`
    - `metadata: Dict[str, Any] = {}`
- [ ] Create prompt template library `app/prompts/templates.json`:
  - [ ] System templates for main assistant.
  - [ ] Guardrail templates (off-topic, unsafe content, PII warning).
  - [ ] A small loader `app/prompts/loader.py` with `load_templates()`.
- [ ] Implement guardrails in `app/tools/guardrails.py`:
  - [ ] `detect_pii(text) -> dict` – simple regex for emails, phone numbers, IDs.
  - [ ] `is_off_topic(text) -> dict` – keyword-based off-topic detection.
  - [ ] `detect_prompt_injection(text) -> dict` – look for “ignore previous instructions”, etc.
  - [ ] Return a structured result `{ "triggered": bool, "rule": str, "details": str }`.
- [ ] Integrate guardrails into `frontend/streamlit_app.py`:
  - [ ] Before calling `call_faq_llm`, run all guardrails.
  - [ ] If any guardrail triggers, return a safe canned response using `templates.json`.
- [ ] Replace ad-hoc JSON handling with Pydantic:
  - [ ] Ask the LLM to return a JSON block for `FAQResponse`.
  - [ ] Parse with `FAQResponse.parse_raw` or `parse_obj`.
  - [ ] On parse failure, re-prompt once with a stricter instruction.

**Acceptance criteria**

- [ ] All responses are validated instances of `FAQResponse` in the UI.
- [ ] PII/off-topic/injection attempts are blocked with safe guardrail responses.
- [ ] `templates.json` is the single source of truth for system and guardrail messages.

---

## Phase 3 – Introduce LangChain, SQLite memory, semantic example selection (Week 4 start)

**Goal**  
Refactor the FAQ chatbot into a LangChain-based chain with SQLite-backed conversational memory and semantic few-shot selection while preserving Week 2 behavior.

**Tasks**

- [ ] Set up base LangChain utilities in `app/chains/base_chain.py`:
  - [ ] Helper to create the OpenAI chat model.
  - [ ] Helper to create a `ChatPromptTemplate` from templates and examples.
- [ ] Implement SQLite-backed memory in `app/memory/sqlite_memory.py`:
  - [ ] Use `sqlite3` or LangChain’s `SQLiteChatMessageHistory`.
  - [ ] Provide functions:
    - `get_history(session_id: str) -> list[BaseMessage]`.
    - `append_message(session_id: str, role: str, content: str)`.
  - [ ] Ensure at least 10 turns are preserved per session.
- [ ] Implement semantic few-shot selection in `app/prompts/examples_store.py`:
  - [ ] Store all FAQ examples with `embedding` (computed via OpenAI embeddings).
  - [ ] Function `select_examples(query: str, k: int) -> list[Example]` using cosine similarity.
- [ ] Build LangChain FAQ chain in `app/chains/faq_chain.py`:
  - [ ] Combine guardrails (pre-filter) + semantic examples + `ChatPromptTemplate` + LLM.
  - [ ] Use `RunnableWithMessageHistory` (or custom) for memory integration.
- [ ] Update `frontend/streamlit_app.py` (temporarily still direct to Python, not API):
  - [ ] Replace `call_faq_llm` with calls to `faq_chain.invoke()`.
  - [ ] Pass a `session_id` (from `st.session_state`) to memory layer.

**Acceptance criteria**

- [ ] Multi-turn conversations (≥10 turns) retain context via SQLite across messages.
- [ ] Few-shot examples are selected based on semantic similarity.
- [ ] Week 2 features (guardrails, structured FAQResponse) still work as before.

---

## Phase 3A – Multi-turn memory reliability hardening (non-breaking)

**Goal**  
Improve multi-turn context reliability for same-session conversations without changing the currently working FAQ, tool, or API flows.

**Tasks**

- [ ] Keep the existing SQLite memory and agent/tool logic intact.
- [ ] Add a single, explicit path in the agent layer to read the active session history before tool and follow-up decisions.
- [ ] Ensure a follow-up turn in the same `session_id` can reuse previously supplied policy numbers, claim IDs, incident dates, and other extracted context.
- [ ] Add a regression test for at least 3 consecutive turns in one session.
- [ ] Add a simple validation command for memory continuity so the phase can be verified independently.

**Acceptance criteria**

- [ ] The current FAQ, policy status, claim intake, fraud, and settlement flows still pass their existing test suite.
- [ ] Running `PYTHONPATH=. pytest -q tests/test_claims_intake.py -k 'history or multiturn'` passes.
- [ ] Running `PYTHONPATH=. python scripts/validate_sqlite_context.py` passes.
- [ ] A new multi-turn regression test confirms that a later turn in the same `session_id` can reuse earlier context without asking the user to repeat it.
- [ ] No existing endpoint, frontend, or tool behavior is removed; only context handling and validation are strengthened.

---

## Phase 4 – FastAPI backend with /chat, /reset, /health and Streamlit as client (Week 4 core)

**Goal**  
Create a FastAPI backend exposing `/chat`, `/reset`, `/health`, and refactor Streamlit to call these endpoints. Add structured JSON logging and basic retry/fallback behavior.

**Tasks**

- [ ] Implement FastAPI app in `app/api/server.py` and `main.py`:
  - [ ] `/health` (GET): returns JSON with version, uptime, model name, DB status.
  - [ ] `/chat` (POST):
    - Request model: `session_id: str`, `message: str`, optional `metadata: dict`.
    - Response model: `answer_text: str`, `structured: FAQResponse`, `chain_metadata: dict`.
  - [ ] `/reset` (POST): clears conversation history for `session_id` in SQLite.
- [ ] Wire `faq_chain` into FastAPI handlers:
  - [ ] Use shared memory layer (`sqlite_memory.py`).
  - [ ] Add a simple retry: on exception or parse error, run a fallback chain with a safe, short answer.
- [ ] Add structured JSON logging via `json_logger.py`:
  - [ ] Log each `/chat` call with: `timestamp`, `session_id`, `user_message`, `intent`, `latency_ms`, `guardrail_triggered`, `error_info`.
- [ ] Refactor `frontend/streamlit_app.py` to use HTTP:
  - [ ] Maintain `session_id` in `st.session_state`.
  - [ ] Call FastAPI `/chat` via `requests` (or `httpx`).
  - [ ] Use `/reset` endpoint for “Reset conversation” button.
  - [ ] Optionally show `chain_metadata` in an expandable panel per message.

**Acceptance criteria**

- [ ] `uvicorn main:app --reload` starts the backend and `/health` returns correct JSON.
- [ ] Streamlit frontend no longer calls LangChain directly; all chat goes through FastAPI.
- [ ] Logs for `/chat` requests appear as JSON lines with required fields.
- [ ] Validation command: `PYTHONPATH=. pytest -q tests/test_api_chat.py tests/test_api_health_reset.py` passes.
- [ ] Existing working chat and reset behavior remains intact after the memory-hardening change.

---

## Phase 5 – Domain tools: Claims Intake, Fraud Detector, Settlement Calculator (Week 4 tools)

**Goal**  
Implement the three custom domain tools and integrate them as LangChain tools within the chat flow.

**Tasks**

- [ ] Define domain models in `app/models/domain.py`:
  - [ ] `Policy` with fields like `policy_number`, `sum_insured`, `deductible`, `copay_percent`, `sub_limits: Dict[str, float]`, `status`, etc.
  - [ ] `Claim` with fields like `claim_id`, `policy_number`, `claim_amount`, `diagnosis_code`, `hospital_name`, `admission_date`, `discharge_date`, etc.
  - [ ] `FraudSignal` and `FraudScoreResult` with fields such as `score`, `signals: List[str]`.
  - [ ] `SettlementBreakdown` with `gross_amount`, `deductible`, `copay_amount`, `approved_amount`, `notes: List[str]`.
- [ ] Create a small in-memory or SQLite-backed data store for demo policies/claims:
  - [ ] Utility functions to fetch policies/claims by ID.
- [ ] Tool 1 – Claims Intake & Validation Engine (`app/tools/claims_intake.py`):
  - [ ] Function `register_and_validate_claim(policy_number: str, claim_amount: float, extra_info: dict) -> ClaimValidationResult`.
  - [ ] Responsibilities:
    - Create a new claim ID.
    - Check basic policy status and constraints.
    - Produce flags for obvious coverage gaps/exclusions (generic rules only).
- [ ] Tool 2 – Fraud Signal Detector (`app/tools/fraud_detector.py`):
  - [ ] Function `compute_fraud_score(claim_id: str) -> FraudScoreResult`.
  - [ ] Use simple fraud indicators:
    - Time between policy start and claim.
    - Count of recent claims.
    - Duplicate claim amounts.
- [ ] Tool 3 – Settlement Calculator (`app/tools/settlement_calculator.py`):
  - [ ] Function `calculate_settlement(claim_id: str) -> SettlementBreakdown`.
  - [ ] Apply generic rules using policy fields: deductible, copay, simple depreciation factor, sub-limits.
- [ ] Integrate tools into LangChain as `StructuredTool` or `@tool` wrappers.
- [ ] Build an agent-style chain in `app/chains/agent_chain.py`:
  - [ ] Use `FAQResponse.intent` to decide when to call which tools.
  - [ ] Merge tool outputs into final `answer_text` and `FAQResponse.metadata`.
- [ ] Update `/chat` endpoint to use `agent_chain` instead of FAQ-only chain.

**Acceptance criteria**

- [ ] Queries like “Register a new claim…”, “What is the fraud score for claim X?”, “Calculate settlement for claim Y” invoke the appropriate tools.
- [ ] Tool inputs/outputs are validated with Pydantic.
- [ ] Tools are modular and decoupled from the API/front-end code.

---

## Phase 6 – Observability, LangSmith, error handling, and tests (Week 4 hardening)

**Goal**  
Add robust error handling, LangSmith tracing, latency tracking, and unit tests with ≥80% coverage for the custom tools.

**Tasks**

- [ ] Enhance error handling in FastAPI:
  - [ ] Global exception handler that wraps unhandled errors into a safe JSON response.
  - [ ] Attach a correlation ID to each `/chat` request and propagate in logs.
  - [ ] Ensure fallback chain is used on LLM/tool errors.
- [ ] Integrate LangSmith tracing:
  - [ ] Configure LangSmith project name and API key via `.env`.
  - [ ] Enable tracing for `agent_chain` and tool calls.
- [ ] Add latency metrics:
  - [ ] Measure total `/chat` latency, LLM call time, and tool execution time per request.
  - [ ] Include these in `chain_metadata` and logs.
  - [ ] Verify simple queries < 3 seconds and tool-augmented ones < 8 seconds under normal conditions.
- [ ] Implement unit tests with `pytest` in `tests/`:
  - [ ] `test_claims_intake.py`: normal and edge cases (inactive policy, large claim).
  - [ ] `test_fraud_detector.py`: scenarios with 0, medium, and high fraud scores.
  - [ ] `test_settlement_calculator.py`: varying deductibles and copays.
  - [ ] Use `pytest-cov` to ensure ≥80% coverage for each `app/tools/*.py` module.

**Acceptance criteria**

- [ ] FastAPI never returns raw stack traces to clients; errors are logged and user gets a helpful fallback message.
- [ ] LangSmith (or equivalent) shows traces of chains with tool calls and memory.
- [ ] `pytest --cov=app/tools` reports ≥80% coverage for tools.

---

## Phase 7 – Evaluation, Week 2 & Week 4 validation, and documentation

**Goal**  
Validate the system against the 20 specified test queries, and finalize documentation showing that both Week 2 and Week 4 requirements are met.

**Tasks**

- [ ] Implement a simple evaluation harness (CLI script or notebook):
  - [ ] Call `/chat` for each of the 20 queries (FAQ, tools, multi-turn).
  - [ ] Save responses (answer, `FAQResponse`, `chain_metadata`) to a JSON or CSV file.
- [ ] Manually verify behavior for each category:
  - [ ] FAQ-only questions.
  - [ ] Tool usage questions.
  - [ ] Multi-turn scenarios leveraging memory.
- [ ] Create `docs/evaluation_report.md`:
  - [ ] For each query: input, expected behavior (1–2 lines), actual behavior (summary), and notes on accuracy, formatting, and guardrail effectiveness.
  - [ ] Explicitly mark which queries exercised which tools.
- [ ] Update `README.md` with:
  - [ ] High-level architecture description and simple diagram (optional ASCII).
  - [ ] Setup and run instructions (backend, frontend, env variables, DB initialization).
  - [ ] How to run tests and evaluation harness.
  - [ ] Note on Week 2 vs Week 4 requirements and where each is satisfied.
- [ ] Capture or link LangSmith traces/screenshots showing chain execution and tool usage.

**Acceptance criteria**

- [ ] All 20 test queries run end-to-end through `/chat` and are documented in the evaluation report.
- [ ] README and docs clearly explain how this project satisfies both Week 2 and Week 4 requirements.
- [ ] Repo is ready for review as a capstone project.

## Phase 7A – Golden dataset and regression validation for future RAG extension

**Goal**  
Prepare the project for a future RAG extension by creating a phase-specific golden dataset and regression validation process for each phase and tool.

**Tasks**

- [ ] Define a golden dataset format that captures: user query, expected tool/intent, expected structured response, relevant context history, and evaluation labels.
- [ ] Create a `data/golden_dataset/` folder pattern in the repo and add at least one seed golden dataset file for:
  - FAQ responses.
  - Claim intake workflow.
  - Fraud detection workflow.
  - Settlement calculation workflow.
  - Multi-turn memory/regression scenarios.
- [ ] Implement a validation harness or script (e.g. `scripts/validate_golden_dataset.py`) that:
  - [ ] Loads each golden dataset item.
  - [ ] Replays it against the current system phase/tool chain.
  - [ ] Compares actual outputs to expected structured outputs.
  - [ ] Produces a regression summary with pass/fail status per item.
- [ ] Add phase/tool-specific validation checks:
  - [ ] FAQ phase: verify intent classification, JSON schema, and answer correctness against golden expected labels.
  - [ ] Tool phase: verify correct tool selection and expected tool output structure.
  - [ ] Memory phase: verify multi-turn context reuse and stateful references across turns.
- [ ] Document the golden dataset schema and validation command in `README.md` or `docs/evaluation_report.md`.
- [ ] Add or update tests to ensure the golden dataset validation harness works as a regression gate.

**Acceptance criteria**

- [ ] A golden dataset schema is documented and stored in the repo.
- [ ] A regression validation script can execute the dataset and report pass/fail results.
- [ ] The phase/tool regression dataset covers FAQ, claim intake, fraud, settlement, and multi-turn memory.
- [ ] Future RAG extension decisions can leverage the dataset and regression results as a clear baseline.

## Phase 8 – SQLite schema for policy and claim domain data

**Goal**  
Create a persistent SQL schema for demo policy and claim records while keeping the current in-memory domain store intact.

**Tasks**

- [ ] Extend `app/memory/sqlite_memory.py` to create `policies` and `claims` tables alongside `chat_history`.
- [ ] Define policy-specific columns for coverage, sub-limits, dates, and underwriting metadata.
- [ ] Define claim-specific columns for claim details, optional enrichment, and fraud/settlement state.
- [ ] Keep the existing `chat_history` conversation memory unchanged.
- [ ] Optionally add a schema validation helper or startup check that confirms all three tables exist.

**Acceptance criteria**

- [ ] `SQLiteMemory` initializes the `policies` and `claims` tables on startup.
- [ ] The schema includes optional enrichments such as `status`, `approved_amount`, `fraud_score`, and `settlement_status`.
- [ ] The SQLite database file contains `chat_history`, `policies`, and `claims` after app initialization.

## Phase 9 – Migrate domain data to SQLite and enforce fraud-gated claims

**Goal**  
Move policy and claim storage from in-memory dictionaries to SQL tables, and require fraud validation before claim persistence.

**Tasks**

- [ ] Replace in-memory domain access functions in `app/models/domain.py` with SQL-backed implementations using `policies` and `claims`.
- [ ] Use the SQL tables for `get_policy`, `get_claim`, `save_claim`, `get_claims_for_policy`, and `get_claims_for_policy_holder`.
- [ ] Update `app/tools/claims_intake.py` so claim registration computes fraud score before saving a claim.
- [ ] Persist `claim.status`, `claim.fraud_score`, and optional settlement status on saved claim records.
- [ ] Add a demo data seeding helper that populates SQL tables from the current `_DEMO_POLICIES` and `_DEMO_CLAIMS` seed data.

**Acceptance criteria**

- [ ] Claim registration only persists claim records when fraud score is within acceptable bounds.
- [ ] Fraud detection can operate both against saved claims and against a pre-save claim evaluation.
- [ ] Existing tool and API tests continue to pass with SQL-backed policy/claim persistence.

---

## Copilot execution guidelines

- Work **top-down**: complete Phase 0, then 1, …, 7 in order.  
- For each phase:
  - Read the phase’s **Goal**, then execute the **Tasks** using Copilot for code generation.
  - Use the **Acceptance criteria** as a regression checklist before moving forward.
- You may paste in or adapt prompt/chain patterns from classroom materials, but keep all code organized into the modules indicated above.


---

## Week 6 extension – RAG layer for Claims Processing

> These phases extend the existing Week 2/Week 4 claims assistant with a production-grade RAG layer,
> aligning with the Week 6 requirements document and the trainer's banking RAG reference project.
> Implement them only after Week 4 phases are stable. Each phase should be independently runnable.

### Week 6 overall goals

- Add a document-backed knowledge layer over policy wordings, IRDAI regulations, exclusion lists,
  hospital network agreements, and prior adjudication memos.
- Expose a RAG-backed knowledge_retrieval tool to the existing LangChain agent so it can decide
  per query whether to call a deterministic tool, the RAG tool, or both.
- Implement hybrid retrieval (BM25 + dense vectors), re-ranking, answer + citations, and an
  evaluation harness measuring retrieval and answer quality against a golden set.

The repository layout for new RAG modules should follow this pattern (adapted from the trainer's
banking assistant):

```text
project-root/
├── app/                  # existing FastAPI + chains + tools + prompts (Week 4)
│   ├── rag/              # new: RAG pipeline
│   │   ├── loaders.py    # multi-format document loaders
│   │   ├── chunkers.py   # recursive + semantic chunking strategies
│   │   ├── embeddings.py # embedding adapters (OpenAI + sentence-transformers)
│   │   ├── vectorstores/ # FAISS, Chroma, Pinecone store implementations
│   │   ├── retriever_basic.py  # dense-only retriever
│   │   ├── retriever_bm25.py   # BM25 retriever
│   │   ├── retriever_hybrid.py # BM25 + dense hybrid retriever with re-ranking
│   │   ├── query_transform.py  # multi-query expansion / HyDE
│   │   └── qa_chain.py         # answer + citations QA chain
│   └── ... existing modules ...
├── eval/                 # new/extended: RAG eval harness
│   ├── golden_set.json   # Week 6 golden set (50 queries: 20 Week 4 + 30 RAG/hybrid)
│   ├── intrinsic.py      # retrieval metrics (Hit@K, MRR, NDCG, context precision/recall)
│   ├── extrinsic.py      # answer faithfulness/answer correctness metrics
│   ├── llm_judge.py      # LLM-as-judge scoring
│   └── run_eval.py       # CLI entrypoint
├── docs/
│   ├── vector_backend_choice.md  # justification for FAISS/Chroma/Pinecone
│   ├── eval_methodology.md       # metrics + thresholds
│   ├── eval_baseline.md          # baseline scores
│   └── eval_final.md             # final scores after tuning
└── ... existing files ...
```

---

## Phase 10 – RAG foundations: manifest‑driven loaders, chunking, embeddings, basic retriever

**Goal**  
Stand up an end‑to‑end retrieval pipeline over the claims knowledge base so the agent has a
knowledge layer to draw on. This phase focuses on a manifest‑driven loader, chunking strategies,
embeddings, and a basic dense retriever.

**Tasks**

- [ ] Create `data/knowledge_base/manifest.yaml` (if not already present) listing all KB sources:
  - Health policy wordings:  
    - `policies/health_hdfcergo_wording.pdf`  
    - `policies/health_kotakmahindra_wording.pdf`  
    - `policies/health_sbihealth_wording.pdf`
  - Motor policy wordings:  
    - `policies/motor_sbi_private_wording.pdf`  
    - `policies/motor_sbi_wording.pdf`
  - Regulations:  
    - `regulations/irDAI_health_regulations_2016.pdf`
  - Network agreements:  
    - `network/hospital_network_agreement_bopartitemodel.docx`  
    - `network/hospital_network_agreement.pdf`
  - Exclusions:  
    - `exclusions/health_exclusions_summary.pdf`
  - Adjudication memos:  
    - `adjudication_memos/Memo 1.md`  
    - `adjudication_memos/memo.json` (collection of synthetic memos)

  Each entry should include: `id`, `path`, `doc_type`, `insurance_type`, `product_code`,
  `product_name` (where applicable), `claim_type` (for memos), and `jurisdiction`.

- [ ] Create `app/rag/loaders.py`:

  - [ ] Implement `load_manifest()` that reads `manifest.yaml` from `KNOWLEDGE_BASE_DIR`
        (e.g., `data/knowledge_base/manifest.yaml`) using `pyyaml` and returns the parsed dict.
  - [ ] Implement `iter_manifest_sources()` that yields a normalized structure for each source:
        `id`, full `path`, `doc_type`, `insurance_type`, `product_code`, `product_name`,
        `claim_type`, `jurisdiction`, and any extra `metadata`.
  - [ ] Implement format‑specific loaders for PDF, DOCX, Markdown, and JSON using tools
        specified in `requirements.txt` (`pypdf`, `python-docx`, `docx2txt`, `beautifulsoup4`,
        `lxml`, etc.).  
        - PDF → text via `pypdf`.  
        - DOCX → text via `python-docx` or `docx2txt`.  
        - Markdown → text via simple file read.  
        - JSON memos → flatten relevant fields (`facts`, `decision`, `cited_*`) into text blocks.
  - [ ] Provide a function `load_documents_from_manifest() -> list[Document]` where `Document`
        includes `text`, `source_id`, `source_path`, `doc_type`, `insurance_type`, `product_code`,
        `claim_type`, and raw metadata.

- [ ] Create `app/rag/chunkers.py`:

  - [ ] Implement `recursive_chunk(text, config)` using `RecursiveCharacterTextSplitter` with
        Week‑6 defaults (size ~800, overlap ~100).
  - [ ] Implement a semantic chunker for policy/regulation/network docs that respects headings,
        clause numbering, and section breaks.
  - [ ] Ensure chunk metadata preserves `doc_type`, `insurance_type`, `product_code`,
        `claim_type`, and `section/clause_id` where available.

- [ ] Create `app/rag/embeddings.py`:

  - [ ] Wrap OpenAI embeddings (e.g., `text-embedding-3-small`) and sentence‑transformer models
        from `sentence-transformers`.
  - [ ] Provide an adapter `get_embedding_fn(model_name: str)` that returns a callable used by
        vector stores, ensuring the same model is used for ingestion and querying (version‑pinned
        via config).

- [ ] Create `app/rag/vectorstores/base.py`:

  - [ ] Define an abstract `VectorStore` interface (`add`, `search`, `delete`, `persist`,
        `as_retriever`).
  - [ ] Implement concrete vector stores:
        - `faiss_store.py` using `faiss-cpu`.
        - `chroma_store.py` using `chromadb`.
        - `pinecone_store.py` (optional, if using managed Pinecone).
  - [ ] Implement `get_vector_store(backend: str)` in `app/rag/vectorstores/__init__.py` that
        returns the appropriate store instance based on `VECTOR_BACKEND` (`faiss`, `chroma`,
        `pinecone`).

- [ ] Create `app/rag/retriever_basic.py`:

  - [ ] Implement `build_basic_retriever()` that:
        - Calls `load_documents_from_manifest()` to read all sources defined in `manifest.yaml`.
        - Applies recursive/semantic chunking to each document type.
        - Embeds chunks using the chosen embedding model from `get_embedding_fn`.
        - Upserts chunks into the configured vector store with metadata fields:
          `doc_type`, `insurance_type`, `product_code`, `claim_type`, `section`, `clause_id`.
        - Returns a LangChain `VectorStoreRetriever` (or equivalent `Runnable`) over this store.

- [ ] Add a CLI script `python -m app.rag.ingest_basic`:

  - [ ] Entry point that:
        - Loads manifest sources.  
        - Runs loaders → chunkers → embeddings → vector store upsert.  
        - Prints a summary per `doc_type` (`policy_wording`, `regulation`, `network`,
          `exclusion_summary`, `memo`, etc.) including document counts and total chunk counts.

**How to run this phase independently**

- [ ] Ensure `.env` includes:
  - `KNOWLEDGE_BASE_DIR` (root for claims KB documents, e.g., `data/KNOWLEDGE_BASE`).  
  - `VECTOR_BACKEND` (`faiss`, `chroma`, or `pinecone`).  
  - `EMBEDDING_MODEL` (e.g., `text-embedding-3-small` or the sentence‑transformer you choose).
- [ ] Run: `python -m app.rag.ingest_basic`.
- [ ] Verify that the chosen vector store directory/collection contains the expected number of
      chunks and that per‑type counts match the manifest (3 health policies, 2 motor policies,
      1 regulations PDF, 2 network agreements, 1 exclusions file, N adjudication memos).

**Acceptance criteria**

- [ ] Multi‑format document ingestion (PDF, DOCX, Markdown, JSON) succeeds for all sources listed
      in `manifest.yaml`.
- [ ] Corpus is chunked with both recursive and semantic strategies (configurable via code or
      `.env`), and chunk metadata includes `doc_type`, `insurance_type`, `product_code`,
      `claim_type`.
- [ ] Embeddings are stored in the selected vector backend and can be retrieved via a basic
      retriever built from `build_basic_retriever()`.
- [ ] RAG retrieval can filter by `doc_type` and `product_code` using metadata, aligning with
      Week‑6 requirements for clause/exclusion/regulatory lookups.
---

## Phase 11 – Vector backends and metadata schema selection

**Goal**  
Abstract the storage layer so you can choose FAISS, Chroma, or Pinecone per environment, and define
metadata schemas suitable for claims processing (policy wordings, regulations, network agreements,
prior memos).

**Tasks**

- [ ] Implement a vector store factory in `app/rag/vectorstores/__init__.py`:
  - [ ] Function `get_vector_store(backend: str)` that returns an instance of FAISS/Chroma/Pinecone
        store based on `VECTOR_BACKEND`.
- [ ] Add a benchmark script `app/rag/benchmarks/vector_backend_bench.py`:
  - [ ] Measure ingestion time, top-5 retrieval latency (p50/p95), recall@5, and storage footprint
        for each backend on a sample corpus.
- [ ] Define per-chunk metadata schema in `app/rag/metadata.py`:
  - [ ] Fields: `doc_type` (policy_wording | regulation | network | memo), `insurance_type`,
        `insurer`, `product_code`, `claim_type`, `section`, `clause_id`.
  - [ ] Ensure metadata is attached to every chunk on upsert.
- [ ] Create `docs/vector_backend_choice.md`:
  - [ ] Summarize benchmark results and justify chosen backend for your local dev environment
        (e.g., FAISS for local experiments, Chroma for simple persistence).

**How to run this phase independently**

- [ ] Run: `python -m app.rag.benchmarks.vector_backend_bench`.
- [ ] Inspect the generated metrics and choose a default backend via `.env`.

**Acceptance criteria**

- [ ] All three backends (FAISS, Chroma, Pinecone if configured) are pluggable without code changes
      beyond config.
- [ ] Each chunk in the vector store has the required metadata fields for filtering.
- [ ] `docs/vector_backend_choice.md` is committed and explains the selection.

---

## Phase 12 – Advanced RAG: hybrid retrieval and QA chain

**Goal**  
Raise retrieval quality by combining sparse (BM25) and dense signals, applying query transforms, and
building an answer + citations QA chain that the claims agent can call.

**Tasks**

- [ ] Implement BM25 retriever in `app/rag/retriever_bm25.py` using `rank-bm25` over the same
      chunks.
- [ ] Implement hybrid retriever in `app/rag/retriever_hybrid.py`:
  - [ ] Fuse BM25 and dense scores via weighted sum or RRF (reciprocal rank fusion).
  - [ ] Support HyDE / multi-query expansion in `app/rag/query_transform.py`.
- [ ] Implement QA chain in `app/rag/qa_chain.py`:
  - [ ] Input: user query + optional claim context (policy number, claim id).
  - [ ] Steps: hybrid retrieval → re-ranking via cross-encoder (e.g., `Cohere rerank` or
        `sentence-transformers` cross-encoder) → answer generation → citations.
  - [ ] Output: JSON with `answer_text`, `citations: list[ChunkCitation]`, `confidence`.
  - [ ] Enforce citation rule: every factual claim in the answer must reference a `chunk_id`; the
        response JSON includes the exact chunk text for each citation.
- [ ] Create `python -m app.rag.qa_demo` for local testing:
  - [ ] Accept a query from stdin.
  - [ ] Print answer + citations and show top-5 retrieved chunks.

**How to run this phase independently**

- [ ] After Phase 10–11 ingestion, run: `python -m app.rag.qa_demo`.
- [ ] Test queries drawn from Week 6 per-project RAG test table (coverage lookup, exclusion,
      regulatory reference, sub-limit, past cases, network, comparative, hard/ambiguous, refusal,
      cite-required rejection letter).

**Acceptance criteria**

- [ ] Hybrid retrieval improves top-5 recall and answer quality versus dense-only retriever.
- [ ] QA chain returns answers with at least one citation per factual claim.
- [ ] Critical query patterns (policy clause interpretation, exclusion lookup, partial-rejection
      decisions, regulatory limits) perform well.

---

## Phase 13 – RAG evaluation harness and acceptance thresholds

**Goal**  
Make the RAG pipeline measurable and align it with Week 6 acceptance thresholds for claims.

**Tasks**

- [ ] Build a 50-item golden set in `eval/golden_set.json`:
  - [ ] Include the 20 Week 4 queries plus 30 RAG/hybrid queries (10 per-project RAG queries
        enumerated in the Week 6 document, plus 20 hybrid queries combining tools + RAG).
- [ ] Implement `eval/intrinsic.py`:
  - [ ] Metrics: Hit@K, MRR, NDCG, context precision/recall.
- [ ] Implement `eval/extrinsic.py`:
  - [ ] Metrics: faithfulness (groundedness), answer correctness, answer relevance.
- [ ] Implement `eval/llm_judge.py`:
  - [ ] Read separate judge configuration from environment (e.g., `JUDGE_MODEL_NAME`,
        `JUDGE_OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`).
  - [ ] Construct a judge LLM instance that is distinct from the main generation model
        (e.g., app uses `OPENAI_MODEL_NAME=gpt-4o-mini`, judge uses `JUDGE_MODEL_NAME=gpt-4o`
        or an Anthropic Claude model).
  - [ ] LLM-as-judge scoring for correctness, completeness, citation quality, and clarity
        (1–5 scale).
  - [ ] Ensure eval code never falls back to the main app model if judge configuration is
        present; using the same model is allowed only as a temporary fallback.
- [ ] Implement `eval/run_eval.py` CLI:
  - [ ] For each golden set item, run RAG QA chain.
  - [ ] Compute metrics and write JSON results to `eval/results/eval_<timestamp>.json`.
- [ ] Create `docs/eval_methodology.md`, `docs/eval_baseline.md`, and `docs/eval_final.md` with
      baseline and tuned metrics.

**How to run this phase independently**

- [ ] Run: `python -m eval.run_eval`.
- [ ] Inspect metrics and compare against Week 6 acceptance thresholds for Claims (Hit@5, MRR,
      faithfulness, answer correctness, LLM-judge average, citation coverage).

**Acceptance criteria**

- [ ] Metrics meet or approach the thresholds listed for "Claims / Insurance" in the Week 6
      document.
- [ ] Citation coverage is 100% for factual claims in the golden set.
- [ ] Evaluation results and methodology documents are committed.

---

## Phase 14 – Agent integration, new endpoints, and tool routing

**Goal**  
Expose the RAG retrieval tool to the existing claims agent, add Week 6 endpoints, and ensure the
agent can route between deterministic tools and RAG based on intent.

**Tasks**

- [ ] Extend `app/tools.py` (or claims-specific tools module) with a `knowledge_retrieval` tool
      that calls `qa_chain`.
- [ ] Update the agent in `app/chain.py` (or `app/chains/agent_chain.py`):
  - [ ] Include deterministic tools (claims intake, fraud, settlement) + `knowledge_retrieval`.
  - [ ] Use classifier/intent router to decide when to call which tool(s).
  - [ ] Allow hybrid responses (tool output + RAG-backed explanation/citations).
- [ ] Extend FastAPI endpoints in `app/main.py` or `app/api/server.py`:
  - [ ] `/chat` (POST): now returns retrieval trace + citations when RAG is invoked.
  - [ ] `/ingest` (POST): upload docs, run loaders → chunkers → embeddings → upsert; return job ID.
  - [ ] `/ingest/status/{job_id}` (GET): poll ingestion job state.
  - [ ] `/retrieve` (POST): pure retrieval (no LLM), top-k chunks with scores.
  - [ ] `/evaluate` (POST): run evaluation suite against stored golden set.
  - [ ] `/sources` (GET): list indexed documents with metadata and chunk count.
  - [ ] `/sources/{doc_id}` (DELETE): remove a document and its chunks.
- [ ] Update Streamlit frontend (or introduce a new panel) to:
  - [ ] Show clickable citations that surface the exact chunk.
  - [ ] Provide basic document management (list sources, delete source, run RAG-only queries).

**How to run this phase independently**

- [ ] Run backend: `uvicorn app.main:app --reload`.
- [ ] Test new endpoints via Swagger UI (`/docs`) and via HTTPie/curl.
- [ ] Confirm `/chat` uses RAG tool when required and includes citations.

**Acceptance criteria**

- [ ] All Week 4 endpoints continue to work; new Week 6 endpoints are added without regression.
- [ ] Agent can route between tools and RAG based on intent/classification.
- [ ] RAG-backed answers in `/chat` include traceable citations (doc name/page/section).

---

## Phase 15 – Week 6 acceptance criteria validation and sign-off

**Goal**  
Systematically validate all Week 6 requirements and acceptance thresholds for the Claims Processing
& Settlement Automation assistant.

**Tasks**

- [ ] Create `docs/week6_acceptance_mapping.md` that lists each Week 6 criterion and maps it to:
  - [ ] Implemented module(s).
  - [ ] Test(s) or evaluation metric(s).
  - [ ] Evidence (LangSmith trace ID, screenshot, eval JSON path).
- [ ] Add tests in `tests/test_rag_pipeline.py` and `tests/test_api_rag_endpoints.py`:
  - [ ] Cover ingestion endpoints, retrieval-only endpoint, and RAG-backed `/chat` behavior.
- [ ] Run `python -m eval.run_eval` and capture metrics vs thresholds.
- [ ] Capture LangSmith traces for at least 30 sample RAG + hybrid queries showing run trees.
- [ ] Generate a sign-off document `docs/week6_signoff_report.md` summarizing:
  - [ ] Which thresholds are met.
  - [ ] Any gaps and mitigation notes.

**How to run this phase independently**

- [ ] Run: `PYTHONPATH=. pytest -q tests/test_rag_pipeline.py tests/test_api_rag_endpoints.py`.
- [ ] Run: `python -m eval.run_eval`.
- [ ] Review `docs/week6_signoff_report.md` prior to submission.

**Acceptance criteria**

- [ ] Every Week 6 requirement and acceptance threshold from the requirements PDF has a clear,
      implemented check and evidence.
- [ ] Test suite and evaluation harness pass without critical failures.
- [ ] Week 4 functionality remains intact (no regressions), and Week 6 RAG layer is demonstrably
      production-grade for the training context.
