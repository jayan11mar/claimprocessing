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

This plan covers the FAQ chatbot and the LangChain-based app deliverables.

---

## Requirement → Phase mapping

| Requirement                                                                                       | Phase(s)                |
|---------------------------------------------------------------------------------------------------|-------------------------|
| Insurance FAQ Chatbot (prompt engg, few-shot, CoT, JSON, guardrails, simple UI)                  | 1, 2                    |
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
| LangSmith traces/screenshots for chain execution                                                 | 6, 7                    |

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

## Phase 1 – Baseline FAQ chatbot (Streamlit + direct LLM)

**Goal**  
Deliver an initial Insurance FAQ chatbot using direct OpenAI calls from Streamlit, focusing on prompt engineering, few-shot examples, and basic JSON-shaped responses (no LangChain yet).

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

## Phase 2 – Guardrails, Pydantic parsing, and prompt template library

**Goal**  
Upgrade the FAQ chatbot with Pydantic-validated outputs, a prompt template library, and input guardrails (PII, off-topic, injection) so earlier deliverables are fully met.

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

## Phase 3 – Introduce LangChain, SQLite memory, semantic example selection

**Goal**  
Refactor the FAQ chatbot into a LangChain-based chain with SQLite-backed conversational memory and semantic few-shot selection while preserving earlier behavior.

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
- [ ] Earlier features (guardrails, structured FAQResponse) still work as before.

---

## Phase 4 – FastAPI backend with /chat, /reset, /health and Streamlit as client

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

---

## Phase 5 – Domain tools: Claims Intake, Fraud Detector, Settlement Calculator

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

## Phase 6 – Observability, LangSmith, error handling, and tests

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

## Phase 7 – Evaluation, validation, and documentation

**Goal**  
Validate the system against the 20 specified test queries, and finalize documentation showing that the requirements are met.

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
  - [ ] Note on requirements and where each is satisfied.
- [ ] Capture or link LangSmith traces/screenshots showing chain execution and tool usage.

**Acceptance criteria**

- [ ] All 20 test queries run end-to-end through `/chat` and are documented in the evaluation report.
- [ ] README and docs clearly explain how this project satisfies the requirements.
- [ ] Repo is ready for review as a capstone project.

---

## Copilot execution guidelines

- Work **top-down**: complete Phase 0, then 1, …, 7 in order.  
- For each phase:
  - Read the phase’s **Goal**, then execute the **Tasks** using Copilot for code generation.
  - Use the **Acceptance criteria** as a regression checklist before moving forward.
- You may paste in or adapt prompt/chain patterns from classroom materials, but keep all code organized into the modules indicated above.
