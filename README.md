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

### Recommended phase-update validation

After each project phase update, run the integration suite to verify core tool chaining flows:

```bash
PYTHONPATH=. pytest tests/test_policy_status_integration.py -q
```

## Evaluate end-to-end behavior

An evaluation harness is included at `scripts/evaluate.py`.
It sends 20 sample queries covering FAQ, tool usage, and multi-turn conversation behavior to the running `/chat` endpoint and saves the results to `scripts/results.json`.

```bash
python scripts/evaluate.py
```

## Golden dataset regression validation

The project includes a golden dataset seed structure in `data/golden_dataset/` and a validation harness at `scripts/validate_golden_dataset.py`.
The golden dataset categories are:

- `faq.json`: FAQ intent, category, confidence, and answer formatting validation
- `claims.json`: claim intake workflows and tool decision validation
- `fraud.json`: fraud score logic and signal validation
- `settlement.json`: settlement breakdown validation
- `memory.json`: multi-turn memory persistence and retrieval validation
- `guardrails.json`: PII, off-topic, and prompt injection guardrail validation

Run the regression validation script with:

```bash
python scripts/validate_golden_dataset.py
```

## Architecture

The system is built as a modular FastAPI + Streamlit application with LangChain orchestration, custom domain tools, SQLite memory, Pydantic validation, JSON logging, and optional LangSmith tracing.

## Documentation

- `docs/evaluation_report.md`: evaluation summary of the 20 queries.

## Database Schema

The `claims.db` SQLite database contains the following tables:

### policies
Stores insurance policy information:
- `policy_number` (TEXT, PRIMARY KEY)
- `policy_holder_id` (TEXT)
- `status` (TEXT, NOT NULL)
- `sum_insured` (REAL, NOT NULL)
- `deductible` (REAL, NOT NULL)
- `copay_percent` (REAL, NOT NULL)
- `sub_limits` (TEXT)
- `depreciation_schedule` (TEXT)
- `start_date` (TEXT)
- `end_date` (TEXT)
- `product_code` (TEXT)
- `coverage_type` (TEXT)
- `underwriting_class` (TEXT)
- `risk_category` (TEXT)
- `created_at` (TEXT, NOT NULL)
- `updated_at` (TEXT, NOT NULL)

### claims
Stores claim records with foreign key to policies:
- `claim_id` (TEXT, PRIMARY KEY)
- `policy_number` (TEXT, NOT NULL, FOREIGN KEY to policies)
- `policy_holder_id` (TEXT)
- `claim_amount` (REAL, NOT NULL)
- `incident_date` (TEXT)
- `admission_date` (TEXT)
- `discharge_date` (TEXT)
- `diagnosis_code` (TEXT)
- `hospital_name` (TEXT)
- `supporting_documents` (TEXT)
- `extra_info` (TEXT)
- `status` (TEXT)
- `loss_type` (TEXT)
- `reported_date` (TEXT)
- `closed_date` (TEXT)
- `approved_amount` (REAL)
- `fraud_score` (REAL)
- `settlement_status` (TEXT)
- `created_at` (TEXT, NOT NULL)
- `updated_at` (TEXT, NOT NULL)

### chat_history
Stores chat conversation data:
- `id` (INTEGER, PRIMARY KEY AUTOINCREMENT)
- `session_id` (TEXT, NOT NULL)
- `role` (TEXT, NOT NULL)
- `content` (TEXT, NOT NULL)
- `created_at` (TEXT, NOT NULL)

## Project layout

- `app/`: application modules
- `frontend/streamlit_app.py`: chat UI client
- `scripts/evaluate.py`: evaluation harness
- `docs/evaluation_report.md`: evaluation report
- `.env.example`: environment variable template
