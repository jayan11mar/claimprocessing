# Human-In-The-Loop (HITL) Workflows

## Purpose
This document describes the HITL (Human-In-The-Loop) workflow that pauses automated claim processing when configurable trigger rules match, creates persistent review tasks, and provides an approval UI for human reviewers.

## Trigger Rules

Trigger rules are defined in **`config/hitl_rules.yaml`** — 5 configurable rules:

| Rule ID | Field | Operator | Value | Reason |
|---------|-------|----------|-------|--------|
| `high_amount` | `claim_amount` | `>` | 6000 | High-value claim requires manual review |
| `claim_rejection` | `decision` | `equals` | `reject` | Claim rejection requires human confirmation |
| `partial_settlement` | `decision` | `equals` | `partial` | Partial settlement requires human approval |
| `fraud_flag` | `fraud_flag` | `is_true` | `true` | Fraud detection requires manual review |
| `policy_exclusion` | `policy_exclusion` | `is_true` | `true` | Policy exclusion requires human verification |

### Supported Operators
- Numeric: `>`, `>=`, `<`, `<=`
- Equality: `equals`
- Boolean: `is_true`, `is_false`

## Implementation Components

### Rule Evaluation Engine (`app/hitl/triggers.py`)
- `load_rules()` — Loads trigger rules from YAML config file (cached after first load)
- `clear_rules_cache()` — Clears cache (used in tests)
- `evaluate_triggers()` — Evaluates all rules against a context dict
- `_evaluate_rule()` — Evaluates a single rule with operator dispatch

### HITL Manager (`app/hitl/manager.py`)
- `HITLManager` class orchestrates pause/resume lifecycle
- `pause(context)` — Evaluates triggers; if matched, creates persistent task via store
- `resume(task_id, decision, comments)` — Records human decision
- `list_pending()` — Returns all pending tasks
- Singleton via `get_hitl_manager()` / `reset_hitl_manager_singleton()`

### Persistent Task Store (`app/hitl/store.py`)
- `HITLTaskStore` — SQLite-backed persistent storage
- **Database file:** `data/hitl_tasks.db` (20,480 bytes)
- Thread-safe with per-instance lock
- Full CRUD: `create_task()`, `get_task()`, `list_pending()`, `list_all()`, `update_decision()`
- Tasks serialised as JSON rows for round-trip fidelity
- Indices on `status` and `session_id`

### Data Models (`app/hitl/models.py`)
- `HITLTask` — Task data class with fields: task_id, session_id, rule_id, rule_reason, status, decision, reviewer_comments, retrieved_chunks, reasoning_trace, confidence, recommendation, timestamps
- `HITLTriggerResult` — Result with triggered boolean, matched_rules list, task reference

### LCEL Chain (`app/chains/hitl_chain.py`)
- `hitl_lcel_chain: Runnable` — RunnableLambda wrapping `_run_hitl()`
- When HITL is enabled: evaluates triggers, pauses chain if rule matched
- When HITL is disabled: pass-through (no-op)
- Output dict augmented with `hitl_paused`, `hitl_task_id`, `hitl_status`

## API Endpoints

All endpoints are in **`app/api/server.py`**:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/hitl/pending` | List all pending tasks |
| `POST` | `/hitl/review/{task_id}` | Approve or reject a task |
| `GET` | `/hitl/task/{task_id}` | Get a single task by ID |

### Review Request Format
```json
{
    "decision": "approved",
    "comments": "Looks correct, proceeding with settlement"
}
```
Decision values: `"approved"` or `"rejected"`.

## Approval UI Flow

The HITL Review tab is in **`app/frontend/streamlit_app.py`** (lines 508-660).

### User Flow
1. User navigates to "🛂 HITL Review" tab
2. Clicks "🔄 Refresh" to fetch pending tasks from `/hitl/pending`
3. Each pending task displays:
   - Task ID, session ID, creation timestamp
   - Trigger rule that matched and reason
   - User message and proposed agent response
   - Retrieved chunks with relevance scores
   - Reasoning trace
   - Confidence metric and proposed action
   - Recommendation details
4. Reviewer can:
   - Add optional comments
   - Click "✅ Approve" — POST to `/hitl/review/{task_id}` with `decision: "approved"`
   - Click "❌ Reject" — POST to `/hitl/review/{task_id}` with `decision: "rejected"`
5. UI refreshes automatically after decision

### Status Display
- HITL disabled: Warning banner "Set ENABLE_HITL=true to enable"
- No pending tasks: "✅ No pending HITL tasks"
- Tasks available: Success banner with task count

## Runtime Flow

```
Agent chain processes request
    │
    ▼
hitl_lcel_chain.invoke()
    │
    ├── [HITL Disabled] → Pass through (hitl.status = "skipped")
    │
    └── [HITL Enabled]
        │
        ├── Build context dict (claim_amount, decision, fraud_flag, etc.)
        │
        ├── evaluate_triggers(context)
        │   └── For each rule:
        │       ├── If match → add to matched list
        │       └── If no match → continue
        │
        ├── [No match] → Pass through (hitl.status = "passed")
        │
        └── [Match found]
            │
            ├── Create HITLTask with recommendation + context
            ├── store.create_task(task) — persist to SQLite
            └── Return pause signal: hitl_paused=True, hitl_task_id=...
    
    ── (Human reviews in Streamlit UI) ──

    │
    ▼
POST /hitl/review/{task_id} {"decision": "approved"}
    │
    ├── manager.resume(task_id, decision)
    │   └── store.update_decision(task_id, decision, comments)
    │
    └── Return updated task with decision
```

## Test Evidence

- **Test file:** `tests/test_hitl_workflow.py` — 32 test functions
- **Test files:**
  - `tests/verify_hitl_preconditions.py` — Precondition validation
  - `tests/verify_hitl_e2e.py` — End-to-end workflow test
- Coverage includes: trigger rule loading, rule evaluation, HITLTask creation, store CRUD, manager pause/resume, edge cases

## Reviewer Demo

```bash
# View configured trigger rules
python -c "
import yaml
with open('config/hitl_rules.yaml') as f:
    data = yaml.safe_load(f)
for rule in data['trigger_rules']:
    print(f\"  {rule['rule_id']}: {rule['field']} {rule['operator']} {rule.get('value', '')} — {rule['reason']}\")
"

# Run HITL tests
python -m pytest tests/test_hitl_workflow.py -v
```
