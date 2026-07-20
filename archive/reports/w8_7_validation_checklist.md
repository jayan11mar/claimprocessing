# Week 8 — Task 7: Streamlit Evaluation Dashboard — Validation Checklist

## 1. Role Selector Validation

- [ ] **Role dropdown renders** — `st.selectbox` shows options: `claims_processor`, `senior_adjuster`, `claims_manager`, `fraud_investigator`
- [ ] **Default role** — defaults to `claims_processor` on first load
- [ ] **Get Token button** — sends `POST /auth/token` with `{"sub": "demo_user", "role": <selected_role>}`
- [ ] **Token stored in session state** — `st.session_state.jwt_token` is set on 200 response
- [ ] **Token status displayed** — success message shows `"✓ Token active (<role>)"` when token present
- [ ] **Clear button** — clears `jwt_token` and `auth_context` from session state
- [ ] **Anonymous state** — shows `"No token — anonymous access"` when no token
- [ ] **Role preserved across reruns** — selected role persists in `st.session_state.selected_role`

## 2. Auth Context Validation

- [ ] **Authenticated context** — when JWT token active, expander shows `GET /auth/context` response as JSON
- [ ] **Anonymous context** — when no token, expander shows `GET /auth/context` response or `"Backend unreachable"`
- [ ] **Error handling** — failed requests show `st.error` with status code or exception message

## 3. HITL Approval/Reject Validation

- [ ] **Refresh button** — sends `GET /hitl/pending`
- [ ] **Disabled backend message** — shows warning `"⚠️ HITL is disabled on the backend"` when `enabled: false`
- [ ] **No pending tasks** — shows `"✅ No pending HITL tasks."` when count is 0
- [ ] **Task list renders** — pending tasks displayed in bordered containers
- [ ] **Task metadata shown** — task_id, session_id, created_at, rule_id, rule_reason
- [ ] **User message expander** — shows user message content
- [ ] **Proposed agent response expander** — shows agent response content
- [ ] **Retrieved chunks** — expandable list with chunk text, source_id, score
- [ ] **Reasoning trace** — expandable text area
- [ ] **Confidence metric** — displayed as percentage or "N/A"
- [ ] **Proposed action metric** — shows action type from recommendation
- [ ] **Approve button** — sends `POST /hitl/review/{task_id}` with `{"decision": "approved", "comments": "..."}`
- [ ] **Reject button** — sends `POST /hitl/review/{task_id}` with `{"decision": "rejected", "comments": "..."}`
- [ ] **Success feedback** — shows success message and reruns on 200
- [ ] **Error feedback** — shows error with status code/text on failure

## 4. Prompt Rollback Validation

- [ ] **Prompt list fetched** — `GET /prompts` returns prompt names
- [ ] **Selector renders** — dropdown with sorted prompt names
- [ ] **Version history fetched** — `GET /prompts/{name}/history` returns version list
- [ ] **Active version indicated** — badge shows `"✅ ACTIVE"` on current version
- [ ] **Inactive versions** — show `"Inactive"` badge
- [ ] **Version metadata** — author, last_updated, changelog displayed
- [ ] **Model compatibility** — comma-separated model list shown
- [ ] **Input variables** — comma-separated variable list shown
- [ ] **Template content** — expandable text area
- [ ] **Sub-templates** — expandable per-key text areas
- [ ] **Activate button** — `POST /prompts/{name}/activate` with `{"version": "..."}`
- [ ] **Activate success** — shows success message and reruns
- [ ] **Activate failure** — shows error with status code/text

## 5. Evaluation Dashboard Validation

- [ ] **Fourth tab present** — labeled `"📊 Evaluation Dashboard"`
- [ ] **Fetch button** — labeled `"🔄 Fetch Evaluation"`, type primary
- [ ] **API call** — sends `POST /eval/regression` with `json={}`, timeout 120s
- [ ] **Success handling** — stores response in `st.session_state.eval_data`
- [ ] **Error handling** — stores error message in `st.session_state.eval_fetch_error`
- [ ] **Latest Metric Values section** — 4-column layout with:
  - [ ] Pass Rate (formatted as percentage)
  - [ ] Total Cases
  - [ ] Passed
  - [ ] Failed
- [ ] **prepare_trend_data called** — `prepare_trend_data([data])` invoked with API response
- [ ] **Trend charts rendered** — `st.line_chart` for each `available_metric`:
  - [ ] `pass_rate`
  - [ ] `golden_set_pass_rate`
  - [ ] `answer_stability`
  - [ ] `regulatory_compliance`
  - [ ] `role_appropriateness`
  - [ ] `hitl_trigger_precision`
- [ ] **Metric value card** — shown beside each chart with formatted value or "N/A"
- [ ] **Raw data expander** — shows DataFrame under `"📋 View Raw Data"`
- [ ] **No data state** — shows `"No evaluation history available"` when no data fetched
- [ ] **No metrics state** — shows `"No evaluation history available"` when metrics list is empty
- [ ] **Error state** — shows error message + `"Ensure the backend is running and try again."`

## 6. Citation Validation

- [ ] **Citation rendering** — `render_citations()` displays numbered source list
- [ ] **URL citations** — clickable markdown links for `http://`/`https://` source paths
- [ ] **File path citations** — formatted with backtick code style for local paths
- [ ] **Unknown source** — shows source_id with no path fallback
- [ ] **Chunk detail expanders** — per-citation expander with Chunk ID, Source ID, Relevance Score
- [ ] **Chunk text area** — shows chunk content in scrollable text area
- [ ] **Unique keys** — each text area has unique key via `message_index` + citation index + chunk_id
- [ ] **Live message citations** — `live_` prefix used for unsaved messages
- [ ] **History citations** — numeric index used for saved history messages
- [ ] **Empty citations** — function returns early if citations list is empty

## 7. FastAPI-Only Validation

- [ ] **No `app.rag.*` imports** — zero occurrences
- [ ] **No `qa_chain` references** — zero occurrences
- [ ] **No `retriever` references** — zero occurrences
- [ ] **No `vector db` references** — zero occurrences
- [ ] **No `regression_suite` import** — zero occurrences (uses `/eval/regression` endpoint)
- [ ] **No `LangChain` execution** — zero occurrences
- [ ] **All backend calls via `requests`** — uses `api_get`/`api_post`/`api_delete` helpers only
- [ ] **All endpoints are FastAPI routes** — `/chat`, `/health`, `/auth/*`, `/hitl/*`, `/prompts/*`, `/sources/*`, `/ingest`, `/eval/regression`

---

## Summary

| Section | Status | Notes |
|---------|--------|-------|
| 1. Role Selector | ⬜ | |
| 2. Auth Context | ⬜ | |
| 3. HITL Approval/Reject | ⬜ | |
| 4. Prompt Rollback | ⬜ | |
| 5. Evaluation Dashboard | ⬜ | |
| 6. Citations | ⬜ | |
| 7. FastAPI-Only | ⬜ | |