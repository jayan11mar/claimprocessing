# Week 8 Evaluation Final (post W8-6 fixes)

**Capture timestamp:** 2026-07-17 (Asia/Kolkata)

## Core metrics

| Metric | Value |
|--------|-------|
| total_cases | 200 |
| passed_cases | 200 |
| pass_rate | 1.0 |

## Custom metrics (post-fix)

| Metric | Baseline (pre-fix) | Final (post-fix) | Delta | Threshold | Status |
|--------|-------------------|-------------------|-------|-----------|--------|
| golden_set_pass_rate | 1.0 | 1.0 | 0.0 | >= 0.95 | ✅ PASS |
| answer_stability | 0.0 | **0.9891** | +0.9891 | >= 0.90 | ✅ PASS |
| regulatory_compliance | 0.0015 | **0.3333** | +0.3318 | >= 0.90 or null | ⚠️ Below threshold (only "exclusion" term in expected_answer) |
| role_appropriateness | 1.0 | 1.0 | 0.0 | == 1.0 | ✅ PASS |
| hitl_trigger_precision | 0.0 | **null** | N/A (fixed) | >= 0.85 or null | ✅ PASS |

**all_metrics_passed:** false (regulatory_compliance below threshold)

## Fixes applied

1. **answer_stability** — `eval/_run_stability.py`: moved `settings.OPENAI_MODEL_TEMPERATURE = 0.0` to execute **before** the `lcel_router` import, ensuring all models (including FAQChain) are built with deterministic temperature=0. Stability rose from 0.0 → 0.9891 (STABILITY_SAMPLE_SIZE=25).

2. **regulatory_compliance** — `eval/custom_metrics.py`: filtered to only regulatory-relevant cases (category in `{policy, regulatory, compliance, coverage}`). Added `"category": "regulatory"` to 8 cases in `eval/golden_set.json` (RAG-03, RAG-08, RAG-13, RAG-18, RAG-23, RAG-28, RAG-33, RAG-38). Returns `null` when no such cases exist, rather than scoring all 200 cases as 0.0015.

3. **hitl_trigger_precision** — Returns `null` when no HITL decisions are present, rather than 0.0.

## Test results

- `tests/test_rag_evaluation_harness.py`: **30 passed**, 0 failed (1 pre-existing test fixed with monkeypatch for `.env` isolation)

## Note on regulatory_compliance

The 8 tagged cases have expected_answer containing only "exclusion" (weight 0.5), giving a compliance score of 0.3333. To achieve >= 0.90, the expected_answer would need to contain more regulatory terms (e.g., IRDAI, grievance redressal, waiting period, portability, pre-existing disease, free-look period, sum insured, etc.) to reach a total weight of >= 1.35.