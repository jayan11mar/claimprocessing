# Week 4 Evaluation Baseline

**Report purpose:** Baseline evaluation of the RAG pipeline at the Week 4 milestone, capturing the state before planned remediation and final tuning.

**Evaluation date / source artifact:** 2026-07-22 (derived from earliest evaluation artifacts). Source artifacts: `reports/eval_baseline.md`, `reports/report.md`, `reports/summary.json`, `reports/regression_report.json`.

**Scope:** Baseline assessment of the claims processing RAG system covering:
- Intrinsic retrieval metrics (Hit Rate @ K, MRR, NDCG, Context Precision, Context Recall)
- Extrinsic answer quality metrics (Faithfulness, Answer Correctness)
- LLM-as-Judge scoring
- Custom metrics (golden set pass rate, answer stability, regulatory compliance, role appropriateness, HITL trigger precision)
- Failure analysis bucketing

**Dataset / golden set reference:** Initial golden set defined in `eval/eval_set.json` and early version of `eval/golden_set.json`. The baseline report from `reports/report.md` describes 10 evaluation cases (1 passed, 9 failed) with synthetic failure scenarios covering retrieval gaps, answer quality issues, and citation errors. The full regression run documented in `reports/regression_report.json` later expanded to 200 cases across 4 projects.

**Metrics evaluated:**

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| total_cases | 200 | — | — |
| passed_cases | 200 | — | — |
| pass_rate | 1.0 | — | — |
| hit_rate_at_5 | 1.0 | ≥ 0.85 | ✅ PASS |
| mrr | 1.0 | ≥ 0.65 | ✅ PASS |
| faithfulness | 1.0 | ≥ 0.90 | ✅ PASS |
| answer_correctness | 1.0 | ≥ 0.80 | ✅ PASS |
| llm_judge_avg | 3.775 | ≥ 4.0 | ❌ FAIL |
| context_precision | 0.667 | ≥ 0.80 | ❌ FAIL |
| golden_set_pass_rate | 1.0 | ≥ 0.95 | ✅ PASS |
| answer_stability | Metric not available in existing artifacts | ≥ 0.90 | ❓ NOT EVALUATED |
| regulatory_compliance | 1.0 | ≥ 0.90 | ✅ PASS |
| role_appropriateness | 1.0 | ≥ 1.0 | ✅ PASS |
| hitl_trigger_precision | 1.0 | ≥ 0.85 | ✅ PASS |
| all_metrics_passed | true (summary) | — | — |

**Results summary:**
- The full regression run (`reports/regression_report.json`, 2026-07-22) reports 200/200 cases passed with pass_rate = 1.0.
- Failure analysis (`reports/report.md`, `reports/summary.json`) documents an earlier run with 1/10 passed and 9 failed cases, indicating the system underwent significant remediation between early and latest evaluations.
- Per-case Week 6 threshold analysis reveals **context_precision** failures across all 200 cases (actual=0.667 against threshold=0.80).
- The LLM Judge average score of 3.775 falls below the 4.0 threshold.
- All custom metrics pass with the exception of answer_stability which is null/not evaluated in the available source data.

**Pass/fail threshold summary (if available):**

| Metric | Threshold | Source |
|--------|-----------|--------|
| Hit Rate @ 5 | ≥ 0.85 | `eval/regression_suite.py` (Week 6 thresholds) |
| MRR | ≥ 0.65 | `eval/regression_suite.py` |
| NDCG | ≥ 0.75 | `eval/regression_suite.py` |
| Context Precision | ≥ 0.80 | `eval/regression_suite.py` |
| Context Recall | ≥ 0.85 | `eval/regression_suite.py` |
| Faithfulness | ≥ 0.90 | `eval/regression_suite.py` |
| Answer Correctness | ≥ 0.80 | `eval/regression_suite.py` |
| LLM Judge Avg | ≥ 4.0/5.0 | `eval/regression_suite.py` |
| Golden Set Pass Rate | ≥ 0.95 | `eval/custom_metrics.py` |
| Answer Stability | ≥ 0.90 | `eval/custom_metrics.py` |
| Regulatory Compliance | ≥ 0.90 | `eval/custom_metrics.py` |
| Role Appropriateness | 1.00 | `eval/custom_metrics.py` |
| HITL Trigger Precision | ≥ 0.85 | `eval/custom_metrics.py` |

**Observations:**
1. The evaluation framework is fully operational with intrinsic, extrinsic, LLM-as-judge, and custom metric modules implemented.
2. Context precision at 0.667 is a systemic gap — all 200 cases fail this individual threshold, suggesting the retriever returns too many non-relevant chunks in the top-K results.
3. LLM Judge average score (3.775) is below the 4.0 acceptance threshold, indicating room for improvement in answer quality as judged by the LLM evaluator.
4. Answer stability was not evaluated at this point (`answer_stability: null` in the regression report), meaning the system lacks dual-run semantic similarity measurement.
5. The early failure analysis (10-case run) showed 4 retrieval failures and 5 answer quality failures, providing a diagnostic baseline for remediation priorities.
6. Regulatory compliance, role appropriateness, and HITL trigger precision all score 1.0, but these may reflect early-stage implementation where the testing scenarios were less rigorous.
7. The remediated state described in `reports/remediation_baseline.md` documents known retriever configuration issues: BM25-only mode, OpenAI embedding model access failure (403), and cross-encoder fallback behavior.

**Known gaps:**
1. Answer stability metric is not available — dual-run stability evaluation was not wired at the Week 4 baseline stage.
2. Context precision below threshold (0.667 vs 0.80) with no documented remediation plan at this stage.
3. LLM Judge average score below acceptance threshold (3.775 vs 4.0).
4. Exact Week-4 specific evaluation timestamps are not preserved; values are derived from the closest available regression run (2026-07-22).
5. Failure analysis (10 cases) shows mostly synthetic failures rather than real-world production issues, limiting diagnostic value.
6. The early eval package (`eval_baseline.md`, `eval_final.md`) provides high-level structural descriptions without detailed metric data, so Week 4 values must reference later regression data.

**Reviewer evidence references:**
- `reports/eval_baseline.md` — Initial evaluation framework description
- `reports/eval_final.md` — Initial evaluation summary report
- `reports/report.md` — Failure analysis with 10-case breakdown
- `reports/summary.json` — Summary metrics for early evaluation round
- `reports/regression_report.json` — Full 200-case regression run with per-case metrics
- `reports/remediation_baseline.md` — Retriever configuration diagnostic
- `docs/eval_methodology.md` — Evaluation framework methodology document
- `eval/regression_suite.py` — Regression suite implementation and thresholds
- `eval/custom_metrics.py` — Custom metric definitions and thresholds