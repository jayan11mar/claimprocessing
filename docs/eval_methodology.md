# Evaluation Methodology

## Purpose
This document describes the evaluation framework for the claims processing RAG system, including the regression runner, intrinsic/extrinsic/custom metrics, acceptance thresholds, the trend dashboard, and known test coverage gaps.

## Regression Runner

**File:** `eval/regression_suite.py`

The regression runner evaluates the RAG pipeline against a **golden set** of test cases and produces a structured report with per-case and aggregate metrics.

### Key Functions
- `load_golden_set(path)` — Loads test cases from `eval/golden_set.json` or `eval/eval_set.json`
- `evaluate_single_case(...)` — Runs full evaluation on one query with expected answer and chunks
- `run_regression(...)` — Full regression run with summary, custom metrics, and baseline comparison
- `compare_to_baseline(current, baseline)` — Delta comparison (regressions, improvements, pass rate delta)
- `compute_week6_pass_fail(result)` — Compares single case against Week 6 thresholds

### Data Sources
- `eval/golden_set.json` — Primary golden set (project-based format)
- `eval/eval_set.json` — Fallback evaluation set (flat format)

## Evaluation Metrics

### Intrinsic Metrics (`eval/intrinsic.py`)
Retrieval quality metrics:
- **Hit Rate @ K** — Whether at least one relevant chunk is in top-K (threshold: ≥0.85)
- **MRR (Mean Reciprocal Rank)** — Rank of first relevant chunk (threshold: ≥0.65)
- **NDCG@K** — Normalized Discounted Cumulative Gain (threshold: ≥0.75)
- **Context Precision** — Precision of retrieved context (threshold: ≥0.80)
- **Context Recall** — Recall of retrieved context (threshold: ≥0.85)

### Extrinsic Metrics (`eval/extrinsic.py`)
Answer quality metrics:
- **Faithfulness** — Whether answer is grounded in retrieved chunks (threshold: ≥0.90)
- **Answer Correctness** — Whether answer matches expected answer (threshold: ≥0.80)

### LLM-as-Judge (`eval/llm_judge.py`)
- **Overall Score** — LLM evaluation of answer quality (threshold: ≥4.0/5.0)

## Custom Metrics (Spec 3.5)

**File:** `eval/custom_metrics.py` — **5 custom metrics**:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Golden Set Pass Rate | ≥0.95 | Fraction of cases passing all metric thresholds |
| Answer Stability | ≥0.90 | Semantic similarity between two runs of same queries |
| Regulatory Compliance | ≥0.90 | How well answers reference insurance regulatory language |
| Role Appropriateness | 1.00 (100%) | No leakage of restricted content to roles |
| HITL Trigger Precision | ≥0.85 | Precision of HITL trigger decisions |

### Key Components
- `SemanticSimilarityScorer` — Sentence-transformer based scorer (falls back to character n-gram Jaccard)
- `compute_golden_set_pass_rate(results)` — Multi-threshold case pass/fail
- `compute_answer_stability(answers_a, answers_b)` — Pairwise semantic similarity
- `compute_regulatory_compliance(answers)` — 27 regulatory pattern matchers
- `compute_role_appropriateness(answers, roles)` — Keyword-based restriction checker
- `compute_hitl_trigger_precision(decisions)` — True positive / false positive analysis

## Acceptance Thresholds

Defined in `eval/regression_suite.py` (Week 6 thresholds) and `eval/custom_metrics.py`:

| Metric | Threshold |
|--------|-----------|
| Hit Rate @ 5 | ≥0.85 |
| MRR | ≥0.65 |
| NDCG | ≥0.75 |
| Context Precision | ≥0.80 |
| Context Recall | ≥0.85 |
| Faithfulness | ≥0.90 |
| Answer Correctness | ≥0.80 |
| LLM Judge Avg | ≥4.0/5.0 |
| Golden Set Pass Rate | ≥0.95 |
| Answer Stability | ≥0.90 |
| Regulatory Compliance | ≥0.90 |
| Role Appropriateness | 1.00 |
| HITL Trigger Precision | ≥0.85 |

## Trend Dashboard

### Data Preparation (`eval/dashboard.py`)
- `prepare_trend_data(regression_results)` — Transforms multiple regression reports into chart-ready structure
- `extract_all_metrics(regression_data)` — Extracts 6 top-level metrics into flat dict
- `build_trend_rows(reports)` — List of trend-friendly rows
- `extract_pass_rate()`, `extract_golden_set_pass_rate()`, `extract_answer_stability()`, `extract_regulatory_compliance()`, `extract_role_appropriateness()`, `extract_hitl_trigger_precision()` — Individual metric extractors
- `extract_comparison_deltas()` — Baseline comparison deltas

### Streamlit Dashboard (`app/frontend/streamlit_app.py`, lines 665-832)
- **Tab "📊 Evaluation Dashboard"**
- "🔄 Fetch Evaluation" button calls `/eval/regression`
- Displays:
  - **Pass Rate** — Overall pass/fail ratio
  - **Acceptance Thresholds** — Comparison table with pass/fail indicators
  - **Metric Trends** — Line charts for available metrics over time
  - **Raw Data** — Expandable dataframe

## Latest Available Reports

| Report | Path | Size |
|--------|------|------|
| Week 8 Baseline | `reports/eval_week8_baseline.md` | 685 bytes |
| Week 8 Final | `reports/eval_week8_final.md` | 2,842 bytes |
| Earlier Baseline | `reports/eval_baseline.md` | 402 bytes |
| Earlier Final | `reports/eval_final.md` | 503 bytes |
| Regression Report (JSON) | `reports/regression_report.json` | Variable |
| Regression Summary (JSON) | `reports/regression_summary.json` | Variable |
| Drift Baseline (JSON) | `reports/drift_baseline.json` | Variable |
| Remediation Baseline | `reports/remediation_baseline.md` | 3,455 bytes |

## Additional Evaluation Components

| File | Purpose |
|------|---------|
| `eval/intrinsic.py` | Retrieval metrics (hit rate, MRR, NDCG, context precision/recall) |
| `eval/extrinsic.py` | Answer metrics (faithfulness, correctness) |
| `eval/llm_judge.py` | LLM-as-judge scoring |
| `eval/comparator.py` | Answer comparison utilities |
| `eval/failure_analysis.py` | Bucket failures by type/category for targeted improvement |
| `eval/drift.py` | Data drift detection for RAG |
| `eval/export.py` | Export evaluation results to various formats |
| `eval/run_eval.py` | CLI entry point for evaluation runs |
| `eval/run_failure_eval.py` | Focused failure analysis evaluation |

## Known Test Coverage Gaps

Based on test mapping:

| Area | Test File | Tests | Coverage Assessment |
|------|-----------|-------|---------------------|
| Regression suite | `tests/test_eval_package.py` | 6 | ⚠️ Minimal — only package structure tests; no direct tests for `run_regression()`, `evaluate_single_case()`, or `compare_to_baseline()` |
| Custom metrics | (covered by test_eval_package) | — | ❌ No direct tests for `compute_all_custom_metrics()`, `compute_golden_set_pass_rate()`, etc. |
| Dashboard | — | — | ❌ No dedicated tests for `dashboard.py` functions |
| Intrinsic metrics | `tests/test_rag_evaluation_harness.py` | 35 | ✅ Comprehensive |
| Failure analysis | — | — | ❌ No dedicated tests |

## Reviewer Demo

```bash
# View evaluation framework structure
ls -la eval/*.py

# Run package structure tests
python -m pytest tests/test_eval_package.py -v

# Run RAG evaluation harness tests
python -m pytest tests/test_rag_evaluation_harness.py -v

# View available evaluation metrics (dry run of regression config)
python -c "
from eval.regression_suite import WEEK_6_THRESHOLDS
print('Week 6 Thresholds:')
for k, v in WEEK_6_THRESHOLDS.items():
    print(f'  {k}: {v}')
print()
from eval.custom_metrics import compute_all_custom_metrics
import inspect
sig = inspect.signature(compute_all_custom_metrics)
print(f'compute_all_custom_metrics params: {list(sig.parameters.keys())}')
"
```
