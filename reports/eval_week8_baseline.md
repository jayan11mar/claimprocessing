# Week 8 RAG Evaluation Baseline Report

**Date**: 2026-07-17  
**Suite**: Production-Grade Regression + Monitoring Suite  
**Spec Reference**: 3.5 (RAG Evaluation - Expanded)  
**Status**: Baseline Established

---

## Overview

This report establishes the baseline metrics for the production-grade RAG evaluation suite. All five Spec 3.5 custom metrics are measured against the golden set and recorded for future regression comparison.

### Golden Set Summary

| Metric | Value |
|--------|-------|
| Total Cases | 130 (50 claims/insurance + 50 customer svc + 30 loan/aml) |
| Baseline Pass Rate | 94.6% |
| Projects Evaluated | 3 |

---

## Custom Metrics (Spec 3.5)

### 1. Golden Set Pass Rate

- **Target**: >= 95%
- **Measured**: 94.6%
- **Status**: ⚠️ Below threshold (0.4% deficit)

**Per-project breakdown**:

| Project | Pass Rate | Status |
|---------|-----------|--------|
| claims/insurance | 96.0% | ✅ Pass |
| customer svc | 94.0% | ❌ Fail |
| loan/underwriting | 93.3% | ❌ Fail |

### 2. Answer Stability

- **Target**: >= 0.90
- **Measured**: 0.92
- **Status**: ✅ Pass

Answer stability measured across two runs of the golden set using semantic similarity (offline sentence-transformers fallback: n-gram Jaccard similarity).

| Statistic | Value |
|-----------|-------|
| Mean Similarity | 0.92 |
| Min Similarity | 0.78 |
| Max Similarity | 0.99 |
| Std Dev | 0.06 |

### 3. Regulatory Compliance

- **Target**: >= 0.90
- **Measured**: 0.93
- **Status**: ✅ Pass

Compliance measured via weighted keyword matching against 25 IRDAI/insurance regulatory patterns.

**Top matched patterns**:

| Pattern | Frequency |
|---------|-----------|
| Claim settlement | 87 |
| Exclusion mention | 76 |
| Sum insured mention | 72 |
| Reimbursement mention | 65 |
| Pre-existing disease | 58 |
| Waiting period mention | 52 |
| Network hospital | 48 |
| IRDAI regulation | 42 |

### 4. Role Appropriateness

- **Target**: 100%
- **Measured**: 100%
- **Status**: ✅ Pass

No information leakage detected across any role context. All answers respect role-based access constraints.

| Role | Cases | Appropriate | Violations |
|------|-------|-------------|------------|
| customer | 50 | 50 (100%) | 0 |
| agent | 50 | 50 (100%) | 0 |
| underwriter | 30 | 30 (100%) | 0 |

### 5. HITL Trigger Precision

- **Target**: >= 0.85
- **Measured**: 0.90
- **Status**: ✅ Pass

| Metric | Value |
|--------|-------|
| True Positives | 36 |
| False Positives | 4 |
| Total Triggers | 40 |
| Precision | 0.90 |

---

## Metric Drift Baselines

The following per-metric drift thresholds are established for future comparison:

| Metric | Baseline Value | Drift Threshold |
|--------|---------------|-----------------|
| hit_rate_at_5 | 0.89 | ±10% |
| mrr | 0.72 | ±10% |
| faithfulness | 0.94 | ±10% |
| answer_correctness | 0.88 | ±10% |
| llm_judge_avg | 4.2/5.0 | ±10% |
| ndcg | 0.81 | ±10% |
| context_precision | 0.85 | ±10% |
| context_recall | 0.90 | ±10% |

---

## Distribution Profile (Judge Scores)

| Statistic | Value |
|-----------|-------|
| Mean | 0.84 |
| Median | 0.86 |
| Std Dev | 0.09 |
| P5 | 0.68 |
| P25 | 0.78 |
| P75 | 0.92 |
| P95 | 0.97 |

KS threshold for drift detection: **0.30**  
PSI threshold for drift detection: **0.25**

---

## Failure Analysis

### Retrieval Failures
- customer svc RAG-12: hit_rate_at_5 = 0.60 (threshold: 0.85)
- customer svc RAG-24: mrr = 0.50 (threshold: 0.65)
- loan/underwriting RAG-08: hit_rate_at_5 = 0.40 (threshold: 0.85)

### Answer Quality Failures
- customer svc RAG-18: faithfulness = 0.82 (threshold: 0.90)
- loan/underwriting RAG-15: answer_correctness = 0.75 (threshold: 0.80)

### Citation Failures
- customer svc RAG-30: judge_citation_quality = 2.0/5.0
- loan/underwriting RAG-22: judge_citation_quality = 2.5/5.0

---

## CI Export Summary

| Format | File |
|--------|------|
| JSON | `reports/regression_report.json` |
| CSV | `reports/evaluation_results.csv` |
| JUnit XML | `reports/evaluation_results.xml` |
| Markdown | `reports/evaluation_summary.md` |

---

## Recommendations

1. **Customer Svc**: Investigate retrieval pipeline for customer service domain — hit rate and MRR below thresholds for 3 cases.
2. **Loan Underwriting**: Improve chunk relevance scoring and citation quality for loan domain queries.
3. **Golden Set Pass Rate**: Address the 0.4% deficit by improving retrieval for hard-difficulty cases.
4. **Monitoring**: Establish CI pipeline to run regression suite on every prompt/model deploy and compare against this baseline.