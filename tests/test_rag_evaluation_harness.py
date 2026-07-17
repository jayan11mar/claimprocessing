"""Tests for the RAG evaluation harness and its components.

Covers:
  - evaluate_rag_queries basic pass/fail logic
  - Settings threshold configuration
  - LLM judge fallback behavior (sentence-transformers)
  - JUDGE_MODEL_NAME + JUDGE_ANTHROPIC_API_KEY configuration
  - Week 6 threshold comparison
  - SemanticSimilarityScorer fallback
"""

import os
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np
import pytest

from app.config import get_settings
from app.rag.chunkers import Chunk
from app.rag.evaluation_harness import evaluate_rag_queries

from eval.custom_metrics import (
    SemanticSimilarityScorer,
    compute_answer_stability,
    compute_golden_set_pass_rate,
    compute_hitl_trigger_precision,
    compute_regulatory_compliance,
    compute_role_appropriateness,
)
from eval.llm_judge import judge_answer, judge_pairwise, _get_judge_config
from eval.regression_suite import (
    WEEK_6_THRESHOLDS,
    compute_week6_pass_fail,
    evaluate_single_case,
    load_golden_set,
)


# =========================================================================
# evaluate_rag_queries tests (existing)
# =========================================================================


def test_evaluate_rag_queries_marks_case_passed_when_keywords_and_citations_are_present():
    chunks = [
        Chunk(
            text="Hospitalization coverage applies for inpatient care and related treatment expenses.",
            source_id="policy-1",
            source_path="policy.md",
            doc_type="policy_wording",
            insurance_type="health",
            chunk_index=0,
        )
    ]
    cases = [
        {
            "name": "coverage lookup",
            "query": "Is hospitalization covered?",
            "expected_keywords": ["hospitalization", "coverage"],
            "min_retrieval_score": 0.5,
            "min_answer_score": 0.5,
            "min_citations": 1,
        }
    ]

    def fake_retrieval(chunks_to_search, query, k=3):
        return [
            {
                "chunk": chunks_to_search[0],
                "combined_score": 0.92,
                "source_id": chunks_to_search[0].source_id,
                "source_path": chunks_to_search[0].source_path,
            }
        ]

    def fake_answer(query, chunks=None, claim_context=None, top_k=3, embedding_fn=None):
        return {
            "answer_text": "Hospitalization coverage applies for inpatient care.",
            "citations": [{"source_id": "policy-1", "source_path": "policy.md"}],
            "confidence": 0.91,
        }

    report = evaluate_rag_queries(
        cases=cases,
        chunks=chunks,
        retrieval_fn=fake_retrieval,
        answer_fn=fake_answer,
    )

    assert report["summary"]["passed_cases"] == 1
    assert report["cases"][0]["passed"] is True
    assert report["cases"][0]["retrieval_score"] >= 0.5
    assert report["cases"][0]["answer_score"] >= 0.5


def test_settings_reads_rag_evaluation_thresholds(monkeypatch):
    get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RAG_EVALUATION_CONTEXT", "aml / fraud")
    monkeypatch.setenv("RAG_EVALUATION_HIT_RATE_THRESHOLD", "0.8")
    monkeypatch.setenv("RAG_EVALUATION_MRR_THRESHOLD", "0.7")
    monkeypatch.setenv("RAG_EVALUATION_FAITHFULNESS_THRESHOLD", "0.95")
    monkeypatch.setenv("RAG_EVALUATION_ANSWER_CORRECTNESS_THRESHOLD", "0.85")
    monkeypatch.setenv("RAG_EVALUATION_LLM_JUDGE_AVG_THRESHOLD", "4.2")
    monkeypatch.setenv("RAG_EVALUATION_CITATION_COVERAGE_THRESHOLD", "1.0")
    monkeypatch.setenv("RAG_EVALUATION_MIN_CITATIONS", "2")

    settings = get_settings()

    assert settings.RAG_EVALUATION_CONTEXT == "aml / fraud"
    assert settings.RAG_EVALUATION_HIT_RATE_THRESHOLD == 0.8
    assert settings.RAG_EVALUATION_MRR_THRESHOLD == 0.7
    assert settings.RAG_EVALUATION_FAITHFULNESS_THRESHOLD == 0.95
    assert settings.RAG_EVALUATION_ANSWER_CORRECTNESS_THRESHOLD == 0.85
    assert settings.RAG_EVALUATION_LLM_JUDGE_AVG_THRESHOLD == 4.2
    assert settings.RAG_EVALUATION_CITATION_COVERAGE_THRESHOLD == 1.0
    assert settings.RAG_EVALUATION_MIN_CITATIONS == 2

    get_settings.cache_clear()


# =========================================================================
# Judge config tests
# =========================================================================


def test_judge_config_defaults_to_none():
    """JUDGE_MODEL_NAME and JUDGE_ANTHROPIC_API_KEY should default to None."""
    judge_model, api_key = _get_judge_config()
    assert judge_model is None
    assert api_key is None


def test_judge_config_reads_env(monkeypatch):
    """JUDGE_MODEL_NAME and JUDGE_ANTHROPIC_API_KEY should be read from env."""
    monkeypatch.setenv("JUDGE_MODEL_NAME", "claude-sonnet-4-20250514")
    monkeypatch.setenv("JUDGE_ANTHROPIC_API_KEY", "sk-ant-test123")
    judge_model, api_key = _get_judge_config()
    assert judge_model == "claude-sonnet-4-20250514"
    assert api_key == "sk-ant-test123"


def test_judge_config_partial(monkeypatch):
    """Only JUDGE_MODEL_NAME set should return None for api_key."""
    monkeypatch.setenv("JUDGE_MODEL_NAME", "claude-sonnet-4-20250514")
    monkeypatch.delenv("JUDGE_ANTHROPIC_API_KEY", raising=False)
    judge_model, api_key = _get_judge_config()
    assert judge_model == "claude-sonnet-4-20250514"
    assert api_key is None


# =========================================================================
# LLM judge fallback tests (no JUDGE_ANTHROPIC_API_KEY)
# =========================================================================


def test_judge_answer_fallback_when_no_api_key(monkeypatch):
    """judge_answer should fall back when no JUDGE_ANTHROPIC_API_KEY."""
    monkeypatch.delenv("JUDGE_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("JUDGE_MODEL_NAME", raising=False)

    result = judge_answer(
        query="What is covered?",
        answer="Hospitalization coverage applies.",
        expected_answer="Hospitalization coverage for inpatient care is covered.",
        retrieved_chunks=["Hospitalization coverage applies for inpatient care."],
    )

    assert result["is_fallback"] is True
    assert result["judge_model"] == "no-judge-configured"
    assert "fallback" in result["reason"]
    assert result["overall_score"] >= 0
    assert result["overall_score"] <= 5


def test_judge_answer_fallback_reason_semantic_similarity(monkeypatch):
    """When sentence-transformers is available, the fallback reason should mention it."""
    monkeypatch.delenv("JUDGE_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("JUDGE_MODEL_NAME", raising=False)

    result = judge_answer(
        query="What is covered?",
        answer="Hospitalization coverage applies.",
        expected_answer="Hospitalization coverage for inpatient care is covered.",
    )

    assert "semantic-similarity" in result["reason"].lower() or "token-overlap" in result["reason"].lower()
    assert "gpt-4o-mini" not in result["reason"]


def test_judge_pairwise_fallback_when_no_api_key(monkeypatch):
    """judge_pairwise should fall back when no JUDGE_ANTHROPIC_API_KEY."""
    monkeypatch.delenv("JUDGE_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("JUDGE_MODEL_NAME", raising=False)

    result = judge_pairwise(
        query="What is covered?",
        answer_a="Hospitalization coverage applies.",
        answer_b="Coverage for hospitalization.",
        expected_answer="Hospitalization coverage for inpatient care is covered.",
        randomize_labels=False,
    )

    assert result["is_fallback"] is True
    assert result["judge_model"] == "no-judge-configured"
    assert "fallback" in result["reason"]
    assert result["answer_a"]["overall_score"] >= 0
    assert result["answer_b"]["overall_score"] >= 0


def test_judge_answer_uses_claude_when_configured(monkeypatch):
    """When JUDGE_ANTHROPIC_API_KEY is set, judge_answer should attempt Claude."""
    monkeypatch.setenv("JUDGE_MODEL_NAME", "claude-sonnet-4-20250514")
    monkeypatch.setenv("JUDGE_ANTHROPIC_API_KEY", "sk-ant-test123")

    # Since the API key is fake, it should fail and fall back
    result = judge_answer(
        query="What is covered?",
        answer="Hospitalization coverage applies.",
        expected_answer="Hospitalization coverage for inpatient care is covered.",
    )

    # The Claude call will fail with the fake key, so it falls through
    assert result["is_fallback"] is True
    assert result["judge_model"] == "claude-sonnet-4-20250514"


def test_judge_answer_generation_model_read_from_env(monkeypatch):
    """generation_model should be read from LLM_GENERATION_MODEL or OPENAI_MODEL_NAME."""
    monkeypatch.setenv("LLM_GENERATION_MODEL", "gpt-4")
    monkeypatch.delenv("JUDGE_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("JUDGE_MODEL_NAME", raising=False)

    result = judge_answer(
        query="What is covered?",
        answer="Hospitalization coverage applies.",
        expected_answer="Hospitalization coverage for inpatient care is covered.",
    )

    assert result["generation_model"] == "gpt-4"


def test_judge_answer_never_falls_back_to_generation_model(monkeypatch):
    """The fallback should NEVER use gpt-4o-mini or any generation model as judge."""
    monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    monkeypatch.delenv("JUDGE_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("JUDGE_MODEL_NAME", raising=False)

    result = judge_answer(
        query="What is covered?",
        answer="Hospitalization coverage applies.",
        expected_answer="Hospitalization coverage for inpatient care is covered.",
    )

    # The judge model should never be gpt-4o-mini
    assert result["judge_model"] != "gpt-4o-mini"
    assert "gpt-4o-mini" not in result["reason"]


# =========================================================================
# SemanticSimilarityScorer tests
# =========================================================================


def test_semantic_similarity_scorer_jaccard_fallback():
    """SemanticSimilarityScorer should return a valid similarity score."""
    scorer = SemanticSimilarityScorer()
    sim = scorer.similarity(
        "Hospitalization coverage applies for inpatient care.",
        "Coverage for hospitalization of inpatients.",
    )
    assert 0.0 <= sim <= 1.0


def test_semantic_similarity_scorer_identical_texts():
    """Identical texts should have similarity ~1.0."""
    scorer = SemanticSimilarityScorer()
    text = "Hospitalization coverage applies for inpatient care."
    sim = scorer.similarity(text, text)
    assert sim > 0.8


def test_semantic_similarity_scorer_different_texts():
    """Very different texts should have low similarity."""
    scorer = SemanticSimilarityScorer()
    sim = scorer.similarity(
        "Hospitalization coverage applies for inpatient care.",
        "The weather is sunny today.",
    )
    assert sim < 0.5


def test_semantic_similarity_scorer_empty_text():
    """Empty text should not crash."""
    scorer = SemanticSimilarityScorer()
    sim = scorer.similarity("", "Hospitalization coverage applies.")
    assert 0.0 <= sim <= 1.0


# =========================================================================
# Week 6 threshold comparison tests
# =========================================================================


def test_week6_thresholds_exist():
    """WEEK_6_THRESHOLDS should contain all expected metric keys."""
    expected_keys = [
        "hit_rate_at_5", "mrr", "faithfulness", "answer_correctness",
        "llm_judge_avg", "citation_coverage", "ndcg",
        "context_precision", "context_recall",
    ]
    for key in expected_keys:
        assert key in WEEK_6_THRESHOLDS, f"Missing Week 6 threshold: {key}"


def test_compute_week6_pass_fail_all_pass():
    """When all metrics exceed thresholds, week6_passed should be True."""
    result = {
        "intrinsic": {
            "hit_at_k": 0.95,
            "mrr": 0.80,
            "ndcg": 0.85,
            "context_precision": 0.90,
            "context_recall": 0.95,
        },
        "extrinsic": {
            "faithfulness": 0.95,
            "answer_correctness": 0.90,
        },
        "judge": {
            "overall_score": 4.5 / 5.0,
        },
    }
    week6 = compute_week6_pass_fail(result)
    assert week6["week6_passed"] is True
    assert len(week6["week6_failures"]) == 0


def test_compute_week6_pass_fail_some_fail():
    """When metrics are below thresholds, week6_passed should be False."""
    result = {
        "intrinsic": {
            "hit_at_k": 0.50,
            "mrr": 0.40,
            "ndcg": 0.50,
            "context_precision": 0.60,
            "context_recall": 0.50,
        },
        "extrinsic": {
            "faithfulness": 0.60,
            "answer_correctness": 0.50,
        },
        "judge": {
            "overall_score": 2.0 / 5.0,
        },
    }
    week6 = compute_week6_pass_fail(result)
    assert week6["week6_passed"] is False
    assert len(week6["week6_failures"]) > 0


def test_compute_week6_pass_fail_thresholds_override():
    """Threshold overrides should work."""
    result = {
        "intrinsic": {
            "hit_at_k": 0.70,
            "mrr": 0.60,
            "ndcg": 0.65,
            "context_precision": 0.70,
            "context_recall": 0.75,
        },
        "extrinsic": {
            "faithfulness": 0.80,
            "answer_correctness": 0.75,
        },
        "judge": {
            "overall_score": 4.0 / 5.0,
        },
    }
    # Override thresholds to be lower
    relaxed = {k: v - 0.2 for k, v in WEEK_6_THRESHOLDS.items()}
    week6 = compute_week6_pass_fail(result, thresholds=relaxed)
    assert week6["week6_passed"] is True


def test_compute_week6_pass_fail_returns_comparisons():
    """Should return per-metric comparisons with actual, threshold, passed."""
    result = {
        "intrinsic": {
            "hit_at_k": 0.95,
            "mrr": 0.80,
            "ndcg": 0.85,
            "context_precision": 0.90,
            "context_recall": 0.95,
        },
        "extrinsic": {
            "faithfulness": 0.95,
            "answer_correctness": 0.90,
        },
        "judge": {
            "overall_score": 4.5 / 5.0,
        },
    }
    week6 = compute_week6_pass_fail(result)
    comparisons = week6["week6_comparisons"]
    assert "hit_rate_at_5" in comparisons
    assert comparisons["hit_rate_at_5"]["actual"] == 0.95
    assert comparisons["hit_rate_at_5"]["passed"] is True
    assert comparisons["mrr"]["passed"] is True


# =========================================================================
# evaluate_single_case integration tests
# =========================================================================


def test_evaluate_single_case_returns_expected_structure():
    """evaluate_single_case should return a well-structured result."""
    result = evaluate_single_case(
        query="What is covered?",
        expected_answer="Hospitalization coverage for inpatient care is covered.",
        expected_chunks=["Hospitalization coverage applies for inpatient care."],
        retrieved_chunks=["Hospitalization coverage applies for inpatient care."],
        case_id="test-001",
        difficulty="easy",
        project="test",
    )

    assert result["id"] == "test-001"
    assert result["difficulty"] == "easy"
    assert result["project"] == "test"
    assert "intrinsic" in result
    assert "extrinsic" in result
    assert "judge" in result
    assert "passed" in result


# =========================================================================
# Custom metrics tests
# =========================================================================


def test_golden_set_pass_rate_all_pass():
    """compute_golden_set_pass_rate with all passing cases."""
    results = [
        {
            "id": "case-1",
            "query": "test",
            "intrinsic": {"hit_at_k": 0.95, "mrr": 0.80},
            "extrinsic": {"faithfulness": 0.95, "answer_correctness": 0.90},
            "judge": {"overall_score": 4.5 / 5.0},
        },
        {
            "id": "case-2",
            "query": "test2",
            "intrinsic": {"hit_at_k": 0.90, "mrr": 0.75},
            "extrinsic": {"faithfulness": 0.92, "answer_correctness": 0.85},
            "judge": {"overall_score": 4.2 / 5.0},
        },
    ]
    result = compute_golden_set_pass_rate(results)
    assert result["pass_rate"] == 1.0
    assert result["passed_count"] == 2


def test_golden_set_pass_rate_some_fail():
    """compute_golden_set_pass_rate with some failures."""
    results = [
        {
            "id": "case-1",
            "query": "test",
            "intrinsic": {"hit_at_k": 0.95, "mrr": 0.80},
            "extrinsic": {"faithfulness": 0.95, "answer_correctness": 0.90},
            "judge": {"overall_score": 4.5 / 5.0},
        },
        {
            "id": "case-2",
            "query": "test2",
            "intrinsic": {"hit_at_k": 0.30, "mrr": 0.25},
            "extrinsic": {"faithfulness": 0.40, "answer_correctness": 0.35},
            "judge": {"overall_score": 2.0 / 5.0},
        },
    ]
    result = compute_golden_set_pass_rate(results)
    assert result["pass_rate"] == 0.5
    assert result["passed_count"] == 1


def test_answer_stability_identical():
    """compute_answer_stability with identical answers."""
    answers = ["Hospitalization coverage applies.", "Coverage for inpatient care."]
    result = compute_answer_stability(answers_a=answers, answers_b=answers)
    assert result["stability_score"] >= 0.0


def test_answer_stability_different():
    """compute_answer_stability with different answers."""
    result = compute_answer_stability(
        answers_a=["Hospitalization coverage applies."],
        answers_b=["The weather is sunny today."],
    )
    assert result["stability_score"] < 0.8


def test_regulatory_compliance_empty():
    """compute_regulatory_compliance with empty list."""
    result = compute_regulatory_compliance([])
    assert result["compliance_score"] == 0.0


def test_role_appropriateness_no_violations():
    """compute_role_appropriateness with a customer that has no violations."""
    result = compute_role_appropriateness(
        answers=["Your policy covers hospitalization."],
        role_contexts=["customer"],
    )
    assert result["appropriateness_score"] == 1.0


def test_hitl_trigger_precision_all_accurate():
    """compute_hitl_trigger_precision with all accurate triggers."""
    decisions = [
        {"task_id": "t1", "triggered": True, "approved": True},
        {"task_id": "t2", "triggered": True, "approved": True},
        {"task_id": "t3", "triggered": False},
    ]
    result = compute_hitl_trigger_precision(decisions)
    assert result["precision"] == 1.0
    assert result["true_positives"] == 2


def test_hitl_trigger_precision_some_false_positives():
    """compute_hitl_trigger_precision with false positives."""
    decisions = [
        {"task_id": "t1", "triggered": True, "approved": True},
        {"task_id": "t2", "triggered": True, "approved": False},
        {"task_id": "t3", "triggered": True, "approved": True},
    ]
    result = compute_hitl_trigger_precision(decisions)
    expected = round(2 / 3, 4)
    assert result["precision"] == expected
    assert result["true_positives"] == 2
    assert result["false_positives"] == 1


def test_compute_week6_pass_fail_missing_metrics():
    """compute_week6_pass_fail with missing metrics should treat them as 0."""
    result = {
        "intrinsic": {},
        "extrinsic": {},
        "judge": {},
    }
    week6 = compute_week6_pass_fail(result)
    assert week6["week6_passed"] is False
    assert "hit_rate_at_5" in week6["week6_failures"]