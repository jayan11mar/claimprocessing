"""A/B comparator for prompt/model version evaluation.

Compares two sets of answers (e.g., from different prompt versions or
different models) and determines which performs better using:
  - Pairwise LLM judge scoring with label randomization (anti position-bias)
  - Semantic similarity comparison
  - Per-metric win/loss/tie classification
  - Statistical significance testing (paired bootstrap)
"""

import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from eval.custom_metrics import SemanticSimilarityScorer, compute_answer_stability
from eval.llm_judge import judge_answer, judge_pairwise


# ---------------------------------------------------------------------------
# A/B evaluation on a single query pair
# ---------------------------------------------------------------------------

def compare_single_pair(
    query: str,
    answer_a: str,
    answer_b: str,
    expected_answer: str,
    retrieved_chunks: Optional[List[str]] = None,
    randomize_labels: bool = True,
) -> Dict[str, Any]:
    """Compare two answers for a single query using pairwise LLM judging.

    Args:
        query: The original query.
        answer_a: Answer from version A.
        answer_b: Answer from version B.
        expected_answer: The expected/reference answer.
        retrieved_chunks: Optional retrieved chunk texts.
        randomize_labels: If True, randomly swap A/B labels.

    Returns:
        Dict with pairwise judge results, individual judge scores, and
        semantic similarity comparison.
    """
    # Pairwise comparison via LLM judge (with randomization)
    pairwise = judge_pairwise(
        query=query,
        answer_a=answer_a,
        answer_b=answer_b,
        expected_answer=expected_answer,
        retrieved_chunks=retrieved_chunks,
        randomize_labels=randomize_labels,
    )

    # Individual judge scores
    judge_a = judge_answer(
        query=query,
        answer=answer_a,
        expected_answer=expected_answer,
        retrieved_chunks=retrieved_chunks,
    )
    judge_b = judge_answer(
        query=query,
        answer=answer_b,
        expected_answer=expected_answer,
        retrieved_chunks=retrieved_chunks,
    )

    # Semantic similarity between A and B
    scorer = SemanticSimilarityScorer()
    semantic_sim = scorer.similarity(answer_a, answer_b)

    # Determine winner
    score_a = pairwise["answer_a"]["overall_score"]
    score_b = pairwise["answer_b"]["overall_score"]

    if score_a > score_b + 0.1:
        winner = "A"
    elif score_b > score_a + 0.1:
        winner = "B"
    else:
        winner = "tie"

    return {
        "query": query,
        "winner": winner,
        "pairwise": pairwise,
        "judge_a": judge_a,
        "judge_b": judge_b,
        "semantic_similarity": round(semantic_sim, 4),
        "labels_swapped": pairwise.get("labels_swapped", False),
    }


# ---------------------------------------------------------------------------
# Full A/B evaluation across a set of queries
# ---------------------------------------------------------------------------

def run_ab_comparison(
    queries: List[str],
    answers_a: List[str],
    answers_b: List[str],
    expected_answers: Optional[List[str]] = None,
    retrieved_chunks_list: Optional[List[List[str]]] = None,
    labels: Optional[Dict[str, str]] = None,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Run a full A/B comparison across multiple queries.

    Args:
        queries: List of query strings.
        answers_a: List of answers from version A.
        answers_b: List of answers from version B.
        expected_answers: Optional list of expected answers.
        retrieved_chunks_list: Optional list of retrieved chunk lists.
        labels: Dict with 'a' and 'b' labels describing each version.
            E.g., {"a": "prompt:v2.1", "b": "prompt:v2.0"}
        n_bootstrap: Number of bootstrap iterations for significance testing.
        random_seed: Random seed for reproducibility.

    Returns:
        Dict with per-pair results, aggregate stats, winner determination,
        and statistical significance.
    """
    if not (len(queries) == len(answers_a) == len(answers_b)):
        raise ValueError("queries, answers_a, and answers_b must have the same length")

    if expected_answers is None:
        expected_answers = [""] * len(queries)
    if retrieved_chunks_list is None:
        retrieved_chunks_list = [[] for _ in queries]

    if labels is None:
        labels = {"a": "Version A", "b": "Version B"}

    random.seed(random_seed)
    np.random.seed(random_seed)

    # Evaluate each pair
    pairs = []
    a_wins = 0
    b_wins = 0
    ties = 0
    a_scores = []
    b_scores = []

    for i, (query, ans_a, ans_b, exp_ans) in enumerate(
        zip(queries, answers_a, answers_b, expected_answers)
    ):
        chunks = retrieved_chunks_list[i] if i < len(retrieved_chunks_list) else []
        result = compare_single_pair(
            query=query,
            answer_a=ans_a,
            answer_b=ans_b,
            expected_answer=exp_ans,
            retrieved_chunks=chunks,
        )
        pairs.append(result)
        a_scores.append(result["judge_a"]["overall_score"])
        b_scores.append(result["judge_b"]["overall_score"])

        if result["winner"] == "A":
            a_wins += 1
        elif result["winner"] == "B":
            b_wins += 1
        else:
            ties += 1

    # Aggregate
    n = len(queries)
    a_mean = round(float(np.mean(a_scores)), 4) if a_scores else 0.0
    b_mean = round(float(np.mean(b_scores)), 4) if b_scores else 0.0
    a_std = round(float(np.std(a_scores)), 4) if len(a_scores) > 1 else 0.0
    b_std = round(float(np.std(b_scores)), 4) if len(b_scores) > 1 else 0.0

    # Semantic stability between A and B runs
    stability = compute_answer_stability(answers_a, answers_b, queries=queries)

    # Compute win rates
    a_win_rate = round(a_wins / n, 4) if n > 0 else 0.0
    b_win_rate = round(b_wins / n, 4) if n > 0 else 0.0
    tie_rate = round(ties / n, 4) if n > 0 else 0.0

    # Statistical significance via paired bootstrap
    significance = _bootstrap_significance(
        a_scores, b_scores, n_iterations=n_bootstrap, random_seed=random_seed
    )

    # Determine overall winner
    if a_wins > b_wins and significance.get("p_value", 1.0) < 0.05:
        overall_winner = "A"
        winner_label = labels["a"]
    elif b_wins > a_wins and significance.get("p_value", 1.0) < 0.05:
        overall_winner = "B"
        winner_label = labels["b"]
    else:
        overall_winner = "tie"
        winner_label = (
            f"{labels['a']} vs {labels['b']} (no significant difference)"
        )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "labels": labels,
        "n_queries": n,
        "overall_winner": overall_winner,
        "winner_label": winner_label,
        "results": {
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
            "a_win_rate": a_win_rate,
            "b_win_rate": b_win_rate,
            "tie_rate": tie_rate,
        },
        "scores": {
            "a_mean": a_mean,
            "b_mean": b_mean,
            "a_std": a_std,
            "b_std": b_std,
            "a_scores": a_scores,
            "b_scores": b_scores,
        },
        "stability": stability,
        "significance": significance,
        "per_pair": pairs,
    }


# ---------------------------------------------------------------------------
# Paired bootstrap significance testing
# ---------------------------------------------------------------------------

def _bootstrap_significance(
    scores_a: List[float],
    scores_b: List[float],
    n_iterations: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Compute statistical significance using paired bootstrap.

    Tests the null hypothesis that there is no difference between A and B.
    The p-value is the proportion of bootstrap iterations where the observed
    direction of the difference is reversed.

    Args:
        scores_a: List of scores for version A.
        scores_b: List of scores for version B.
        n_iterations: Number of bootstrap resamples.
        random_seed: Random seed.

    Returns:
        Dict with p_value, mean_difference, ci_lower, ci_upper, significant.
    """
    if len(scores_a) != len(scores_b) or len(scores_a) == 0:
        return {"p_value": 1.0, "mean_difference": 0.0, "significant": False}

    n = len(scores_a)
    observed_diff = float(np.mean(scores_a) - np.mean(scores_b))

    rng = np.random.RandomState(random_seed)
    bootstrap_diffs = []

    for _ in range(n_iterations):
        indices = rng.randint(0, n, size=n)
        diff = float(
            np.mean([scores_a[i] for i in indices])
            - np.mean([scores_b[i] for i in indices])
        )
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Two-sided p-value: proportion of bootstrap diffs with sign opposite to observed
    if observed_diff > 0:
        p_value = float(np.mean(bootstrap_diffs <= 0))
    elif observed_diff < 0:
        p_value = float(np.mean(bootstrap_diffs >= 0))
    else:
        p_value = 1.0

    # 95% confidence interval
    ci_lower = float(np.percentile(bootstrap_diffs, 2.5))
    ci_upper = float(np.percentile(bootstrap_diffs, 97.5))

    return {
        "p_value": round(p_value, 4),
        "mean_difference": round(observed_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "significant": p_value < 0.05,
        "n_iterations": n_iterations,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for running A/B comparisons."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run A/B comparison between two answer sets."
    )
    parser.add_argument(
        "--answers-a", "-a",
        required=True,
        help="Path to JSON file with answers A (list of strings).",
    )
    parser.add_argument(
        "--answers-b", "-b",
        required=True,
        help="Path to JSON file with answers B (list of strings).",
    )
    parser.add_argument(
        "--queries", "-q",
        required=True,
        help="Path to JSON file with queries (list of strings).",
    )
    parser.add_argument(
        "--expected", "-e",
        default=None,
        help="Path to JSON file with expected answers (list of strings).",
    )
    parser.add_argument(
        "--label-a",
        default="Version A",
        help="Label for version A.",
    )
    parser.add_argument(
        "--label-b",
        default="Version B",
        help="Label for version B.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations (default: 1000).",
    )

    args = parser.parse_args()

    def load_list(path: str) -> list:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "items" in data:
            return data["items"]
        raise ValueError(f"File {path} does not contain a list")

    queries = load_list(args.queries)
    answers_a = load_list(args.answers_a)
    answers_b = load_list(args.answers_b)
    expected = load_list(args.expected) if args.expected else None

    if len(queries) != len(answers_a) or len(queries) != len(answers_b):
        print("Error: query/answer list length mismatch")
        return

    if expected and len(expected) != len(queries):
        print("Error: expected answers length mismatch")
        return

    result = run_ab_comparison(
        queries=queries,
        answers_a=answers_a,
        answers_b=answers_b,
        expected_answers=expected,
        labels={"a": args.label_a, "b": args.label_b},
        n_bootstrap=args.bootstrap,
    )

    output = json.dumps(result, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()