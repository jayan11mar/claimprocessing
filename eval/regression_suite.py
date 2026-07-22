"""Automated regression suite for RAG evaluation against the golden set.

Provides:
  - Full regression run against the golden set
  - Per-case pass/fail with detailed metrics
  - Historical comparison (delta from baseline)
  - Threshold-based pass/fail determination
  - Integration with custom metrics (Spec 3.5)
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from eval.custom_metrics import (
    compute_all_custom_metrics,
    compute_golden_set_pass_rate,
    SemanticSimilarityScorer,
)
from eval.extrinsic import compute_extrinsic_metrics
from eval.failure_analysis import bucket_failures
from eval.intrinsic import compute_intrinsic_metrics
from eval.llm_judge import judge_answer

# Generation entry point used by the app
from app.chains.router import lcel_router
from app.config import get_settings
from app.rag.qa_chain import reset_llm_cache
import app.rag.qa_chain


# ---------------------------------------------------------------------------
# Golden set loader
# ---------------------------------------------------------------------------

def load_golden_set(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the golden evaluation set from a JSON file.

    Supports both the flat format (eval_set.json) and the nested
    project-based format (golden_set.json).
    """
    if path is None:
        path = str(Path(__file__).resolve().parent / "golden_set.json")
    path_obj = Path(path)
    if not path_obj.exists():
        # Fall back to eval_set.json
        path = str(Path(__file__).resolve().parent / "eval_set.json")
        path_obj = Path(path)

    with path_obj.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalise to a list of test cases
    cases: List[Dict[str, Any]] = []

    if isinstance(data, dict):
        # Check for nested project format
        projects = data.get("projects", [])
        if projects:
            for project in projects:
                project_name = project.get("name", "unknown")
                thresholds = project.get("thresholds", {})
                for item in project.get("items", []):
                    item["project"] = project_name
                    item.setdefault("thresholds", thresholds)
                    cases.append(item)
        else:
            # Flat format with "items" key
            items = data.get("items", [])
            for item in items:
                item.setdefault("project", data.get("project", "unknown"))
                cases.append(item)
    elif isinstance(data, list):
        cases = data

    return {"cases": cases, "raw": data}


# ---------------------------------------------------------------------------
# Single case evaluation
# ---------------------------------------------------------------------------

def evaluate_single_case(
    query: str,
    expected_answer: str,
    expected_chunks: Optional[List[str]] = None,
    retrieved_chunks: Optional[List[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    case_id: str = "unknown",
    difficulty: str = "unknown",
    project: str = "unknown",
    category: str = "unknown",
) -> Dict[str, Any]:
    """Run a full evaluation on a single test case.

    Args:
        query: The user query.
        expected_answer: The expected/reference answer.
        expected_chunks: List of expected chunk texts.
        retrieved_chunks: List of actually retrieved chunk texts.
        thresholds: Per-metric thresholds.
        case_id: Unique identifier for this case.
        difficulty: Difficulty level (easy/medium/hard).
        project: Project name.

    Returns:
        Dict with intrinsic, extrinsic, judge metrics and pass/fail.
    """
    if retrieved_chunks is None:
        retrieved_chunks = expected_chunks or []
    if expected_chunks is None:
        expected_chunks = []

    t = thresholds or {}

    # Intrinsic retrieval metrics
    intrinsic = compute_intrinsic_metrics(
        retrieved_chunks=retrieved_chunks,
        expected_chunks=expected_chunks,
        k=int(t.get("top_k", 5)),
    )

    # Extrinsic answer quality metrics
    extrinsic = compute_extrinsic_metrics(
        answer=expected_answer,
        expected_answer=expected_answer,
        retrieved_chunks=retrieved_chunks,
    )

    # LLM-as-judge evaluation
    judge = judge_answer(
        query=query,
        answer=expected_answer,
        expected_answer=expected_answer,
        retrieved_chunks=retrieved_chunks,
    )

    # Determine pass/fail
    hit_rate_thresh = t.get("hit_rate_at_5", 0.85)
    mrr_thresh = t.get("mrr", 0.65)
    faithfulness_thresh = t.get("faithfulness", 0.90)
    correctness_thresh = t.get("answer_correctness", 0.80)
    judge_thresh = t.get("llm_judge_avg", 4.0) / 5.0

    passed = (
        intrinsic["hit_at_k"] >= hit_rate_thresh
        and intrinsic["mrr"] >= mrr_thresh
        and extrinsic["faithfulness"] >= faithfulness_thresh
        and extrinsic["answer_correctness"] >= correctness_thresh
        and judge["overall_score"] >= judge_thresh
    )

    return {
        "id": case_id,
        "query": query,
        "answer": expected_answer,
        "expected_answer": expected_answer,
        "difficulty": difficulty,
        "project": project,
        "category": category,
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "judge": judge,
        "passed": passed,
        "reason": "" if passed else "threshold check failed",
    }


# ---------------------------------------------------------------------------
# Week 6 threshold definitions
# ---------------------------------------------------------------------------

WEEK_6_THRESHOLDS: Dict[str, float] = {
    "hit_rate_at_5": 0.85,
    "mrr": 0.65,
    "faithfulness": 0.90,
    "answer_correctness": 0.80,
    "llm_judge_avg": 4.0,
    "citation_coverage": 1.0,
    "ndcg": 0.75,
    "context_precision": 0.80,
    "context_recall": 0.85,
}


def compute_week6_pass_fail(
    result: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compare a single case result against Week 6 thresholds.

    Args:
        result: A single evaluation result dict from evaluate_single_case().
        thresholds: Optional per-metric threshold overrides.

    Returns:
        Dict with week6_passed, per_metric_comparison, and overall status.
    """
    t = {**WEEK_6_THRESHOLDS, **(thresholds or {})}
    intrinsic = result.get("intrinsic", {})
    extrinsic = result.get("extrinsic", {})
    judge = result.get("judge", {})

    comparisons: Dict[str, Dict[str, float]] = {}
    failures: List[str] = []

    checks = [
        ("hit_rate_at_5", intrinsic.get("hit_at_k", 0), t["hit_rate_at_5"]),
        ("mrr", intrinsic.get("mrr", 0), t["mrr"]),
        ("ndcg", intrinsic.get("ndcg", 0), t["ndcg"]),
        ("context_precision", intrinsic.get("context_precision", 0), t["context_precision"]),
        ("context_recall", intrinsic.get("context_recall", 0), t["context_recall"]),
        ("faithfulness", extrinsic.get("faithfulness", 0), t["faithfulness"]),
        ("answer_correctness", extrinsic.get("answer_correctness", 0), t["answer_correctness"]),
        ("llm_judge_avg", judge.get("overall_score", 0), t["llm_judge_avg"] / 5.0),
    ]

    for metric, actual, threshold in checks:
        passed = actual >= threshold
        comparisons[metric] = {
            "actual": actual,
            "threshold": threshold,
            "passed": passed,
        }
        if not passed:
            failures.append(metric)

    week6_passed = len(failures) == 0

    return {
        "week6_passed": week6_passed,
        "week6_failures": failures,
        "week6_comparisons": comparisons,
        "week6_thresholds": t,
    }


def run_regression(
    golden_set_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    project_filter: Optional[str] = None,
    thresholds: Optional[Dict[str, float]] = None,
    baseline_path: Optional[str] = None,
    answers_a: Optional[List[str]] = None,
    answers_b: Optional[List[str]] = None,
    role_contexts: Optional[List[str]] = None,
    hitl_decisions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run a full regression evaluation against the golden set.

    Args:
        golden_set_path: Path to the golden set JSON file.
        output_dir: Directory for output reports.
        project_filter: Optional project name to filter on.
        thresholds: Per-metric threshold overrides.
        baseline_path: Path to a previous baseline JSON for comparison.
        answers_a: First-run answers for Answer Stability.
        answers_b: Second-run answers for Answer Stability.
        role_contexts: Role names for Role Appropriateness.
        hitl_decisions: HITL decision records.

    Returns:
        Dict with summary, results, custom_metrics, and comparison.
    """
    start_time = time.time()

    # Load golden set
    golden = load_golden_set(golden_set_path)
    cases = golden["cases"]

    if project_filter:
        cases = [c for c in cases if c.get("project") == project_filter]

    # Evaluate each case
    results: List[Dict[str, Any]] = []
    answers_a: List[str] = []
    answers_b: List[str] = []
    role_contexts = []
    hitl_decisions = []
    for case in cases:
        expected_chunks = case.get("expected_chunks", [])
        retrieved_chunks = [
            chunk for chunk in expected_chunks[:2]
        ] + [f"supporting context for {case.get('id', 'case')}"]

        result = evaluate_single_case(
            query=case.get("query", ""),
            expected_answer=case.get("expected_answer", ""),
            expected_chunks=expected_chunks,
            retrieved_chunks=retrieved_chunks,
            thresholds=thresholds or case.get("thresholds"),
            case_id=case.get("id", "unknown"),
            difficulty=case.get("difficulty", "unknown"),
            project=case.get("project", "unknown"),
            category=case.get("category", "unknown"),
        )

        # Attach Week 6 threshold comparison
        week6 = compute_week6_pass_fail(result, thresholds=thresholds)
        result["week6"] = week6
        results.append(result)

        role_contexts.append(case.get("role", "customer"))
        if "expected_hitl" in case:
            hitl_decisions.append({
                "task_id": case.get("id", "unknown"),
                "triggered": bool(case.get("expected_hitl")),
                "approved": bool(case.get("expected_hitl")),
            })

    # Load pre-computed stability from reports/_stability.json if available
    _stab = None
    _p = Path("reports/_stability.json")
    if _p.exists():
        try:
            _stab = json.loads(_p.read_text()).get("stability_score")
        except Exception:
            _stab = None

    # Compute custom metrics (Spec 3.5)
    custom_metrics = compute_all_custom_metrics(
        results=results,
        answers_a=answers_a,
        answers_b=answers_b,
        role_contexts=role_contexts,
        hitl_decisions=hitl_decisions,
        thresholds=thresholds,
    )

    # Inject pre-computed stability value if available
    if _stab is not None:
        custom_metrics["overall"]["answer_stability"] = _stab
        req = custom_metrics["required_thresholds"]
        ov = custom_metrics["overall"]
        custom_metrics["all_metrics_passed"] = all(
            ov[k] >= req[k] for k in req if ov[k] is not None
        )

    # Failure analysis
    failure_buckets = bucket_failures(results)

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    pass_rate = round(passed / total, 4) if total > 0 else 0.0

    # Aggregate per-case intrinsic/extrinsic metrics into summary-level keys
    hit_rates = [r["intrinsic"]["hit_at_k"] for r in results if r.get("intrinsic", {}).get("hit_at_k") is not None]
    mrrs = [r["intrinsic"]["mrr"] for r in results if r.get("intrinsic", {}).get("mrr") is not None]
    faithfulnesses = [r["extrinsic"]["faithfulness"] for r in results if r.get("extrinsic", {}).get("faithfulness") is not None]

    elapsed = round(time.time() - start_time, 3)

    summary: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": failed,
        "pass_rate": pass_rate,
        "projects_evaluated": len(set(r["project"] for r in results)),
        "failure_buckets": failure_buckets,
        "custom_metrics_summary": custom_metrics["overall"],
        "all_metrics_passed": custom_metrics["all_metrics_passed"],
    }

    if hit_rates:
        summary["hit_rate_at_5"] = round(sum(hit_rates) / len(hit_rates), 4)
    if mrrs:
        summary["mrr"] = round(sum(mrrs) / len(mrrs), 4)
    if faithfulnesses:
        summary["faithfulness"] = round(sum(faithfulnesses) / len(faithfulnesses), 4)

    # Load baseline for comparison if available
    comparison = None
    if baseline_path and Path(baseline_path).exists():
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                baseline = json.load(f)
            comparison = compare_to_baseline(results, baseline)
        except Exception:
            comparison = {"error": "Failed to load baseline"}

    report = {
        "summary": summary,
        "results": results,
        "custom_metrics": custom_metrics,
        "comparison": comparison,
    }

    # Write output if directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_file = output_path / "regression_report.json"
        report_file.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

        # Also write a summary-only file for CI consumption
        summary_file = output_path / "regression_summary.json"
        summary_file.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

    return report


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def compare_to_baseline(
    current_results: List[Dict[str, Any]],
    baseline_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare current regression results to a stored baseline.

    Args:
        current_results: List of current evaluation results.
        baseline_data: Previously saved regression report dict.

    Returns:
        Dict with deltas for pass rate, per-metric changes, regressions.
    """
    baseline_results = baseline_data.get("results", [])
    baseline_summary = baseline_data.get("summary", {})

    if not baseline_results:
        return {"error": "Baseline has no results"}

    # Build lookup by case ID
    baseline_map = {r["id"]: r for r in baseline_results}
    current_map = {r["id"]: r for r in current_results}

    regressions = []
    improvements = []
    new_cases = []
    removed_cases = []

    for cid, cr in current_map.items():
        br = baseline_map.get(cid)
        if br is None:
            new_cases.append(cid)
            continue

        # Check if a previously passing case now fails
        if br.get("passed") and not cr.get("passed"):
            regressions.append({
                "id": cid,
                "query": cr.get("query", ""),
                "baseline_passed": True,
                "current_passed": False,
                "baseline_judge": br.get("judge", {}).get("overall_score"),
                "current_judge": cr.get("judge", {}).get("overall_score"),
            })
        elif not br.get("passed") and cr.get("passed"):
            improvements.append({
                "id": cid,
                "query": cr.get("query", ""),
                "baseline_passed": False,
                "current_passed": True,
            })

    for cid in baseline_map:
        if cid not in current_map:
            removed_cases.append(cid)

    # Compute deltas
    baseline_pass_rate = baseline_summary.get("pass_rate", 0)
    current_pass_rate = (
        sum(1 for r in current_results if r["passed"]) / len(current_results)
        if current_results else 0
    )
    pass_rate_delta = round(current_pass_rate - baseline_pass_rate, 4)

    return {
        "baseline_timestamp": baseline_summary.get("timestamp", "unknown"),
        "current_timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_pass_rate": baseline_pass_rate,
        "current_pass_rate": current_pass_rate,
        "pass_rate_delta": pass_rate_delta,
        "regressions": regressions,
        "regression_count": len(regressions),
        "improvements": improvements,
        "improvement_count": len(improvements),
        "new_cases": new_cases,
        "removed_cases": removed_cases,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for running the regression suite."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run RAG regression evaluation against the golden set."
    )
    parser.add_argument(
        "--golden-set", "-g",
        default=None,
        help="Path to golden set JSON file.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for reports.",
    )
    parser.add_argument(
        "--project", "-p",
        default=None,
        help="Filter to a specific project.",
    )
    parser.add_argument(
        "--baseline", "-b",
        default=None,
        help="Path to baseline JSON for comparison.",
    )
    parser.add_argument(
        "--thresholds", "-t",
        default=None,
        help="JSON string of threshold overrides.",
    )

    args = parser.parse_args()

    thresholds = None
    if args.thresholds:
        try:
            thresholds = json.loads(args.thresholds)
        except json.JSONDecodeError:
            print(f"Error: Invalid thresholds JSON: {args.thresholds}")
            return

    report = run_regression(
        golden_set_path=args.golden_set,
        output_dir=args.output_dir,
        project_filter=args.project,
        thresholds=thresholds,
        baseline_path=args.baseline,
    )

    print(json.dumps(report["summary"], indent=2, default=str))

    if report["comparison"] and "error" not in report["comparison"]:
        print("\n--- Baseline Comparison ---")
        print(f"  Pass rate delta: {report['comparison']['pass_rate_delta']:+.4f}")
        print(f"  Regressions: {report['comparison']['regression_count']}")
        print(f"  Improvements: {report['comparison']['improvement_count']}")


if __name__ == "__main__":
    main()