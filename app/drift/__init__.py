"""Drift Detection Module for RAG Evaluation Monitoring.

Detects drift in RAG system performance by comparing current evaluation
results against historical baselines. Supports:

  - Metric drift: Detect significant changes in evaluation metrics
  - Distribution drift: Compare score distributions using KS test
  - Alert generation: Configurable thresholds for drift severity levels
  - /eval/drift API endpoint integration
"""

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _kolmogorov_smirnov_test(sample_a: List[float], sample_b: List[float]) -> float:
    """Two-sample Kolmogorov-Smirnov test statistic.

    Returns the KS statistic (D value) between 0 and 1.
    Higher values indicate greater distribution difference.
    """
    if not sample_a or not sample_b:
        return 1.0

    # Sort both samples
    a = sorted(sample_a)
    b = sorted(sample_b)

    # Compute ECDF difference
    n_a = len(a)
    n_b = len(b)
    all_values = sorted(set(a + b))

    d = 0.0
    for v in all_values:
        ecdf_a = sum(1 for x in a if x <= v) / n_a
        ecdf_b = sum(1 for x in b if x <= v) / n_b
        d = max(d, abs(ecdf_a - ecdf_b))

    return d


def _compute_psi(expected: List[float], actual: List[float], n_bins: int = 10) -> float:
    """Population Stability Index (PSI).

    Measures how much a distribution has shifted. PSI > 0.25 indicates
    significant shift, 0.1-0.25 indicates moderate shift, < 0.1 indicates
    no significant shift.
    """
    if not expected or not actual:
        return 1.0

    all_values = expected + actual
    min_val = min(all_values)
    max_val = max(all_values)

    if max_val == min_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    psi = 0.0
    for e_count, a_count in zip(expected_counts, actual_counts):
        e_prop = e_count / len(expected) if len(expected) > 0 else 0
        a_prop = a_count / len(actual) if len(actual) > 0 else 0

        # Avoid division by zero / log(0)
        e_prop = max(e_prop, 0.001)
        a_prop = max(a_prop, 0.001)

        psi += (a_prop - e_prop) * math.log(a_prop / e_prop)

    return psi


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def detect_metric_drift(
    current_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Detect drift in individual evaluation metrics.

    Compares current metric values against baseline and flags any that
    have changed beyond the configured threshold.

    Args:
        current_metrics: Dict of current metric values.
        baseline_metrics: Dict of baseline metric values.
        thresholds: Per-metric drift thresholds (default 10% relative change).

    Returns:
        Dict with drift_score, drifted_metrics, alerts.
    """
    if not baseline_metrics:
        return {
            "drift_score": 0.0,
            "drifted_metrics": [],
            "alerts": [],
            "message": "No baseline available for comparison",
        }

    default_threshold = 0.10  # 10% relative change
    t = thresholds or {}

    drifted = []
    alerts = []
    max_drift = 0.0

    all_keys = set(current_metrics.keys()) | set(baseline_metrics.keys())

    for key in all_keys:
        current = current_metrics.get(key)
        baseline = baseline_metrics.get(key)

        if current is None or baseline is None:
            continue

        if baseline == 0:
            rel_change = abs(current) if current != 0 else 0.0
        else:
            rel_change = abs(current - baseline) / abs(baseline)

        threshold = t.get(key, default_threshold)
        max_drift = max(max_drift, rel_change)

        if rel_change > threshold:
            direction = "increase" if current > baseline else "decrease"
            severity = "critical" if rel_change > 2 * threshold else "warning"

            drifted.append({
                "metric": key,
                "baseline": baseline,
                "current": current,
                "relative_change": round(rel_change, 4),
                "direction": direction,
                "severity": severity,
            })

            alerts.append({
                "metric": key,
                "severity": severity,
                "message": (
                    f"Metric '{key}' {direction}d by {rel_change:.1%} "
                    f"(baseline: {baseline:.4f}, current: {current:.4f})"
                ),
            })

    return {
        "drift_score": round(max_drift, 4),
        "drifted_metrics": drifted,
        "alerts": alerts,
        "drifted_count": len(drifted),
        "total_metrics": len(all_keys),
    }


def detect_distribution_drift(
    current_scores: List[float],
    baseline_scores: List[float],
    ks_threshold: float = 0.3,
    psi_threshold: float = 0.25,
) -> Dict[str, Any]:
    """Detect drift in score distributions using KS test and PSI.

    Args:
        current_scores: List of current evaluation scores.
        baseline_scores: List of baseline evaluation scores.
        ks_threshold: KS statistic threshold for alerting (default: 0.3).
        psi_threshold: PSI threshold for alerting (default: 0.25).

    Returns:
        Dict with ks_statistic, psi, alerts.
    """
    if not baseline_scores or not current_scores:
        return {
            "ks_statistic": 1.0,
            "psi": 1.0,
            "alerts": [{
                "metric": "distribution",
                "severity": "warning",
                "message": "Insufficient data for distribution comparison",
            }],
        }

    ks_stat = _kolmogorov_smirnov_test(current_scores, baseline_scores)
    psi = _compute_psi(baseline_scores, current_scores)

    alerts = []

    if ks_stat > ks_threshold:
        alerts.append({
            "metric": "ks_test",
            "severity": "critical" if ks_stat > 0.5 else "warning",
            "message": (
                f"KS statistic {ks_stat:.3f} exceeds threshold {ks_threshold}. "
                "Score distribution has shifted significantly."
            ),
        })

    if psi > psi_threshold:
        alerts.append({
            "metric": "psi",
            "severity": "critical" if psi > 0.5 else "warning",
            "message": (
                f"PSI {psi:.3f} exceeds threshold {psi_threshold}. "
                "Population distribution has shifted."
            ),
        })

    return {
        "ks_statistic": round(ks_stat, 4),
        "psi": round(psi, 4),
        "alerts": alerts,
    }


def run_drift_detection(
    current_report: Dict[str, Any],
    baseline_report: Optional[Dict[str, Any]] = None,
    baseline_path: Optional[str] = None,
    metric_thresholds: Optional[Dict[str, float]] = None,
    ks_threshold: float = 0.3,
    psi_threshold: float = 0.25,
) -> Dict[str, Any]:
    """Run full drift detection against a baseline.

    Args:
        current_report: Current evaluation report dict.
        baseline_report: Previous baseline report dict (optional).
        baseline_path: Path to baseline JSON file (alternative to baseline_report).
        metric_thresholds: Per-metric drift thresholds.
        ks_threshold: KS statistic threshold.
        psi_threshold: PSI threshold.

    Returns:
        Dict with metric_drift, distribution_drift, overall_drift_score, alerts.
    """
    # Load baseline from path if not provided
    if baseline_report is None and baseline_path:
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                baseline_report = json.load(f)
        except Exception:
            baseline_report = None

    if baseline_report is None:
        return {
            "overall_drift_score": 0.0,
            "metric_drift": {
                "drift_score": 0.0,
                "drifted_metrics": [],
                "alerts": [{
                    "metric": "baseline",
                    "severity": "info",
                    "message": "No baseline available. Run regression to establish baseline.",
                }],
                "drifted_count": 0,
                "total_metrics": 0,
            },
            "distribution_drift": None,
            "alerts": [],
            "has_baseline": False,
        }

    # Extract current and baseline summary metrics
    current_summary = current_report.get("summary", {})
    baseline_summary = baseline_report.get("summary", {})

    current_custom = current_report.get("custom_metrics", {}).get("overall", {})
    baseline_custom = baseline_report.get("custom_metrics", {}).get("overall", {})

    # Combine all metrics
    current_metrics = {
        "pass_rate": current_summary.get("pass_rate", 0),
        "total_cases": current_summary.get("total_cases", 0),
        "passed_cases": current_summary.get("passed_cases", 0),
        **current_custom,
    }
    baseline_metrics = {
        "pass_rate": baseline_summary.get("pass_rate", 0),
        "total_cases": baseline_summary.get("total_cases", 0),
        "passed_cases": baseline_summary.get("passed_cases", 0),
        **baseline_custom,
    }

    # Metric drift
    metric_drift = detect_metric_drift(
        current_metrics=current_metrics,
        baseline_metrics=baseline_metrics,
        thresholds=metric_thresholds,
    )

    # Distribution drift (using judge scores)
    current_scores = [
        r.get("judge", {}).get("overall_score", 0)
        for r in current_report.get("results", [])
    ]
    baseline_scores = [
        r.get("judge", {}).get("overall_score", 0)
        for r in baseline_report.get("results", [])
    ]

    distribution_drift = detect_distribution_drift(
        current_scores=current_scores,
        baseline_scores=baseline_scores,
        ks_threshold=ks_threshold,
        psi_threshold=psi_threshold,
    )

    # Overall drift score (max of metric and distribution drift)
    metric_drift_score = metric_drift.get("drift_score", 0)
    dist_ks = distribution_drift.get("ks_statistic", 0)
    overall_drift_score = round(max(metric_drift_score, dist_ks), 4)

    # Combine all alerts
    all_alerts = []
    all_alerts.extend(metric_drift.get("alerts", []))
    all_alerts.extend(distribution_drift.get("alerts", []))

    return {
        "overall_drift_score": overall_drift_score,
        "metric_drift": metric_drift,
        "distribution_drift": distribution_drift,
        "alerts": all_alerts,
        "alert_count": len(all_alerts),
        "has_baseline": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Drift report persistence
# ---------------------------------------------------------------------------

def save_drift_report(
    drift_result: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> str:
    """Save drift detection results to a JSON file.

    Args:
        drift_result: The drift detection result dict.
        output_dir: Directory to save to (default: reports/).

    Returns:
        Path to the saved file.
    """
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent.parent / "reports")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_path = output_path / f"drift_report_{timestamp}.json"

    file_path.write_text(json.dumps(drift_result, indent=2, default=str), encoding="utf-8")

    return str(file_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for running drift detection."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run drift detection on RAG evaluation results."
    )
    parser.add_argument(
        "--current", "-c",
        required=True,
        help="Path to current evaluation report JSON.",
    )
    parser.add_argument(
        "--baseline", "-b",
        required=True,
        help="Path to baseline evaluation report JSON.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for drift report JSON.",
    )
    parser.add_argument(
        "--ks-threshold",
        type=float,
        default=0.3,
        help="KS statistic threshold (default: 0.3).",
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=0.25,
        help="PSI threshold (default: 0.25).",
    )

    args = parser.parse_args()

    with open(args.current, "r", encoding="utf-8") as f:
        current = json.load(f)
    with open(args.baseline, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    result = run_drift_detection(
        current_report=current,
        baseline_report=baseline,
        ks_threshold=args.ks_threshold,
        psi_threshold=args.psi_threshold,
    )

    output = json.dumps(result, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Drift report written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()