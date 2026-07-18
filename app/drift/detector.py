"""Drift detection orchestrator.

Coordinates multiple drift detection strategies (prompt and embedding)
and aggregates results against configurable thresholds.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from app.config import get_settings
from app.drift.baseline import snapshot_baseline
from app.drift.embedding_drift import compute_embedding_drift
from app.drift.prompt_drift import compute_prompt_drift
from app.drift.alerts import raise_alerts

logger = logging.getLogger(__name__)


# ── Threshold definitions ──────────────────────────────────────────────────
# Each entry: (key, direction, default_threshold)
#   direction="below" → alert if value < threshold
#   direction="above" → alert if value > threshold

_THRESHOLD_SPEC: List[Dict[str, Any]] = [
    {"metric": "semantic_shift",      "direction": "below", "default": 0.85},
    {"metric": "kl_divergence",        "direction": "above", "default": 0.10},
    {"metric": "spearman_rank_corr",   "direction": "below", "default": 0.80},
    {"metric": "nn_stability",         "direction": "below", "default": 0.70},
]


def _load_thresholds(path: str) -> Dict[str, float]:
    """Load threshold values from a YAML file, falling back to defaults.

    Only the four drift-metric keys are extracted; miscellaneous keys
    (e.g. ``enabled``) are ignored.
    """
    raw: Dict[str, Any] = {}
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:
        logger.warning("Could not load thresholds from %s; using defaults.", path)

    thresholds: Dict[str, float] = {}
    for spec in _THRESHOLD_SPEC:
        key = spec["metric"]
        raw_val = raw.get(key)
        if raw_val is not None and isinstance(raw_val, (int, float)):
            thresholds[key] = float(raw_val)
        else:
            thresholds[key] = spec["default"]

    return thresholds


def _get_current_results(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run a baseline snapshot on *cases* and return only the results list.

    This is a simple helper that reuses :func:`~app.drift.baseline.snapshot_baseline`
    but discards the aggregate ``embedding_stats`` from the baseline file
    (we save the snapshot anyway for consistency).
    """
    baseline = snapshot_baseline(cases, out_path="reports/drift_baseline.json")
    return baseline.get("results", [])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_drift(
    cases: List[Dict[str, Any]],
    baseline_path: str = "reports/drift_baseline.json",
    thresholds_path: str = "config/drift_thresholds.yaml",
) -> Dict[str, Any]:
    """Run all drift checks and evaluate results against thresholds.

    Steps:
    1. Run the QA chain for each case (via :func:`snapshot_baseline`) to
       produce current results.
    2. Compute prompt-drift signals
       (:func:`~app.drift.prompt_drift.compute_prompt_drift`).
    3. Compute embedding-drift signals
       (:func:`~app.drift.embedding_drift.compute_embedding_drift`).
    4. Merge all signals into a single scores dict.
    5. Compare each score against its threshold (respecting direction:
       ``below`` / ``above``) and collect breaches.

    Args:
        cases: List of case dicts (each must have ``question``).
        baseline_path: Path to the persisted baseline JSON.
        thresholds_path: Path to the YAML thresholds file.

    Returns:
        A dict with keys:
            - ``scores`` (dict): the merged drift signals.
            - ``breaches`` (list[dict]): metrics that breached their threshold.
            - ``any_breach`` (bool): ``True`` if at least one breach occurred.
    """
    # 1. Run QA chain on current cases
    current_results = _get_current_results(cases)

    if not current_results:
        logger.warning("No current results produced; cannot run drift detection.")
        return {
            "scores": {"semantic_shift": 0.0, "kl_divergence": 0.0,
                       "spearman_rank_corr": 0.0, "nn_stability": 0.0},
            "breaches": [],
            "any_breach": False,
        }

    # 2. Prompt drift
    prompt_scores = compute_prompt_drift(current_results, baseline_path=baseline_path)

    # 3. Embedding drift
    embedding_scores = compute_embedding_drift(current_results, baseline_path=baseline_path)

    # 4. Merge scores
    scores: Dict[str, float] = {
        "semantic_shift": prompt_scores.get("semantic_shift", 0.0),
        "format_compliance": prompt_scores.get("format_compliance", 0.0),
        "citation_density_change": prompt_scores.get("citation_density_change", 0.0),
        "kl_divergence": embedding_scores.get("kl_divergence", 0.0),
        "spearman_rank_corr": embedding_scores.get("spearman_rank_corr", 0.0),
        "nn_stability": embedding_scores.get("nn_stability", 0.0),
    }

    # 5. Load thresholds and evaluate
    thresholds = _load_thresholds(thresholds_path)

    breaches: List[Dict[str, Any]] = []
    for spec in _THRESHOLD_SPEC:
        metric = spec["metric"]
        direction = spec["direction"]
        threshold = thresholds[metric]
        value = scores.get(metric, 0.0)

        is_breach = False
        if direction == "below":
            is_breach = value < threshold
        elif direction == "above":
            is_breach = value > threshold

        if is_breach:
            breaches.append({
                "metric": metric,
                "value": round(value, 6),
                "threshold": threshold,
                "direction": direction,
            })

    return {
        "scores": scores,
        "breaches": breaches,
        "any_breach": len(breaches) > 0,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_eval_cases() -> List[Dict[str, Any]]:
    """Load evaluation cases from ``eval/eval_set.json``.

    Returns a list of dicts with at least a ``question`` key.
    """
    eval_path = Path("eval/eval_set.json")
    if not eval_path.exists():
        logger.warning("eval/eval_set.json not found; using empty case list.")
        return []
    raw = json.loads(eval_path.read_text(encoding="utf-8"))
    items = raw.get("items", [])
    return [{"question": item["query"]} for item in items if item.get("query")]


def main() -> None:
    """CLI entry point for drift detection.

    Usage::

        python -m app.drift.detector --baseline   # snapshot baseline
        python -m app.drift.detector --compare     # compare & print results
    """
    # ── Guard: drift must be enabled ─────────────────────────────────────
    settings = get_settings()
    if not settings.ENABLE_DRIFT:
        print("drift disabled")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Drift detection CLI (cron-safe, read-only monitoring).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Snapshot a baseline from the golden/eval set.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run drift comparison against the existing baseline.",
    )
    parser.add_argument(
        "--baseline-path",
        default="reports/drift_baseline.json",
        help="Path to the baseline JSON file (default: reports/drift_baseline.json).",
    )
    parser.add_argument(
        "--thresholds-path",
        default="config/drift_thresholds.yaml",
        help="Path to the thresholds YAML file (default: config/drift_thresholds.yaml).",
    )

    args = parser.parse_args()

    if not args.baseline and not args.compare:
        parser.print_help()
        sys.exit(0)

    cases = _load_eval_cases()
    if not cases:
        print("No evaluation cases found; nothing to do.")
        sys.exit(0)

    if args.baseline:
        print(f"Taking baseline snapshot of {len(cases)} cases ...")
        snapshot_baseline(cases, out_path=args.baseline_path)
        print(f"Baseline written to {args.baseline_path}")
        sys.exit(0)

    if args.compare:
        result = run_drift(
            cases,
            baseline_path=args.baseline_path,
            thresholds_path=args.thresholds_path,
        )
        # Print scores as JSON (stdout — cron-safe, never breaks CI)
        print(json.dumps(result.get("scores", {}), indent=2))

        # Print alerts
        alerts = raise_alerts(result)
        if alerts:
            print("\nAlerts:")
            for alert in alerts:
                print(f"  - {alert}")
        else:
            print("\nNo drift alerts — all metrics within threshold.")

        # Always exit 0 (read-only monitoring)
        sys.exit(0)


if __name__ == "__main__":
    main()
