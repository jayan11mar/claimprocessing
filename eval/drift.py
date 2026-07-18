"""Pure functions for drift detection and Population Stability Index (PSI)."""

import json
import math
import os
from typing import Any, Optional


def compute_distribution(scores: list[float], bins: int = 10) -> list[float]:
    """Compute a normalised histogram of *scores* using *bins* equal-width bins.

    Returns a list of probabilities that sum to 1.0 (within floating-point
    precision).  If all scores are identical, all probability mass is placed
    in a single bin (the first one).
    """
    if not scores:
        return [0.0] * bins

    lo = min(scores)
    hi = max(scores)

    # Edge case: zero or single-valued range
    if hi - lo < 1e-12:
        result = [0.0] * bins
        result[0] = 1.0
        return result

    bin_width = (hi - lo) / bins
    counts = [0] * bins
    for s in scores:
        idx = int((s - lo) / bin_width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1

    total = sum(counts)
    return [c / total for c in counts]


def psi(expected: list[float], actual: list[float]) -> float:
    """Population Stability Index between two discrete distributions.

    Both *expected* and *actual* should be lists of probabilities (summing to
    1.0).  A small epsilon (``1e-6``) is added to zero-valued bins to keep
    the logarithm finite.
    """
    if len(expected) != len(actual):
        raise ValueError(
            f"expected length {len(expected)} != actual length {len(actual)}"
        )

    eps = 1e-6
    psi_val = 0.0
    for e, a in zip(expected, actual):
        e_safe = e + eps
        a_safe = a + eps
        psi_val += (a_safe - e_safe) * math.log(a_safe / e_safe)

    return psi_val


def drift_report(
    baseline: dict[str, float],
    current: dict[str, float],
    thresholds: Optional[dict[str, float]] = None,
) -> dict[str, dict[str, Any]]:
    """Compare per-metric baseline vs. current and flag drift.

    For each metric present in *baseline* and *current* the report contains::

        {metric: {"baseline": float, "current": float,
                  "delta": float, "drifted": bool}}

    A metric is considered **drifted** when ``abs(delta) > threshold`` **or**
    when the Population Stability Index (computed from the distribution of
    scores across all metrics) exceeds 0.2.

    Parameters
    ----------
    baseline
        Baseline metric values, e.g. ``{"pass_rate": 0.95, ...}``.
    current
        Current metric values for the same keys.
    thresholds
        Per-metric drift thresholds (default 0.05 for any metric not present
        in the dict, or when *thresholds* is ``None``).

    Returns
    -------
    dict
        Drift report keyed by metric name.
    """
    if thresholds is None:
        thresholds = {}

    report: dict[str, dict[str, Any]] = {}
    for metric in baseline:
        if metric not in current:
            continue

        b = baseline[metric]
        c = current[metric]
        delta = c - b

        threshold = thresholds.get(metric, 0.05)

        # Build a 2-bin distribution for PSI over the metric values
        dist_expected = compute_distribution([b], bins=2)
        dist_actual = compute_distribution([c], bins=2)
        psi_val = psi(dist_expected, dist_actual)

        drifted = (abs(delta) > threshold) or (psi_val > 0.2)

        report[metric] = {
            "baseline": b,
            "current": c,
            "delta": delta,
            "drifted": drifted,
        }

    return report


def _extract_metrics(report: dict) -> dict[str, float]:
    """Extract the drift-relevant metrics from a regression report dict.

    Returns a flat dict containing ``pass_rate`` (from the top-level
    ``summary.pass_rate``) plus the four ``custom_metrics_summary`` keys
    required by the drift specification::

        pass_rate, answer_stability, regulatory_compliance,
        hitl_trigger_precision

    Handles both the nested format (``{"summary": {"pass_rate": ...}}``) and
    the flat format (``{"pass_rate": ..., "custom_metrics_summary": ...}``)
    used by ``_baseline_summary.json``.
    """
    # Try nested format first, then fall back to flat top-level keys
    if "summary" in report:
        src = report["summary"]
        custom = src.get("custom_metrics_summary", {})
    else:
        src = report
        custom = report.get("custom_metrics_summary", {})

    metrics: dict[str, float] = {}

    pass_rate = src.get("pass_rate")
    if pass_rate is not None:
        metrics["pass_rate"] = float(pass_rate)

    for key in ("answer_stability", "regulatory_compliance", "hitl_trigger_precision"):
        val = custom.get(key)
        if val is not None:
            metrics[key] = float(val)

    return metrics


def load_and_compare(
    baseline_path: str,
    current_path: str,
    thresholds: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Load two regression report JSONs and return a per-metric drift report.

    Reads the JSON files at *baseline_path* and *current_path*, extracts the
    drift-relevant metrics (``pass_rate``, ``answer_stability``,
    ``regulatory_compliance``, ``hitl_trigger_precision``), and delegates to
    :func:`drift_report`.

    Parameters
    ----------
    baseline_path
        Path to the baseline regression report JSON.
    current_path
        Path to the current regression report JSON.
    thresholds
        Optional per-metric drift thresholds passed through to
        :func:`drift_report`.

    Returns
    -------
    dict
        If both files exist and are valid, returns the drift report dict.
        If a file is missing or unreadable, returns ``{"error": <message>}``.

    Examples
    --------
    >>> import json
    >>> result = load_and_compare(
    ...     "reports/regression_report.json",
    ...     "reports/regression_report.json",
    ... )
    >>> all(not r["drifted"] for r in result.values())
    True
    """
    for path, label in ((baseline_path, "baseline"), (current_path, "current")):
        if not os.path.isfile(path):
            return {"error": f"{label} file not found: {path}"}

    try:
        with open(baseline_path) as fh:
            baseline_data: dict = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return {"error": f"cannot read baseline file: {exc}"}

    try:
        current_data: dict = json.load(open(current_path))
    except (json.JSONDecodeError, OSError) as exc:
        return {"error": f"cannot read current file: {exc}"}

    baseline_metrics = _extract_metrics(baseline_data)
    current_metrics = _extract_metrics(current_data)

    # Ensure both sides have the same metric keys for drift_report
    common_keys = baseline_metrics.keys() & current_metrics.keys()
    baseline_filtered = {k: baseline_metrics[k] for k in common_keys}
    current_filtered = {k: current_metrics[k] for k in common_keys}

    return drift_report(baseline_filtered, current_filtered, thresholds)
