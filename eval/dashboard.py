"""Data-preparation helpers for the Streamlit evaluation dashboard.

Provides pure functions that accept regression result data (as returned by
``/eval/regression``) and normalise metric rows into a trend-friendly
structure suitable for plotting and tabular display.

No FastAPI calls, no Streamlit imports — this is a pure data helper.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Metric extractors — each returns a flat dict safe for tabular/trend display
# ---------------------------------------------------------------------------

def extract_pass_rate(
    regression_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract the overall pass rate from a regression report.

    Args:
        regression_data: The full dict returned by ``/eval/regression``
            (or ``run_regression()``).

    Returns:
        Dict with keys: ``pass_rate``, ``total_cases``, ``passed_cases``,
        ``failed_cases``, ``all_metrics_passed``.
    """
    summary = regression_data.get("summary", {})
    custom = regression_data.get("custom_metrics", {})

    return {
        "pass_rate": _safe_float(summary.get("pass_rate")),
        "total_cases": _safe_int(summary.get("total_cases")),
        "passed_cases": _safe_int(summary.get("passed_cases")),
        "failed_cases": _safe_int(summary.get("failed_cases")),
        "all_metrics_passed": bool(custom.get("all_metrics_passed", False)),
    }


def extract_golden_set_pass_rate(
    regression_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract the golden-set pass-rate sub-metric.

    Args:
        regression_data: The full regression report dict.

    Returns:
        Dict with keys: ``golden_set_pass_rate``, ``golden_passed``,
        ``golden_total``.
    """
    custom = regression_data.get("custom_metrics", {})
    golden = custom.get("golden_set_pass_rate", {})

    return {
        "golden_set_pass_rate": _safe_float(golden.get("pass_rate")),
        "golden_passed": _safe_int(golden.get("passed_count")),
        "golden_total": _safe_int(golden.get("total_count")),
    }


def extract_answer_stability(
    regression_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract the answer-stability sub-metric.

    Args:
        regression_data: The full regression report dict.

    Returns:
        Dict with keys: ``answer_stability``, ``stability_min``,
        ``stability_max``, ``stability_std``, ``stability_pairs``.
    """
    custom = regression_data.get("custom_metrics", {})
    stability = custom.get("answer_stability", {})

    return {
        "answer_stability": _safe_float(stability.get("stability_score")),
        "stability_min": _safe_float(stability.get("min_similarity")),
        "stability_max": _safe_float(stability.get("max_similarity")),
        "stability_std": _safe_float(stability.get("std_similarity")),
        "stability_pairs": _safe_int(len(stability.get("per_pair", []))),
    }


def extract_regulatory_compliance(
    regression_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract the regulatory-compliance sub-metric.

    Args:
        regression_data: The full regression report dict.

    Returns:
        Dict with keys: ``regulatory_compliance``, ``regulatory_cases``,
        ``regulatory_patterns_matched``.
    """
    custom = regression_data.get("custom_metrics", {})
    compliance = custom.get("regulatory_compliance", {})

    return {
        "regulatory_compliance": _safe_float(compliance.get("compliance_score")),
        "regulatory_cases": _safe_int(len(compliance.get("per_case", []))),
        "regulatory_patterns_matched": _safe_int(
            len(compliance.get("matched_patterns", {}))
        ),
    }


def extract_role_appropriateness(
    regression_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract the role-appropriateness sub-metric.

    Args:
        regression_data: The full regression report dict.

    Returns:
        Dict with keys: ``role_appropriateness``, ``role_violations``,
        ``role_cases``.
    """
    custom = regression_data.get("custom_metrics", {})
    appropriateness = custom.get("role_appropriateness", {})

    return {
        "role_appropriateness": _safe_float(
            appropriateness.get("appropriateness_score")
        ),
        "role_violations": _safe_int(appropriateness.get("total_violations")),
        "role_cases": _safe_int(len(appropriateness.get("per_case", []))),
    }


def extract_hitl_trigger_precision(
    regression_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract the HITL-trigger-precision sub-metric.

    Args:
        regression_data: The full regression report dict.

    Returns:
        Dict with keys: ``hitl_trigger_precision``, ``hitl_true_positives``,
        ``hitl_false_positives``, ``hitl_total_triggers``.
    """
    custom = regression_data.get("custom_metrics", {})
    hitl = custom.get("hitl_trigger_precision", {})

    return {
        "hitl_trigger_precision": _safe_float(hitl.get("precision")),
        "hitl_true_positives": _safe_int(hitl.get("true_positives")),
        "hitl_false_positives": _safe_int(hitl.get("false_positives")),
        "hitl_total_triggers": _safe_int(hitl.get("total_triggers")),
    }


# ---------------------------------------------------------------------------
# Composite extractors
# ---------------------------------------------------------------------------

def extract_all_metrics(
    regression_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract all six top-level metrics into a single flat dict.

    This is the primary entry point for building a trend row.

    Args:
        regression_data: The full regression report dict.

    Returns:
        A single flat dict with all metric keys prefixed for clarity.
        Missing or ``None`` values are left as ``None`` (not defaulted to 0)
        so the caller can distinguish "not evaluated" from "zero".
    """
    row: Dict[str, Any] = {}

    # Pass rate (from summary)
    row.update(extract_pass_rate(regression_data))

    # Custom sub-metrics
    row.update(extract_golden_set_pass_rate(regression_data))
    row.update(extract_answer_stability(regression_data))
    row.update(extract_regulatory_compliance(regression_data))
    row.update(extract_role_appropriateness(regression_data))
    row.update(extract_hitl_trigger_precision(regression_data))

    # Timestamp from the report
    summary = regression_data.get("summary", {})
    row["timestamp"] = summary.get("timestamp", "")

    return row


def build_trend_rows(
    regression_reports: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build a list of trend-friendly rows from multiple regression reports.

    Each report becomes one row via ``extract_all_metrics()``.  This is
    designed to be called with a list of historical regression results so
    the frontend can plot metrics over time.

    Args:
        regression_reports: List of regression report dicts, each as
            returned by ``/eval/regression`` or ``run_regression()``.

    Returns:
        List of flat dicts, one per report, suitable for ``st.dataframe``
        or ``st.line_chart``.
    """
    return [extract_all_metrics(r) for r in regression_reports]


# ---------------------------------------------------------------------------
# Trend data preparation for Streamlit charts
# ---------------------------------------------------------------------------

_SUPPORTED_METRICS = frozenset({
    "pass_rate",
    "golden_set_pass_rate",
    "answer_stability",
    "regulatory_compliance",
    "role_appropriateness",
    "hitl_trigger_precision",
})


def prepare_trend_data(
    regression_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Transform a list of regression results into a Streamlit-chart-ready structure.

    The returned dict contains:
    - ``metrics`` — a ``list[dict]`` where each dict has only the six supported
      metric keys plus a ``timestamp``.  Missing values are set to ``None``.
    - ``available_metrics`` — a ``list[str]`` of metric names that are present
      (non-None) in at least one row.

    This structure can be consumed directly by the Streamlit frontend:

    .. code-block:: python

        data = prepare_trend_data(results)
        df = pd.DataFrame(data["metrics"])
        st.line_chart(df.set_index("timestamp"))

    Args:
        regression_results: List of regression report dicts, each as returned
            by ``run_regression()`` or the ``/eval/regression`` endpoint.

    Returns:
        Dict with ``metrics`` (list of flat dicts) and ``available_metrics``
        (list of metric names with data).
    """
    rows: List[Dict[str, Any]] = []

    for report in regression_results:
        if not isinstance(report, dict):
            continue

        row: Dict[str, Any] = {}

        # Capture timestamp from the summary (or the top-level report)
        summary = report.get("summary", {})
        row["timestamp"] = _safe_timestamp(summary) or _safe_timestamp(report) or ""

        # Extract the six supported top-level metrics
        row["pass_rate"] = _safe_float(summary.get("pass_rate"))

        custom = report.get("custom_metrics", {})

        golden = custom.get("golden_set_pass_rate", {})
        row["golden_set_pass_rate"] = _safe_float(golden.get("pass_rate"))

        stability = custom.get("answer_stability", {})
        row["answer_stability"] = _safe_float(stability.get("stability_score"))

        compliance = custom.get("regulatory_compliance", {})
        row["regulatory_compliance"] = _safe_float(compliance.get("compliance_score"))

        appropriateness = custom.get("role_appropriateness", {})
        row["role_appropriateness"] = _safe_float(
            appropriateness.get("appropriateness_score")
        )

        hitl = custom.get("hitl_trigger_precision", {})
        row["hitl_trigger_precision"] = _safe_float(hitl.get("precision"))

        rows.append(row)

    # Compute which metrics actually have at least one non-None value
    available_metrics: List[str] = []
    for metric in sorted(_SUPPORTED_METRICS):
        if any(r.get(metric) is not None for r in rows):
            available_metrics.append(metric)

    return {
        "metrics": rows,
        "available_metrics": available_metrics,
    }


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def extract_comparison_deltas(
    regression_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract before/after comparison deltas from a regression report.

    Args:
        regression_data: The full regression report dict (must contain a
            ``comparison`` key from baseline comparison).

    Returns:
        Dict with delta keys, or an empty dict if no comparison exists.
    """
    comparison = regression_data.get("comparison")
    if not comparison or not isinstance(comparison, dict):
        return {}

    return {
        "pass_rate_delta": _safe_float(comparison.get("pass_rate_delta")),
        "baseline_pass_rate": _safe_float(comparison.get("baseline_pass_rate")),
        "current_pass_rate": _safe_float(comparison.get("current_pass_rate")),
        "regression_count": _safe_int(comparison.get("regression_count")),
        "improvement_count": _safe_int(comparison.get("improvement_count")),
        "new_cases": _safe_int(len(comparison.get("new_cases", []))),
        "removed_cases": _safe_int(len(comparison.get("removed_cases", []))),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert *value* to float, returning *default* on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert *value* to int, returning *default* on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_timestamp(data: Dict[str, Any]) -> Optional[str]:
    """Extract a timestamp string from *data*, returning ``None`` if absent."""
    ts = data.get("timestamp")
    if isinstance(ts, str) and ts.strip():
        return ts.strip()
    return None