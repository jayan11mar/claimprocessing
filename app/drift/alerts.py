"""Alerting for drift detection.

Generates human-readable alert messages when drift thresholds are breached,
and logs them at WARNING level.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ── Direction description map ──────────────────────────────────────────────

_DIRECTION_LABELS = {
    "below": "dropped below",
    "above": "exceeded",
}


def raise_alerts(result: Dict[str, Any]) -> List[str]:
    """Generate human-readable alert strings for breached drift metrics.

    Only metrics that appear in the ``breaches`` list produce an alert;
    metrics within threshold produce no alert (conservative behaviour).

    Each alert is logged at ``WARNING`` level and returned as a string
    in the format::

        "Drift alert: <metric> <direction-label> threshold: <value> (<threshold>)"

    Args:
        result: The drift result dict produced by
            :func:`~app.drift.detector.run_drift`.
            Must contain a ``breaches`` key (list of dicts).

    Returns:
        A list of human-readable alert strings (one per breached metric).
        Returns an empty list when no breaches are present.
    """
    breaches: List[Dict[str, Any]] = result.get("breaches", [])

    if not breaches:
        return []

    alerts: List[str] = []
    for b in breaches:
        metric = b.get("metric", "unknown")
        value = b.get("value", 0.0)
        threshold = b.get("threshold", 0.0)
        direction = b.get("direction", "below")

        label = _DIRECTION_LABELS.get(direction, direction)
        msg = (
            f"Drift alert: {metric} {label} threshold: "
            f"{value:.6f} ({threshold})"
        )
        logger.warning(msg)
        alerts.append(msg)

    return alerts