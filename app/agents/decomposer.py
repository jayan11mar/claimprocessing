"""Query decomposer — splits a composite claim query into sub-tasks.

Rule-based by default; falls back to LLM-based decomposition only when
ENABLE_MULTI_AGENT is True and an LLM is configured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.agents.registry import list_agents


@dataclass(frozen=True)
class SubTask:
    """A single sub-task produced by decomposing a user query."""

    target_capability: str
    text: str


# ── Keyword → capability mapping (rule-based) ────────────────────────────────

_CAPABILITY_KEYWORDS: dict[str, list[str]] = {
    "coverage_verification": [
        "cover", "coverage", "covered", "policy", "exclusion",
        "deductible", "limit", "within policy", "does the policy",
    ],
    "exclusion_analysis": [
        "exclusion", "excluded", "not covered", "exception",
    ],
    "deductible_validation": [
        "deductible", "out-of-pocket", "out of pocket",
    ],
    "fraud_detection": [
        "fraud", "fraudulent", "suspicious", "scam", "fake",
    ],
    "anomaly_scoring": [
        "anomaly", "unusual", "abnormal", "irregular",
    ],
    "entity_linking": [
        "link", "related", "connected", "pattern", "history",
    ],
    "settlement_calculation": [
        "settle", "settlement", "payout", "pay out", "amount",
        "compensation", "award", "calculate", "how much",
    ],
    "damage_estimation": [
        "damage", "estimate", "repair", "loss", "valuation",
    ],
    "negotiation_support": [
        "negotiate", "negotiation", "offer", "counter",
    ],
}


def _match_capabilities(query: str) -> list[str]:
    """Return a deduplicated list of capability names matched by keyword."""
    lower = query.lower()
    matched: list[str] = []
    seen: set[str] = set()
    for cap, keywords in _CAPABILITY_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", lower):
                if cap not in seen:
                    matched.append(cap)
                    seen.add(cap)
                break
    return matched


def decompose(query: str) -> list[SubTask]:
    """Split *query* into sub-tasks based on known agent capabilities.

    Uses a deterministic keyword-matching strategy.  Returns one
    :class:`SubTask` per matched capability.  If no capabilities are
    matched, a single sub-task with an empty ``target_capability`` is
    returned so the caller can fall through to the default single-agent
    path.
    """
    capabilities = _match_capabilities(query)

    if not capabilities:
        return [SubTask(target_capability="", text=query)]

    # Build a sub-task for each matched capability, keeping the full
    # original query as the text so downstream agents have full context.
    return [SubTask(target_capability=cap, text=query) for cap in capabilities]