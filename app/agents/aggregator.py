"""Aggregator — merges child-agent results into a single coherent final answer.

Conflict resolution rules:
- If any agent reports a fraud flag (confidence >= 0.7), settlement
  recommendations are overridden with a fraud warning.
- Coverage denials take precedence over settlement calculations.
- Shared context (e.g. claim_id, policy_number) is passed through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from app.agents.dispatcher import AgentResult


@dataclass(frozen=True)
class FinalAnswer:
    """Coalesced response produced by the aggregator."""

    text: str
    agent_contributions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    shared_context: dict = field(default_factory=dict)


def aggregate(results: list[AgentResult]) -> FinalAnswer:
    """Merge *results* from child agents into a single :class:`FinalAnswer`.

    Conflict resolution
    -------------------
    1. **Fraud overrides settlement** — if any agent with "Fraud" in its
       name reports confidence >= 0.7, settlement answers are replaced
       with a fraud warning.
    2. **Coverage denial blocks settlement** — if a CoverageCheckAgent
       answer contains "denied" or "not covered", settlement is flagged.
    3. **Warnings** are collected for any conflicts detected.
    """
    if not results:
        return FinalAnswer(text="No sub-tasks were dispatched.")

    # ── Phase 1: classify results ──────────────────────────────────────────
    fraud_flagged = False
    coverage_denied = False
    coverage_answer = ""
    fraud_answer = ""
    settlement_answer = ""
    contributions: list[str] = []
    warnings: list[str] = []

    for r in results:
        contributions.append(r.agent_name)

        if "Fraud" in r.agent_name:
            fraud_answer = r.answer
            if r.confidence >= 0.7:
                fraud_flagged = True

        elif "Coverage" in r.agent_name:
            coverage_answer = r.answer
            lower = r.answer.lower()
            if "denied" in lower or "not covered" in lower:
                coverage_denied = True

        elif "Settlement" in r.agent_name:
            settlement_answer = r.answer

    # ── Phase 2: resolve conflicts ─────────────────────────────────────────
    if fraud_flagged:
        warnings.append("Fraud alert — settlement recommendation overridden.")
        settlement_answer = (
            "Settlement cannot be processed due to a fraud alert. "
            "Please escalate for manual review."
        )

    if coverage_denied:
        warnings.append("Coverage denied — settlement cannot proceed.")
        settlement_answer = (
            "Settlement cannot be calculated because the claim is not covered."
        )

    # ── Phase 3: build final text ──────────────────────────────────────────
    parts: list[str] = []

    if coverage_answer:
        parts.append(f"**Coverage Assessment:** {coverage_answer}")

    if fraud_answer:
        parts.append(f"**Fraud Screening:** {fraud_answer}")

    if settlement_answer:
        parts.append(f"**Settlement Recommendation:** {settlement_answer}")

    if warnings:
        parts.append("**Warnings:** " + " | ".join(warnings))

    final_text = "\n\n".join(parts) if parts else "No results to aggregate."

    return FinalAnswer(
        text=final_text,
        agent_contributions=contributions,
        warnings=warnings,
        shared_context={
            "fraud_flagged": fraud_flagged,
            "coverage_denied": coverage_denied,
        },
    )