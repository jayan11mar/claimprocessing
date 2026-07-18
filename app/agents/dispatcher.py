"""Dispatcher — routes sub-tasks to the appropriate child agent.

Each sub-task is resolved to an agent via the registry by matching the
sub-task's ``target_capability`` against each agent's ``capabilities``
list.  The agent is then invoked (stub for now) and an ``AgentResult``
is returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from app.agents.decomposer import SubTask
from app.agents.registry import AgentDescriptor, list_agents


@dataclass(frozen=True)
class AgentResult:
    """Result produced by a child agent."""

    agent_name: str
    answer: str
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0


# ── Capability → agent lookup (built once) ───────────────────────────────────

_CAP_TO_AGENT: dict[str, AgentDescriptor] | None = None


def _build_capability_map() -> dict[str, AgentDescriptor]:
    """Map every capability string to the first agent that declares it."""
    mapping: dict[str, AgentDescriptor] = {}
    for agent in list_agents():
        for cap in agent.capabilities:
            if cap not in mapping:  # first agent wins
                mapping[cap] = agent
    return mapping


def _get_capability_map() -> dict[str, AgentDescriptor]:
    global _CAP_TO_AGENT
    if _CAP_TO_AGENT is None:
        _CAP_TO_AGENT = _build_capability_map()
    return _CAP_TO_AGENT


# ── Stub agent invocation ────────────────────────────────────────────────────

def _invoke_agent(agent: AgentDescriptor, subtask: SubTask) -> AgentResult:
    """Stub invocation — returns a placeholder result.

    In a real implementation this would call the agent's prompt chain
    with the sub-task text and the agent's configured retrieval scope.
    """
    return AgentResult(
        agent_name=agent.name,
        answer=f"[{agent.name}] stub answer for: {subtask.text}",
        evidence=[f"retrieval_scope={agent.retrieval_scope}"],
        confidence=0.5,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def dispatch(subtasks: list[SubTask]) -> list[AgentResult]:
    """Route each *subtask* to the appropriate child agent.

    Returns one :class:`AgentResult` per sub-task.  Sub-tasks whose
    capability cannot be resolved are routed to a generic fallback.
    """
    cap_map = _get_capability_map()
    results: list[AgentResult] = []

    for subtask in subtasks:
        agent = cap_map.get(subtask.target_capability)
        if agent is None:
            # Fallback — no agent claims this capability
            results.append(
                AgentResult(
                    agent_name="FallbackAgent",
                    answer=subtask.text,
                    evidence=[],
                    confidence=0.0,
                )
            )
        else:
            results.append(_invoke_agent(agent, subtask))

    return results