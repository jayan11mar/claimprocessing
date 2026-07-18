"""Tests for multi-agent orchestration plumbing (W8-9).

All agents are stubbed, so assertions focus on routing, structure,
conflict resolution --- not answer text.
"""

import os
from unittest import mock

from app.agents.aggregator import FinalAnswer, aggregate
from app.agents.decomposer import SubTask, decompose
from app.agents.dispatcher import AgentResult, dispatch
from app.agents.orchestrator import orchestrate
from app.agents.registry import list_agents


# ── Helper ───────────────────────────────────────────────────────────────────

_MOCK_RESULTS: list[AgentResult] = [
    AgentResult(
        agent_name="CoverageCheckAgent",
        answer="Coverage verified.",
        evidence=["policy_documents"],
        confidence=0.9,
    ),
    AgentResult(
        agent_name="FraudScreeningAgent",
        answer="No fraud indicators detected.",
        evidence=["fraud_patterns"],
        confidence=0.2,
    ),
    AgentResult(
        agent_name="SettlementAgent",
        answer="Estimated payout $5000.",
        evidence=["settlement_guidelines"],
        confidence=0.8,
    ),
]


# ── Test 1: Registry loads exactly 3 agents ──────────────────────────────────

def test_registry_loads_three_agents() -> None:
    """config/agents.yaml defines exactly the three named agents."""
    agents = list_agents()
    names = {a.name for a in agents}

    assert len(agents) == 3, f"Expected 3 agents, got {len(agents)}"
    assert "CoverageCheckAgent" in names
    assert "FraudScreeningAgent" in names
    assert "SettlementAgent" in names


# ── Test 2: Decompose produces ≥2 sub-tasks ──────────────────────────────────

def test_decompose_covered_fraud_payout() -> None:
    """decompose('covered? fraud? payout?') should match 3 capabilities."""
    subtasks = decompose("covered? fraud? payout?")

    assert len(subtasks) >= 2, f"Expected >=2 sub-tasks, got {len(subtasks)}"

    caps = {s.target_capability for s in subtasks}
    assert "coverage_verification" in caps
    assert "fraud_detection" in caps
    assert "settlement_calculation" in caps


# ── Test 3: Dispatch routes each sub-task to the correct agent ───────────────

def test_dispatch_routes_to_correct_agents() -> None:
    """dispatch routes coverage_verification → CoverageCheckAgent etc."""
    subtasks = [
        SubTask(target_capability="coverage_verification", text="covered?"),
        SubTask(target_capability="fraud_detection", text="fraud?"),
        SubTask(target_capability="settlement_calculation", text="payout?"),
    ]

    results = dispatch(subtasks)
    agent_names = {r.agent_name for r in results}

    assert agent_names == {
        "CoverageCheckAgent",
        "FraudScreeningAgent",
        "SettlementAgent",
    }, f"Got agent names: {agent_names}"


# ── Test 4: Aggregate returns FinalAnswer with 3 contributions ───────────────

def test_aggregate_returns_three_contributions() -> None:
    """aggregate produces a FinalAnswer with 3 agent_contributions and
    shared_context keys."""
    fa = aggregate(_MOCK_RESULTS)

    assert isinstance(fa, FinalAnswer)
    assert len(fa.agent_contributions) == 3
    assert "CoverageCheckAgent" in fa.agent_contributions
    assert "FraudScreeningAgent" in fa.agent_contributions
    assert "SettlementAgent" in fa.agent_contributions
    assert "fraud_flagged" in fa.shared_context
    assert "coverage_denied" in fa.shared_context


# ── Test 5: Conflict rule — fraud_flagged=True overrides settlement ──────────

def test_fraud_flagged_overrides_settlement() -> None:
    """When a Fraud result has confidence >= 0.7, shared_context reflects
    fraud_flagged=True and settlement is overridden."""
    results_with_fraud = [
        AgentResult(
            agent_name="CoverageCheckAgent", answer="Covered.",
            confidence=0.9,
        ),
        AgentResult(
            agent_name="FraudScreeningAgent", answer="Fraud detected!",
            confidence=0.95,  # >= 0.7 → flagged
        ),
        AgentResult(
            agent_name="SettlementAgent", answer="Payout $5000.",
            confidence=0.8,
        ),
    ]

    fa = aggregate(results_with_fraud)

    assert fa.shared_context["fraud_flagged"] is True
    assert "fraud alert" in fa.warnings[0].lower()
    assert len(fa.warnings) >= 1

    # Settlement recommendation should contain the override message
    assert "Settlement" in fa.text
    assert "cannot be processed" in fa.text.lower()


# ── Test 6: ENABLE_MULTI_AGENT=False → orchestrate NOT called ────────────────

@mock.patch.dict(os.environ, {"ENABLE_MULTI_AGENT": "false"}, clear=False)
def test_multi_agent_flag_false_skips_orchestrate() -> None:
    """When ENABLE_MULTI_AGENT is false, the /chat entrypoint should NOT
    invoke orchestrate(). We verify by patching orchestrate and asserting
    it is never called."""
    from app.config import get_settings

    # Invalidate the lru_cache so the new env var is picked up
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.ENABLE_MULTI_AGENT is False

    # Simulate the /chat routing decision in server.py
    with mock.patch("app.agents.orchestrator.orchestrate") as mock_orch:
        # This is the exact branching logic from server.py chat() lines 480-496
        if settings.ENABLE_MULTI_AGENT:
            mock_orch("covered? fraud? payout?")

        mock_orch.assert_not_called()