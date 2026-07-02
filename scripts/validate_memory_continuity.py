#!/usr/bin/env python3
"""
Validate memory continuity for multi-turn conversations.

This script simulates a realistic 3-turn conversation and verifies that
entities (policy numbers, claim IDs, incident dates, claim amounts) can
flow between turns without re-entry.

Run:
    python scripts/validate_memory_continuity.py

Expected output:
    PASS: All continuity checks passed.
    (or FAIL details)
"""

import os
import sys
import uuid

os.environ.setdefault("OPENAI_API_KEY", "test")

from app.chains.agent_chain import AgentChain
from app.memory.sqlite_memory import SQLiteMemory
from app.models.faq import FAQIntent, FAQResponse


# ---------------------------------------------------------------------------
# Controlled fake FAQ chain that returns deterministic intents per turn
# ---------------------------------------------------------------------------

class ContinuityFakeFAQChain:
    """Returns predictable intents for multi-turn validation without LLM."""

    def __init__(self):
        self.turn = 0

    def invoke(self, session_id, user_message, persist_history=True):
        self.turn += 1
        msg_lower = user_message.lower()

        if "status" in msg_lower or self.turn == 1:
            return FAQResponse(
                intent=FAQIntent.POLICY_STATUS,
                category="policy",
                confidence=0.95,
                answer_text="",
                reasoning="Turn 1: policy status",
                metadata={},
            )
        if "register" in msg_lower or self.turn == 2:
            return FAQResponse(
                intent=FAQIntent.CLAIM_REGISTRATION,
                category="claims",
                confidence=0.9,
                answer_text="",
                reasoning="Turn 2: claim registration",
                metadata={},
            )
        return FAQResponse(
            intent=FAQIntent.FRAUD_CHECK,
            category="fraud",
            confidence=0.85,
            answer_text="",
            reasoning="Turn 3: fraud check",
            metadata={},
        )


# ---------------------------------------------------------------------------
# Continuity Checks
# ---------------------------------------------------------------------------

def check_entity_flow_through_turns() -> int:
    """Simulate 3 turns and verify entities flow between them."""
    session_id = f"validate-continuity-{uuid.uuid4()}"
    agent = AgentChain(memory=SQLiteMemory())
    agent.faq_chain = ContinuityFakeFAQChain()

    # ---- Turn 1: Provide policy number ----
    r1 = agent.invoke(session_id, "What is the status of policy P123456?")
    assert r1.intent == FAQIntent.POLICY_STATUS, (
        f"Turn 1 expected POLICY_STATUS, got {r1.intent}"
    )
    assert r1.metadata.get("error") not in ("policy_number_missing",), (
        "Turn 1 should not fail with policy_number_missing"
    )

    # ---- Turn 2: Register a claim ----
    r2 = agent.invoke(session_id, "Register a claim for 1000")
    assert r2.intent == FAQIntent.CLAIM_REGISTRATION, (
        f"Turn 2 expected CLAIM_REGISTRATION, got {r2.intent}"
    )
    # Policy number should flow from turn 1 context
    assert r2.metadata.get("error") != "policy_number_missing", (
        "Turn 2 should reuse policy number from turn 1 history"
    )
    # Claim amount should be extracted from message
    assert r2.metadata.get("error") != "claim_amount_missing", (
        "Turn 2 claim amount should be extracted from message"
    )

    # ---- Turn 3: Fraud check ----
    r3 = agent.invoke(session_id, "Check fraud score")
    assert r3.intent == FAQIntent.FRAUD_CHECK, (
        f"Turn 3 expected FRAUD_CHECK, got {r3.intent}"
    )
    # Claim ID should be discoverable from turn 2's answer in history
    # (the assistant response from turn 2 will contain the claim ID)

    print(f"  Turn 1 (policy_status): intent={r1.intent}")
    print(f"  Turn 2 (claim_registration): intent={r2.intent}, error={r2.metadata.get('error', 'none')}")
    print(f"  Turn 3 (fraud_check): intent={r3.intent}")
    print(f"  Session history message count: {agent.memory.get_message_count(session_id)}")

    return 0


def check_session_context_extraction() -> int:
    """Verify _load_session_context extracts all expected entities."""
    agent = AgentChain(memory=SQLiteMemory())
    session_id = f"validate-extraction-{uuid.uuid4()}"

    # Simulate a conversation
    agent.memory.append_message(session_id, "user", "Check policy P999999")
    agent.memory.append_message(session_id, "assistant", "Policy P999999 is ACTIVE. Coverage: 10000.")
    agent.memory.append_message(
        session_id, "user",
        "Register a claim for 7500 with incident date 2024-08-20"
    )
    agent.memory.append_message(
        session_id, "assistant",
        "Claim registration completed. Claim ID: CVALIDATE01. Eligible: True."
    )

    context = agent._load_session_context(session_id)

    checks = {
        "policy_number": context.get("policy_number") == "P999999",
        "claim_id": context.get("claim_id") == "CVALIDATE01",
        "incident_date": "2024-08-20" in str(context.get("incident_date", "")),
        "claim_amount": context.get("claim_amount") == 7500.0,
    }

    all_pass = True
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name}: {status} (value={context.get(name, '<missing>')})")

    return 0 if all_pass else 1


def check_latest_value_wins() -> int:
    """Verify that the most recent entity value is used when duplicates exist."""
    agent = AgentChain(memory=SQLiteMemory())
    session_id = f"validate-latest-{uuid.uuid4()}"

    # Simulate a conversation where policy number changes
    agent.memory.append_message(session_id, "user", "Check policy P111111")
    agent.memory.append_message(session_id, "assistant", "Policy P111111 active.")
    agent.memory.append_message(session_id, "user", "Now check policy P222222")
    agent.memory.append_message(session_id, "assistant", "Policy P222222 active.")

    context = agent._load_session_context(session_id)

    latest = context.get("policy_number")
    if latest == "P222222":
        print(f"  latest_policy_number: PASS (P222222)")
        return 0
    else:
        print(f"  latest_policy_number: FAIL (expected P222222, got {latest})")
        return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("Memory Continuity Validation")
    print("=" * 60)
    exit_code = 0

    print("\n[1/3] Entity flow through 3 turns...")
    try:
        check_entity_flow_through_turns()
        print("  Result: PASS")
    except AssertionError as e:
        print(f"  Result: FAIL - {e}")
        exit_code = 1

    print("\n[2/3] Session context extraction...")
    try:
        rc = check_session_context_extraction()
        print(f"  Result: {'PASS' if rc == 0 else 'FAIL'}")
        exit_code |= rc
    except AssertionError as e:
        print(f"  Result: FAIL - {e}")
        exit_code = 1

    print("\n[3/3] Latest value wins...")
    try:
        rc = check_latest_value_wins()
        print(f"  Result: {'PASS' if rc == 0 else 'FAIL'}")
        exit_code |= rc
    except AssertionError as e:
        print(f"  Result: FAIL - {e}")
        exit_code = 1

    print("\n" + "=" * 60)
    if exit_code == 0:
        print("PASS: All continuity checks passed.")
    else:
        print(f"FAIL: {exit_code} check(s) failed.")
    print("=" * 60)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())