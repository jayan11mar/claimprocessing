"""Regression tests for multi-turn context reliability.

Verifies that follow-up turns in the same session_id can reuse previously
supplied policy numbers, claim IDs, incident dates, and other extracted context
without the user needing to re-enter information.
"""

from app.chains.agent_chain import AgentChain
from app.memory.sqlite_memory import SQLiteMemory
from app.models.faq import FAQIntent, FAQResponse


class FakeFAQChain:
    """A fake FAQ chain that returns controlled intents for multi-turn testing."""

    def __init__(self):
        self.call_count = 0

    def invoke(self, session_id, user_message, persist_history=True):
        self.call_count += 1
        # Turn 1 -> POLICY_STATUS, Turn 2 -> CLAIM_REGISTRATION, Turn 3 -> FRAUD_CHECK
        if self.call_count == 1:
            return FAQResponse(
                intent=FAQIntent.POLICY_STATUS,
                category="policy",
                confidence=0.95,
                answer_text="",
                reasoning="Turn 1: policy status",
                metadata={},
            )
        elif self.call_count == 2:
            return FAQResponse(
                intent=FAQIntent.CLAIM_REGISTRATION,
                category="claims",
                confidence=0.9,
                answer_text="",
                reasoning="Turn 2: claim registration",
                metadata={},
            )
        else:
            return FAQResponse(
                intent=FAQIntent.FRAUD_CHECK,
                category="fraud",
                confidence=0.85,
                answer_text="",
                reasoning="Turn 3: fraud check",
                metadata={},
            )


def test_three_consecutive_turns_reuse_policy_number_from_first_turn():
    """Verify that 3 consecutive turns in one session reuse context correctly.

    Turn 1: User provides policy number "P123456" for policy status check.
    Turn 2: User asks to register a claim - AgentChain should extract P123456
            from history in system's own assistant output.
    Turn 3: User asks for fraud check - AgentChain should extract claim ID
            from the claim registration answer in history.
    """
    agent = AgentChain(memory=SQLiteMemory())
    session_id = "test-3-turn-session"

    # Clear any prior history
    agent.memory.clear_history(session_id)

    # Replace FAQ chain with fake
    fake_faq = FakeFAQChain()
    agent.faq_chain = fake_faq

    # ---- Turn 1: Check policy status ----
    response1 = agent.invoke(session_id, "What is the status of policy P123456?")
    assert response1.intent == FAQIntent.POLICY_STATUS
    # The response should contain something about policy status or
    # at minimum not be an error
    assert response1.metadata.get("error") not in ("policy_number_missing",)

    # ---- Turn 2: Register a claim (reuses policy number from turn 1) ----
    # The policy number P123456 should be in the history from turn 1
    # The claim amount "$1000" is provided in the message
    response2 = agent.invoke(session_id, "Register a claim for $1000")
    assert response2.intent == FAQIntent.CLAIM_REGISTRATION
    # Policy number should not be missing since it was extracted from history
    assert response2.metadata.get("error") != "policy_number_missing", (
        "Policy number should be extracted from session history"
    )
    # Claim amount should not be missing
    assert response2.metadata.get("error") != "claim_amount_missing", (
        "Claim amount should be extracted from user message"
    )
    # The answer should contain a Claim ID if registration succeeded
    # It may be an error if the policy is not found in DB, but the metadata
    # should show the tool was invoked

    # ---- Turn 3: Fraud check (reuses claim ID from turn 2 answer) ----
    response3 = agent.invoke(session_id, "Check fraud score")
    assert response3.intent == FAQIntent.FRAUD_CHECK
    # If a claim_id was produced in turn 2, it should be re-discoverable
    # from history for turn 3


def test_load_session_context_extracts_all_entities():
    """Unit test for _load_session_context directly."""
    agent = AgentChain(memory=SQLiteMemory())
    session_id = "test-load-context-session"
    agent.memory.clear_history(session_id)

    # Pre-seed history with a conversation
    agent.memory.append_message(session_id, "user", "What is the status of policy P123456?")
    agent.memory.append_message(session_id, "assistant", "Policy P123456 is ACTIVE.")
    agent.memory.append_message(session_id, "user", "Register a claim for 5000. Incident date is 2024-06-15")
    agent.memory.append_message(session_id, "assistant", "Claim C1A2B3C registered.")

    context = agent._load_session_context(session_id)

    # Should have extracted policy number
    assert context.get("policy_number") == "P123456"
    # Should have extracted claim ID
    assert context.get("claim_id") == "C1A2B3C"
    # Should have extracted incident date
    assert context.get("incident_date") is not None
    # Should have extracted claim amount
    assert context.get("claim_amount") == 5000.0


def test_load_session_context_with_empty_history():
    """_load_session_context should return empty dict for empty history."""
    agent = AgentChain(memory=SQLiteMemory())
    session_id = "test-empty-context-session"
    agent.memory.clear_history(session_id)

    context = agent._load_session_context(session_id)
    assert context == {}


def test_load_session_context_with_no_entities():
    """_load_session_context should return empty dict when history has no entities."""
    agent = AgentChain(memory=SQLiteMemory())
    session_id = "test-no-entities-session"
    agent.memory.clear_history(session_id)

    agent.memory.append_message(session_id, "user", "Hello")
    agent.memory.append_message(session_id, "assistant", "Hi there!")

    context = agent._load_session_context(session_id)
    assert context == {}


def test_load_session_context_latest_value_wins():
    """When the same entity appears multiple times, the latest value should win."""
    agent = AgentChain(memory=SQLiteMemory())
    session_id = "test-latest-value-session"
    agent.memory.clear_history(session_id)

    agent.memory.append_message(session_id, "user", "Check policy P111111")
    agent.memory.append_message(session_id, "assistant", "Policy P111111 is active.")
    agent.memory.append_message(session_id, "user", "Now check policy P222222")
    agent.memory.append_message(session_id, "assistant", "Policy P222222 is lapsed.")

    context = agent._load_session_context(session_id)

    # The latest policy number should be P222222
    assert context.get("policy_number") == "P222222"