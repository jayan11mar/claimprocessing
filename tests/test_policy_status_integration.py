from app.chains.agent_chain import AgentChain
from app.models.faq import FAQIntent, FAQResponse
from app.memory.sqlite_memory import SQLiteMemory


def test_agent_chain_handles_policy_status_query():
    """Test that the agent chain properly handles policy status queries."""
    agent = AgentChain(memory=SQLiteMemory())
    session_id = "test-policy-status-session"
    
    # Mock FAQChain to return POLICY_STATUS intent
    agent.faq_chain.invoke = lambda sid, msg, persist_history=True: FAQResponse(
        intent=FAQIntent.POLICY_STATUS,
        category="policy",
        confidence=0.9,
        answer_text="",
        reasoning="",
        metadata={"policy_number": "P654321"},
    )

    response = agent.invoke(session_id, "Can you verify if policy P654321 is active?", context={})

    assert response.intent == FAQIntent.POLICY_STATUS
    assert "LAPSED" in response.answer_text
    assert "cannot be filed" in response.answer_text.lower()
    assert response.metadata["tool"] == "policy_checker"


def test_agent_chain_requires_policy_number_for_status_check():
    """Test that the agent chain requires a policy number for status checks."""
    agent = AgentChain(memory=SQLiteMemory())
    session_id = "test-policy-status-no-policy"
    
    # Mock FAQChain to return POLICY_STATUS intent without policy_number
    agent.faq_chain.invoke = lambda sid, msg, persist_history=True: FAQResponse(
        intent=FAQIntent.POLICY_STATUS,
        category="policy",
        confidence=0.9,
        answer_text="",
        reasoning="",
        metadata={},
    )

    response = agent.invoke(session_id, "Is my policy active?", context={})

    assert response.intent == FAQIntent.POLICY_STATUS
    assert "policy number" in response.answer_text.lower()
    assert response.metadata["tool"] == "policy_checker"
    assert response.metadata["error"] == "policy_number_missing"


def test_agent_chain_extracts_policy_number_from_message():
    """Test that the agent chain extracts policy number from the message."""
    agent = AgentChain(memory=SQLiteMemory())
    session_id = "test-policy-status-extract"
    
    # Mock FAQChain to return POLICY_STATUS intent without policy_number in metadata
    agent.faq_chain.invoke = lambda sid, msg, persist_history=True: FAQResponse(
        intent=FAQIntent.POLICY_STATUS,
        category="policy",
        confidence=0.9,
        answer_text="",
        reasoning="",
        metadata={},
    )

    response = agent.invoke(session_id, "Can you verify if policy P654321 is active?", context={})

    assert response.intent == FAQIntent.POLICY_STATUS
    assert "LAPSED" in response.answer_text
    assert response.metadata["tool"] == "policy_checker"


def test_agent_chain_handles_fraud_check_query():
    """Test that the agent chain properly handles fraud check requests."""
    agent = AgentChain(memory=SQLiteMemory())
    session_id = "test-fraud-check-session"

    # Mock FAQChain to return FRAUD_CHECK intent with a known claim ID.
    agent.faq_chain.invoke = lambda sid, msg, persist_history=True: FAQResponse(
        intent=FAQIntent.FRAUD_CHECK,
        category="fraud",
        confidence=0.9,
        answer_text="",
        reasoning="",
        metadata={"claim_id": "C1001"},
    )

    response = agent.invoke(session_id, "Please run a fraud check for claim C1001.", context={})

    assert response.intent == FAQIntent.FRAUD_CHECK
    assert "Fraud score for claim C1001" in response.answer_text
    assert response.metadata["tool"] == "fraud_detector"
    assert response.metadata["tool_output"]["claim_id"] == "C1001"


def test_agent_chain_requires_claim_id_for_fraud_check():
    """Test that the agent chain asks for a claim ID when fraud check metadata is missing."""
    agent = AgentChain(memory=SQLiteMemory())
    session_id = "test-fraud-check-no-claim-id"

    agent.faq_chain.invoke = lambda sid, msg, persist_history=True: FAQResponse(
        intent=FAQIntent.FRAUD_CHECK,
        category="fraud",
        confidence=0.9,
        answer_text="",
        reasoning="",
        metadata={},
    )

    response = agent.invoke(session_id, "Please check the fraud score.", context={})

    assert response.intent == FAQIntent.FRAUD_CHECK
    assert "claim id" in response.answer_text.lower()
    assert response.metadata["tool"] == "fraud_detector"
    assert response.metadata["error"] == "claim_id_missing"