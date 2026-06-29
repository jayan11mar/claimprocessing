from app.chains.agent_chain import AgentChain
from app.models.faq import FAQIntent, FAQResponse
from app.tools.claims_intake import register_and_validate_claim


def test_register_claim_success():
    res = register_and_validate_claim(policy_number="P123456", claim_amount=1200.0)
    assert res.claim_id.startswith("C")
    assert res.policy_number == "P123456"
    assert res.is_eligible is True
    assert round(res.approved_amount, 2) == 700.0


def test_register_claim_policy_not_found():
    res = register_and_validate_claim(policy_number="P999999", claim_amount=100.0)
    assert res.is_eligible is False
    assert res.approved_amount == 0.0
    assert "Policy not found" in res.validation_messages[0] or res.metadata.get("error") == "policy_not_found"


def test_register_claim_inactive_policy():
    res = register_and_validate_claim(policy_number="P654321", claim_amount=100.0)
    assert res.is_eligible is False
    assert any("Policy status" in m for m in res.validation_messages)


def test_register_claim_non_positive_amount():
    res = register_and_validate_claim(policy_number="P123456", claim_amount=0.0)
    assert res.is_eligible is False
    assert any("greater than zero" in m for m in res.validation_messages)


def test_register_claim_missing_amount_returns_error():
    res = register_and_validate_claim(policy_number="P123456", claim_amount=None)
    assert res.is_eligible is False
    assert res.metadata["error"] == "claim_amount_missing"
    assert any("required" in msg.lower() for msg in res.validation_messages)


def test_register_claim_with_incident_date_and_documents():
    res = register_and_validate_claim(
        policy_number="P123456",
        claim_amount=1500.0,
        extra_info={
            "incident_date": "2024-04-10",
            "supporting_documents": ["hospital bill", "discharge summary"],
            "sub_limit_category": "hospital",
        },
    )
    assert res.is_eligible is True
    assert round(res.approved_amount, 2) == 1000.0
    assert "hospital bill" in res.metadata.get("supporting_documents", [])


def test_register_claim_empty_policy_number_returns_error():
    res = register_and_validate_claim(policy_number="", claim_amount=100.0)
    assert res.is_eligible is False
    assert res.metadata["error"] == "policy_number_missing"
    assert any("required" in msg.lower() for msg in res.validation_messages)


def test_register_claim_policy_not_found_returns_error():
    res = register_and_validate_claim(policy_number="P999999", claim_amount=100.0)
    assert res.is_eligible is False
    assert res.metadata["error"] == "policy_not_found"
    assert any("policy not found" in msg.lower() for msg in res.validation_messages)


def test_register_claim_description_falls_back_to_supporting_documents():
    res = register_and_validate_claim(
        policy_number="P123456",
        claim_amount=1200.0,
        extra_info={
            "description": "Please find the attached invoice and medical report.",
        },
    )
    assert res.is_eligible is True
    assert "invoice" in res.metadata.get("supporting_documents", []) or "medical report" in res.metadata.get("supporting_documents", [])


def test_register_claim_exceeds_sum_insured_and_applies_sub_limit():
    res = register_and_validate_claim(
        policy_number="P123456",
        claim_amount=15000.0,
        extra_info={
            "incident_date": "2024-02-10",
            "supporting_documents": ["invoice"],
            "sub_limit_category": "hospital",
        },
    )
    assert res.is_eligible is True
    assert round(res.approved_amount, 2) == 5000.0
    assert any("capped" in msg.lower() for msg in res.validation_messages)
    assert any("sub-limit" in msg.lower() for msg in res.validation_messages)


def test_register_claim_below_deductible_reports_not_payable():
    res = register_and_validate_claim(
        policy_number="P123456",
        claim_amount=200.0,
        extra_info={"incident_date": "2024-02-15", "supporting_documents": ["invoice"]},
    )
    assert res.approved_amount == 0.0
    assert any("not payable" in msg.lower() for msg in res.validation_messages)


def test_register_claim_after_policy_expiry_is_not_eligible():
    res = register_and_validate_claim(
        policy_number="P654321",
        claim_amount=500.0,
        extra_info={"incident_date": "2024-03-15", "supporting_documents": ["invoice"]},
    )
    assert res.is_eligible is False
    assert any("after policy expiry" in msg.lower() for msg in res.validation_messages)


def test_register_claim_before_policy_start_adds_coverage_gap_warning():
    res = register_and_validate_claim(
        policy_number="P123456",
        claim_amount=1000.0,
        extra_info={"incident_date": "2023-12-25", "supporting_documents": ["invoice"]},
    )
    assert any("before policy inception" in msg.lower() for msg in res.validation_messages)


def test_agent_chain_requires_policy_number_for_claim_registration():
    agent = AgentChain()
    agent.memory.clear_history("session1")
    agent.faq_chain.invoke = lambda sid, msg, **kwargs: FAQResponse(
        intent=FAQIntent.CLAIM_REGISTRATION,
        category="claims",
        confidence=0.9,
        answer_text="",
        reasoning="",
        metadata={},
    )

    response = agent.invoke("session1", "Register a rental reimbursement claim for $300", context={})

    assert response.intent == FAQIntent.CLAIM_REGISTRATION
    assert "policy number" in response.answer_text.lower()
    assert response.metadata["tool"] == "claims_intake"
    assert response.metadata["error"] == "policy_number_missing"


def test_agent_chain_requires_claim_amount_for_registration():
    agent = AgentChain()
    agent.faq_chain.invoke = lambda sid, msg, **kwargs: FAQResponse(
        intent=FAQIntent.CLAIM_REGISTRATION,
        category="claims",
        confidence=0.9,
        answer_text="",
        reasoning="",
        metadata={"policy_number": "P123456"},
    )

    response = agent.invoke("session1", "Register a claim under policy P123456", context={})

    assert response.intent == FAQIntent.CLAIM_REGISTRATION
    assert "claim amount" in response.answer_text.lower()
    assert response.metadata["tool"] == "claims_intake"
    assert response.metadata["error"] == "claim_amount_missing"


def test_extract_policy_number_from_policy_phrase():
    agent = AgentChain()
    assert agent._extract_policy_number("Please register a claim under policy 123456") == "P123456"


def test_register_claim_empty_policy_number_returns_error():
    res = register_and_validate_claim(policy_number="", claim_amount=100.0)
    assert res.is_eligible is False
    assert res.metadata["error"] == "policy_number_missing"
    assert any("required" in msg.lower() for msg in res.validation_messages)


def test_register_claim_amount_exceeds_sum_insured_and_sub_limit_applies():
    res = register_and_validate_claim(
        policy_number="P123456",
        claim_amount=15000.0,
        extra_info={
            "incident_date": "2024-02-01",
            "supporting_documents": ["invoice"],
            "sub_limit_category": "hospital",
        },
    )
    assert res.is_eligible is True
    assert round(res.approved_amount, 2) == 5000.0
    assert any("capped" in msg.lower() for msg in res.validation_messages)
    assert any("sub-limit" in msg.lower() for msg in res.validation_messages)


def test_register_claim_below_deductible_is_not_payable():
    res = register_and_validate_claim(
        policy_number="P123456",
        claim_amount=200.0,
        extra_info={"incident_date": "2024-02-15", "supporting_documents": ["invoice"]},
    )
    assert res.is_eligible is False or res.approved_amount == 0.0
    assert any("not payable" in msg.lower() for msg in res.validation_messages)


def test_agent_chain_uses_policy_number_from_history_for_claim_registration():
    from app.memory.sqlite_memory import SQLiteMemory

    agent = AgentChain(memory=SQLiteMemory())
    session_id = "test-multiturn-session"

    # Simulate first turn: user provides policy number and claim amount
    agent.memory.append_message(session_id, "user", "Register a claim for policy P123456 with amount 1000")
    agent.memory.append_message(session_id, "assistant", "Sure, I can help with that.")

    # Mock FAQChain to return CLAIM_REGISTRATION intent without policy_number in metadata
    agent.faq_chain.invoke = lambda sid, msg, **kwargs: FAQResponse(
        intent=FAQIntent.CLAIM_REGISTRATION,
        category="claims",
        confidence=0.9,
        answer_text="",
        reasoning="",
        metadata={},
    )

    # Second turn: user only provides claim amount, no policy number
    response = agent.invoke(session_id, "claim amount is 1000$", context={})

    assert response.intent == FAQIntent.CLAIM_REGISTRATION
    assert response.metadata.get("error") != "policy_number_missing"
    assert "Claim ID" in response.answer_text or "claim" in response.answer_text.lower()
