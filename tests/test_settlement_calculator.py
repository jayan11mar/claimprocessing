import pytest

from app.tools.settlement_calculator import calculate_settlement


def test_settlement_existing_claim():
    res = calculate_settlement("C1001")
    assert res.claim_id == "C1001"
    assert round(res.gross_amount, 2) == 1200.0
    assert round(res.deductible, 2) == 500.0
    assert round(res.depreciation_amount, 2) == 70.0
    assert round(res.copay_amount, 2) == 63.0
    assert round(res.approved_amount, 2) == 567.0


def test_settlement_missing_claim():
    with pytest.raises(ValueError):
        calculate_settlement("CNOEXIST")


def test_settlement_invalid_depreciation_percent_defaults_to_zero():
    from app.models.domain import Claim, save_claim

    claim = Claim(
        claim_id="C7001",
        policy_number="P123456",
        claim_amount=6000.0,
        extra_info={"depreciation_percent": "not-a-number"},
    )
    save_claim(claim)

    res = calculate_settlement("C7001")
    assert res.depreciation_amount == 0.0
    assert res.copay_amount == round((6000.0 - 500.0) * 0.10, 2)


def test_settlement_applies_sub_limit_category():
    from app.models.domain import Claim, save_claim

    claim = Claim(
        claim_id="C7002",
        policy_number="P123456",
        claim_amount=9000.0,
        extra_info={"sub_limit_category": "hospital", "depreciation_percent": "0"},
    )
    save_claim(claim)

    res = calculate_settlement("C7002")
    assert res.sub_limit_applied == 5000.0
    assert res.approved_amount == 5000.0
    assert any("sub-limit" in note.lower() for note in res.notes)


def test_settlement_policy_missing_raises_error():
    from app.models.domain import Claim, save_claim

    claim = Claim(
        claim_id="C7003",
        policy_number="P000000",
        claim_amount=1000.0,
        extra_info={},
    )
    save_claim(claim)

    with pytest.raises(ValueError, match="Policy P000000 not found"):
        calculate_settlement("C7003")
