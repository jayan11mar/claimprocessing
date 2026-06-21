import io
import logging

from app.tools import fraud_detector as fraud_detector_module
from app.tools.fraud_detector import compute_fraud_score
from app.models.domain import Claim, save_claim
from datetime import date, datetime, timedelta


def test_fraud_score_existing_claim():
    res = compute_fraud_score("C1001")
    assert res.claim_id == "C1001"
    assert 0.0 <= res.score <= 1.0
    assert isinstance(res.signals, list)


def test_fraud_score_missing_claim():
    res = compute_fraud_score("CNOEXIST")
    assert res.score == 0.0
    assert "Claim not found." in res.signals


def test_fraud_score_with_recent_and_duplicate():
    incident_date = datetime.utcnow().date() - timedelta(days=5)
    c1 = Claim(claim_id="C2001", policy_number="P789012", claim_amount=100.0, incident_date=incident_date, extra_info={})
    c2 = Claim(claim_id="C2002", policy_number="P789012", claim_amount=100.0, incident_date=incident_date, extra_info={})
    save_claim(c1)
    save_claim(c2)

    res = compute_fraud_score("C2002")
    assert res.claim_id == "C2002"
    assert "Duplicate claim amounts detected" in res.signals
    assert any("recent" in s.lower() for s in res.signals)
    assert res.details["recent_claim_count"] == "1"
    assert res.score == 0.4


def test_fraud_score_across_policyholder_claims():
    incident_date1 = datetime.utcnow().date() - timedelta(days=10)
    incident_date2 = datetime.utcnow().date() - timedelta(days=5)
    c1 = Claim(
        claim_id="C3001",
        policy_number="P789012",
        policy_holder_id="H1001",
        claim_amount=120000.0,
        hospital_name="City Care Hospital",
        incident_date=incident_date1,
        extra_info={"sub_limit_category": "surgery"},
    )
    c2 = Claim(
        claim_id="C3002",
        policy_number="P123456",
        policy_holder_id="H1001",
        claim_amount=120000.0,
        hospital_name="City Care Hospital",
        incident_date=incident_date2,
        extra_info={"sub_limit_category": "hospital"},
    )
    save_claim(c1)
    save_claim(c2)

    res = compute_fraud_score("C3002")
    assert "Similar claim amount found for the same policyholder across policies" in res.signals
    assert "Same hospital filed multiple same-amount claims for this policyholder" in res.signals
    assert res.score >= 0.5


def test_fraud_score_policy_missing_records_policy_error():
    claim = Claim(
        claim_id="C4001",
        policy_number="P000000",
        policy_holder_id="H2001",
        claim_amount=1000.0,
        incident_date=datetime.utcnow().date() - timedelta(days=5),
        hospital_name="Unknown Hospital",
        extra_info={},
    )
    save_claim(claim)

    res = compute_fraud_score("C4001")
    assert res.score == 1.0
    assert "Policy record is missing." in res.signals
    assert res.details["policy_error"] == "policy_not_found"


def test_fraud_score_early_incident_and_high_cost_surgical_adds_signals():
    claim = Claim(
        claim_id="C4002",
        policy_number="P123456",
        policy_holder_id="H1001",
        claim_amount=60000.0,
        diagnosis_code="S99.1",
        hospital_name="General Hospital",
        incident_date=date.fromisoformat("2024-01-15"),
        extra_info={},
    )
    save_claim(claim)

    res = compute_fraud_score("C4002")
    assert any("early claim" in signal.lower() for signal in res.signals)
    assert any("high-cost surgical" in signal.lower() for signal in res.signals)
    assert res.score > 0.1


def test_fraud_score_logs_timestamps():
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(fraud_detector_module.logger.handlers[0].formatter)
    fraud_detector_module.logger.addHandler(handler)
    try:
        res = compute_fraud_score("C1001")
        text = buffer.getvalue()
    finally:
        fraud_detector_module.logger.removeHandler(handler)

    assert res.claim_id == "C1001"
    assert "fraud_detector_start" in text
    assert "fraud_detector_completed" in text
    assert "timestamp" in text
    assert "duration_ms" in text
