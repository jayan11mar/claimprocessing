from datetime import datetime, timedelta
from typing import List, Optional

from app.logging.json_logger import get_logger
from app.models.domain import (
    Claim,
    FraudScoreResult,
    get_claim,
    get_claims_for_policy,
    get_claims_for_policy_holder,
    get_policy,
)

logger = get_logger(__name__)


def compute_fraud_score(claim_id: Optional[str] = None, claim: Optional[Claim] = None) -> FraudScoreResult:
    start_time = datetime.utcnow()
    if claim is None:
        if claim_id is None:
            raise ValueError("Either claim_id or claim must be provided to compute fraud score.")
        claim = get_claim(claim_id)
    else:
        claim_id = claim.claim_id

    logger.info(
        "fraud_detector_start",
        {"claim_id": claim_id, "started_at": start_time.isoformat() + "Z"},
    )

    if claim is None:
        result = FraudScoreResult(
            claim_id=claim_id or "",
            score=0.0,
            signals=["Claim not found."],
            details={"error": "claim_not_found"},
        )
        completed_at = datetime.utcnow()
        logger.info(
            "fraud_detector_completed",
            {
                "claim_id": claim_id,
                "score": result.score,
                "signals": result.signals,
                "details": result.details,
                "started_at": start_time.isoformat() + "Z",
                "completed_at": completed_at.isoformat() + "Z",
                "duration_ms": int((completed_at - start_time).total_seconds() * 1000),
            },
        )
        return result

    policy = get_policy(claim.policy_number)
    signals: List[str] = []
    score = 0.1
    details = {"policy_number": claim.policy_number}

    if policy is None:
        signals.append("Policy record is missing.")
        details["policy_error"] = "policy_not_found"
        result = FraudScoreResult(claim_id=claim_id, score=1.0, signals=signals, details=details)
        completed_at = datetime.utcnow()
        logger.info(
            "fraud_detector_completed",
            {
                "claim_id": claim_id,
                "score": result.score,
                "signals": result.signals,
                "details": result.details,
                "started_at": start_time.isoformat() + "Z",
                "completed_at": completed_at.isoformat() + "Z",
                "duration_ms": int((completed_at - start_time).total_seconds() * 1000),
            },
        )
        return result

    if claim.incident_date and claim.incident_date < policy.start_date + timedelta(days=30):
        signals.append("Early claim after policy start date")
        score += 0.25

    if claim.incident_date and policy.end_date and claim.incident_date >= policy.end_date - timedelta(days=15):
        signals.append("Incident date occurs near policy expiry")
        score += 0.15

    thirty_days_ago = datetime.utcnow().date() - timedelta(days=30)
    recent_claims: List[Claim] = [
        existing
        for existing in get_claims_for_policy(claim.policy_number)
        if existing.claim_id != claim.claim_id
        and existing.incident_date
        and existing.incident_date >= thirty_days_ago
    ]

    if len(recent_claims) >= 2:
        signals.append("Multiple recent claims on the same policy")
        score += 0.25
    elif len(recent_claims) == 1:
        signals.append("One recent claim found on the same policy")
        score += 0.1

    duplicate_amounts = [
        c
        for c in get_claims_for_policy(claim.policy_number)
        if c.claim_id != claim.claim_id and c.claim_amount == claim.claim_amount
    ]
    if duplicate_amounts:
        signals.append("Duplicate claim amounts detected")
        score += 0.2

    if claim.claim_amount >= policy.sum_insured * 0.8:
        signals.append("Claim amount is close to the sum insured")
        score += 0.15

    if claim.policy_holder_id:
        related_claims = get_claims_for_policy_holder(claim.policy_holder_id)
        related_same_amount = [
            c for c in related_claims if c.claim_id != claim.claim_id and c.claim_amount == claim.claim_amount
        ]
        if related_same_amount:
            signals.append("Similar claim amount found for the same policyholder across policies")
            score += 0.2

        duplicated_hospital = [
            c for c in related_claims
            if c.claim_id != claim.claim_id and c.hospital_name == claim.hospital_name and c.claim_amount == claim.claim_amount
        ]
        if duplicated_hospital:
            signals.append("Same hospital filed multiple same-amount claims for this policyholder")
            score += 0.15

    if claim.diagnosis_code and claim.diagnosis_code.startswith("S") and claim.claim_amount > 50000:
        signals.append("High-cost surgical treatment claim requires additional review")
        score += 0.1

    score = min(1.0, round(score, 2))
    details["policy_status"] = policy.status.value
    details["recent_claim_count"] = str(len(recent_claims))
    details["status"] = policy.status.value
    details["policy_holder_id"] = claim.policy_holder_id or "unknown"

    result = FraudScoreResult(claim_id=claim_id, score=score, signals=signals, details=details)
    completed_at = datetime.utcnow()
    logger.info(
        "fraud_detector_completed",
        {
            "claim_id": claim_id,
            "score": result.score,
            "signals": result.signals,
            "details": result.details,
            "started_at": start_time.isoformat() + "Z",
            "completed_at": completed_at.isoformat() + "Z",
            "duration_ms": int((completed_at - start_time).total_seconds() * 1000),
        },
    )
    return result
