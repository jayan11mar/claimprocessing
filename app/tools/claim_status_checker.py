from typing import Optional

from app.models.domain import get_claim


class ClaimStatusResult:
    def __init__(
        self,
        claim_id: str,
        policy_number: str,
        status: Optional[str],
        claim_amount: float,
        approved_amount: Optional[float],
        fraud_score: Optional[float],
        settlement_status: Optional[str],
        incident_date: Optional[str],
        hospital_name: Optional[str],
        diagnosis_code: Optional[str],
        message: str,
    ):
        self.claim_id = claim_id
        self.policy_number = policy_number
        self.status = status
        self.claim_amount = claim_amount
        self.approved_amount = approved_amount
        self.fraud_score = fraud_score
        self.settlement_status = settlement_status
        self.incident_date = incident_date
        self.hospital_name = hospital_name
        self.diagnosis_code = diagnosis_code
        self.message = message

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "policy_number": self.policy_number,
            "status": self.status,
            "claim_amount": self.claim_amount,
            "approved_amount": self.approved_amount,
            "fraud_score": self.fraud_score,
            "settlement_status": self.settlement_status,
            "incident_date": self.incident_date,
            "hospital_name": self.hospital_name,
            "diagnosis_code": self.diagnosis_code,
            "message": self.message,
        }


def check_claim_status(claim_id: str) -> ClaimStatusResult:
    """
    Look up the status and details of an existing claim from the claims database.

    Args:
        claim_id: The claim ID to look up (e.g., C1001).

    Returns:
        ClaimStatusResult with the claim's current status and details.
    """
    if not claim_id or not claim_id.strip():
        return ClaimStatusResult(
            claim_id=claim_id or "",
            policy_number="",
            status=None,
            claim_amount=0.0,
            approved_amount=None,
            fraud_score=None,
            settlement_status=None,
            incident_date=None,
            hospital_name=None,
            diagnosis_code=None,
            message="Claim ID is required to check status.",
        )

    claim = get_claim(claim_id.strip().upper())

    if claim is None:
        return ClaimStatusResult(
            claim_id=claim_id,
            policy_number="",
            status=None,
            claim_amount=0.0,
            approved_amount=None,
            fraud_score=None,
            settlement_status=None,
            incident_date=None,
            hospital_name=None,
            diagnosis_code=None,
            message=f"Claim {claim_id} not found in the system. Please verify the claim ID.",
        )

    status = claim.status or "PENDING_REVIEW"
    fraud_score_str = f"{claim.fraud_score:.2f}" if claim.fraud_score is not None else "Not computed"
    approved_str = f"${claim.approved_amount:.2f}" if claim.approved_amount is not None else "Not yet calculated"

    message = (
        f"Claim {claim.claim_id}: Status = {status}. "
        f"Policy: {claim.policy_number}. "
        f"Claim Amount: ${claim.claim_amount:.2f}. "
        f"Approved Amount: {approved_str}. "
        f"Fraud Score: {fraud_score_str}. "
        f"Settlement Status: {claim.settlement_status or 'Not yet processed'}. "
        f"Incident Date: {claim.incident_date.isoformat() if claim.incident_date else 'Not recorded'}. "
        f"Hospital: {claim.hospital_name or 'Not recorded'}."
    )

    return ClaimStatusResult(
        claim_id=claim.claim_id,
        policy_number=claim.policy_number,
        status=status,
        claim_amount=claim.claim_amount,
        approved_amount=claim.approved_amount,
        fraud_score=claim.fraud_score,
        settlement_status=claim.settlement_status,
        incident_date=claim.incident_date.isoformat() if claim.incident_date else None,
        hospital_name=claim.hospital_name,
        diagnosis_code=claim.diagnosis_code,
        message=message,
    )