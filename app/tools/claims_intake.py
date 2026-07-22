from datetime import datetime, date
from typing import Dict, List, Optional, Union
from uuid import uuid4

from app.models.domain import Claim, ClaimValidationResult, PolicyStatus, get_policy, save_claim
from app.tools.fraud_detector import compute_fraud_score

FRAUD_SCORE_THRESHOLD = 0.7


def _parse_incident_date(value: Union[str, date, None]) -> Optional[date]:
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        return None

    candidates = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d %b %Y",
        "%d %B %Y",
    ]
    for fmt in candidates:
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _parse_supporting_documents(value: Optional[Union[str, List[str]]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    separators = [",", ";", " and ", " & "]
    for sep in separators:
        if sep in text:
            parts = [part.strip() for part in text.split(sep) if part.strip()]
            if parts:
                return parts
    return [text]


def register_and_validate_claim(
    policy_number: str,
    claim_amount: Optional[float] = None,
    extra_info: Optional[Dict[str, Union[str, List[str]]]] = None,
    persist: bool = True,
) -> ClaimValidationResult:
    extra_info = extra_info or {}
    policy = get_policy(policy_number)
    validation_messages: List[str] = []
    approved_amount = 0.0
    is_eligible = True

    if not policy_number.strip():
        return ClaimValidationResult(
            claim_id="",
            policy_number=policy_number,
            is_eligible=False,
            approved_amount=0.0,
            validation_messages=["Policy number is required to register a claim."],
            metadata={"error": "policy_number_missing"},
        )

    if policy is None:
        return ClaimValidationResult(
            claim_id="",
            policy_number=policy_number,
            is_eligible=False,
            approved_amount=0.0,
            validation_messages=["Policy not found."],
            metadata={"error": "policy_not_found"},
        )

    if policy.status != PolicyStatus.ACTIVE:
        validation_messages.append(f"Policy status is {policy.status}. Claim cannot be processed.")
        is_eligible = False

    if claim_amount is None:
        return ClaimValidationResult(
            claim_id="",
            policy_number=policy_number,
            is_eligible=False,
            approved_amount=0.0,
            validation_messages=["Claim amount is required to process the claim."],
            metadata={"error": "claim_amount_missing"},
        )

    if claim_amount <= 0:
        validation_messages.append("Claim amount must be greater than zero.")
        is_eligible = False

    incident_date = _parse_incident_date(extra_info.get("incident_date"))
    if incident_date is None and isinstance(extra_info.get("description"), str):
        incident_date = _parse_incident_date(extra_info["description"])

    supporting_documents = _parse_supporting_documents(extra_info.get("supporting_documents"))
    if not supporting_documents and isinstance(extra_info.get("description"), str):
        if "invoice" in extra_info["description"].lower() or "bill" in extra_info["description"].lower():
            supporting_documents.append("invoice")
        if "medical report" in extra_info["description"].lower():
            supporting_documents.append("medical report")

    if not supporting_documents:
        validation_messages.append("No supporting documents were provided; this may delay claim processing.")

    effective_claim_amount = min(claim_amount, policy.sum_insured)
    if claim_amount > policy.sum_insured:
        validation_messages.append("Claim amount exceeds the policy sum insured and will be capped.")

    if effective_claim_amount <= policy.deductible:
        validation_messages.append("Claim amount does not exceed the deductible; no payable amount is expected.")
        approved_amount = 0.0
    else:
        approved_amount = effective_claim_amount - policy.deductible

    category = extra_info.get("sub_limit_category")
    if category and category in policy.sub_limits:
        sub_limit = policy.sub_limits[category]
        if approved_amount > sub_limit:
            validation_messages.append(
                f"Claim amount exceeds the {category} sub-limit of {sub_limit:.2f} and will be reduced."
            )
            approved_amount = sub_limit

    if incident_date and policy.start_date and incident_date < policy.start_date:
        validation_messages.append("Claim incident date is before policy inception; coverage gap may apply.")

    if incident_date and policy.end_date and incident_date > policy.end_date:
        validation_messages.append("Claim incident date is after policy expiry; coverage may not apply.")
        is_eligible = False

    if effective_claim_amount >= policy.sum_insured * 0.9:
        validation_messages.append("High-value claim may require manual review.")

    if is_eligible and approved_amount == 0.0:
        validation_messages.append("After deductible, the claim is not payable.")

    claim_id = f"C{uuid4().hex[:8].upper()}"
    claim = Claim(
        claim_id=claim_id,
        policy_number=policy.policy_number,
        claim_amount=claim_amount,
        incident_date=incident_date,
        supporting_documents=supporting_documents,
        extra_info={
            **{k: str(v) for k, v in extra_info.items() if k != "supporting_documents"},
            "sub_limit_category": str(category) if category else "",
        },
        status="PENDING_REVIEW",
    )

    fraud_evaluation = compute_fraud_score(claim=claim)
    if fraud_evaluation.score >= FRAUD_SCORE_THRESHOLD:
        validation_messages.append(
            "Claim registration is blocked because fraud risk validation exceeded the acceptable threshold."
        )
        metadata = {
            "policy_status": policy.status.value,
            "sum_insured": str(policy.sum_insured),
            "deductible": str(policy.deductible),
            "fraud_score": fraud_evaluation.score,
            "fraud_signals": fraud_evaluation.signals,
            "fraud_details": fraud_evaluation.details,
            "fraud_validation": "failed",
        }
        return ClaimValidationResult(
            claim_id="",
            policy_number=policy.policy_number,
            is_eligible=False,
            approved_amount=0.0,
            validation_messages=validation_messages,
            metadata={k: v for k, v in metadata.items() if v is not None},
        )

    claim.approved_amount = round(max(0.0, approved_amount), 2)
    claim.fraud_score = fraud_evaluation.score
    claim.status = "CREATED"
    if persist:
        save_claim(claim)

    metadata = {
        "policy_status": policy.status.value,
        "sum_insured": str(policy.sum_insured),
        "deductible": str(policy.deductible),
        "incident_date": incident_date.isoformat() if incident_date else None,
        "supporting_documents": supporting_documents,
        "fraud_score": fraud_evaluation.score,
        "fraud_signals": fraud_evaluation.signals,
    }
    return ClaimValidationResult(
        claim_id=claim_id,
        policy_number=policy.policy_number,
        is_eligible=is_eligible,
        approved_amount=round(max(0.0, approved_amount), 2),
        validation_messages=validation_messages,
        metadata={k: v for k, v in metadata.items() if v is not None},
    )
