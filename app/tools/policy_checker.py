from typing import Dict, Optional

from app.models.domain import PolicyStatus, get_policy


class PolicyCheckResult:
    def __init__(
        self,
        policy_number: str,
        status: str,
        is_active: bool,
        message: str,
        details: Optional[Dict[str, str]] = None,
    ):
        self.policy_number = policy_number
        self.status = status
        self.is_active = is_active
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, any]:
        return {
            "policy_number": self.policy_number,
            "status": self.status,
            "is_active": self.is_active,
            "message": self.message,
            "details": self.details,
        }


def check_policy_status(policy_number: str) -> PolicyCheckResult:
    """
    Check the status of a policy and determine if claims can be filed.
    
    Args:
        policy_number: The policy number to check
        
    Returns:
        PolicyCheckResult with status information
    """
    if not policy_number or not policy_number.strip():
        return PolicyCheckResult(
            policy_number=policy_number,
            status="UNKNOWN",
            is_active=False,
            message="Policy number is required to check status.",
            details={"error": "policy_number_missing"},
        )

    policy = get_policy(policy_number.strip())
    
    if policy is None:
        return PolicyCheckResult(
            policy_number=policy_number,
            status="NOT_FOUND",
            is_active=False,
            message="Policy not found. Please verify the policy number.",
            details={"error": "policy_not_found"},
        )

    status = policy.status
    is_active = status == PolicyStatus.ACTIVE
    
    if is_active:
        message = (
            f"Policy {policy_number} is ACTIVE. "
            f"You can file claims under this policy. "
            f"Coverage period: {policy.start_date} to {policy.end_date}."
        )
    elif status == PolicyStatus.LAPSED:
        message = (
            f"Policy {policy_number} has LAPSED. "
            f"Claims cannot be filed for this policy. "
            f"Coverage ended on {policy.end_date}."
        )
    elif status == PolicyStatus.CANCELLED:
        message = (
            f"Policy {policy_number} has been CANCELLED. "
            f"No claims can be filed. "
            f"Coverage ended on {policy.end_date}."
        )
    else:
        message = f"Policy {policy_number} status: {status}. Claims may not be eligible."

    details = {
        "status": status.value,
        "sum_insured": str(policy.sum_insured),
        "deductible": str(policy.deductible),
        "start_date": policy.start_date.isoformat(),
        "end_date": policy.end_date.isoformat(),
    }

    return PolicyCheckResult(
        policy_number=policy_number,
        status=status.value,
        is_active=is_active,
        message=message,
        details=details,
    )