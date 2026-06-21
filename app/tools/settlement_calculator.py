from typing import List, Optional

from app.models.domain import SettlementBreakdown, Claim, get_claim, get_policy


def calculate_settlement(claim_id: str) -> SettlementBreakdown:
    claim = get_claim(claim_id)
    if claim is None:
        raise ValueError(f"Claim {claim_id} not found.")

    policy = get_policy(claim.policy_number)
    if policy is None:
        raise ValueError(f"Policy {claim.policy_number} not found for claim {claim_id}.")

    gross_amount = claim.claim_amount
    deductible = min(policy.deductible, gross_amount)
    amount_after_deductible = max(0.0, gross_amount - deductible)

    depreciation_percent = 0.0
    depreciation_category = claim.extra_info.get("depreciation_category")
    if depreciation_category and depreciation_category in policy.depreciation_schedule:
        depreciation_percent = policy.depreciation_schedule[depreciation_category]
    elif claim.extra_info.get("depreciation_percent"):
        try:
            depreciation_percent = float(claim.extra_info["depreciation_percent"])
        except ValueError:
            depreciation_percent = 0.0

    depreciation_amount = round(amount_after_deductible * (depreciation_percent / 100.0), 2)
    amount_after_depreciation = max(0.0, amount_after_deductible - depreciation_amount)
    copay_amount = round(amount_after_depreciation * (policy.copay_percent / 100.0), 2)
    approved_amount = max(0.0, amount_after_depreciation - copay_amount)
    notes: List[str] = []

    notes.append(f"Applied deductible of {deductible:.2f}.")
    if depreciation_amount > 0:
        notes.append(f"Applied depreciation of {depreciation_amount:.2f} at {depreciation_percent:.1f}%.")
    notes.append(f"Computed copay of {copay_amount:.2f} at {policy.copay_percent:.1f}%.")

    category = claim.extra_info.get("sub_limit_category")
    sub_limit_applied = None
    if category:
        notes.append(f"Checking sub-limit for '{category}'.")
        sub_limit = policy.sub_limits.get(category)
        if sub_limit is not None and approved_amount > sub_limit:
            notes.append(f"Applied sub-limit of {sub_limit:.2f} for category '{category}'.")
            approved_amount = sub_limit
            sub_limit_applied = round(sub_limit, 2)

    notes.append(f"Final approved amount is {approved_amount:.2f}.")

    return SettlementBreakdown(
        claim_id=claim_id,
        gross_amount=round(gross_amount, 2),
        deductible=round(deductible, 2),
        depreciation_amount=round(depreciation_amount, 2),
        copay_amount=round(copay_amount, 2),
        approved_amount=round(approved_amount, 2),
        sub_limit_applied=sub_limit_applied,
        notes=notes,
    )
