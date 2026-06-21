from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class PolicyStatus(str, Enum):
    ACTIVE = "ACTIVE"
    LAPSED = "LAPSED"
    CANCELLED = "CANCELLED"


class Policy(BaseModel):
    policy_number: str
    policy_holder_id: Optional[str] = None
    status: PolicyStatus = PolicyStatus.ACTIVE
    sum_insured: float = Field(..., ge=0)
    deductible: float = Field(..., ge=0)
    copay_percent: float = Field(..., ge=0, le=100)
    sub_limits: Dict[str, float] = Field(default_factory=dict)
    depreciation_schedule: Dict[str, float] = Field(default_factory=dict)
    start_date: date
    end_date: date


class Claim(BaseModel):
    claim_id: str
    policy_number: str
    claim_amount: float = Field(..., ge=0)
    policy_holder_id: Optional[str] = None
    diagnosis_code: Optional[str] = None
    hospital_name: Optional[str] = None
    admission_date: Optional[date] = None
    discharge_date: Optional[date] = None
    incident_date: Optional[date] = None
    supporting_documents: List[str] = Field(default_factory=list)
    extra_info: Dict[str, str] = Field(default_factory=dict)


class FraudScoreResult(BaseModel):
    claim_id: str
    score: float = Field(..., ge=0, le=1)
    signals: List[str] = Field(default_factory=list)
    details: Dict[str, str] = Field(default_factory=dict)


class SettlementBreakdown(BaseModel):
    claim_id: str
    gross_amount: float
    deductible: float
    depreciation_amount: float = 0.0
    copay_amount: float
    approved_amount: float
    sub_limit_applied: Optional[float] = None
    notes: List[str] = Field(default_factory=list)


class ClaimValidationResult(BaseModel):
    claim_id: str
    policy_number: str
    is_eligible: bool
    approved_amount: float
    validation_messages: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


_DEMO_POLICIES: Dict[str, Policy] = {
    "P123456": Policy(
        policy_number="P123456",
        policy_holder_id="H1001",
        status=PolicyStatus.ACTIVE,
        sum_insured=10000.0,
        deductible=500.0,
        copay_percent=10.0,
        sub_limits={"hospital": 5000.0, "dental": 1000.0},
        depreciation_schedule={"hospital": 10.0, "dental": 20.0},
        start_date=date.fromisoformat("2024-01-01"),
        end_date=date.fromisoformat("2027-01-01"),
    ),
    "P789012": Policy(
        policy_number="P789012",
        policy_holder_id="H1001",
        status=PolicyStatus.ACTIVE,
        sum_insured=500000.0,
        deductible=10000.0,
        copay_percent=20.0,
        sub_limits={"hospital": 100000.0, "surgery": 150000.0},
        depreciation_schedule={"hospital": 5.0, "surgery": 15.0},
        start_date=date.fromisoformat("2024-02-01"),
        end_date=date.fromisoformat("2028-02-01"),
    ),
    "P654321": Policy(
        policy_number="P654321",
        status=PolicyStatus.LAPSED,
        sum_insured=5000.0,
        deductible=250.0,
        copay_percent=20.0,
        sub_limits={"vision": 300.0},
        depreciation_schedule={"vision": 10.0},
        start_date=date.fromisoformat("2023-01-01"),
        end_date=date.fromisoformat("2024-01-01"),
    ),
}

_DEMO_CLAIMS: Dict[str, Claim] = {
    "C1001": Claim(
        claim_id="C1001",
        policy_number="P123456",
        policy_holder_id="H1001",
        claim_amount=1200.0,
        diagnosis_code="S75.1",
        hospital_name="General Hospital",
        incident_date=date.fromisoformat("2024-03-10"),
        supporting_documents=["hospital_bill", "diagnostic_report"],
        extra_info={"sub_limit_category": "hospital", "depreciation_category": "hospital"},
    ),
    "C2001": Claim(
        claim_id="C2001",
        policy_number="P789012",
        policy_holder_id="H1001",
        claim_amount=120000.0,
        diagnosis_code="T20.2",
        hospital_name="City Care Hospital",
        incident_date=date.fromisoformat("2024-11-15"),
        supporting_documents=["discharge_summary", "itemized_bill"],
        extra_info={"sub_limit_category": "surgery", "depreciation_category": "surgery"},
    ),
    "C2002": Claim(
        claim_id="C2002",
        policy_number="P789012",
        policy_holder_id="H1001",
        claim_amount=120000.0,
        diagnosis_code="T20.2",
        hospital_name="City Care Hospital",
        incident_date=date.fromisoformat("2024-11-20"),
        supporting_documents=["discharge_summary", "itemized_bill"],
        extra_info={"sub_limit_category": "surgery", "depreciation_category": "surgery"},
    ),
}


def create_claim_id() -> str:
    return f"C{uuid4().hex[:8].upper()}"


def get_policy(policy_number: str) -> Optional[Policy]:
    return _DEMO_POLICIES.get(policy_number)


def get_claim(claim_id: str) -> Optional[Claim]:
    return _DEMO_CLAIMS.get(claim_id)


def save_claim(claim: Claim) -> None:
    _DEMO_CLAIMS[claim.claim_id] = claim


def get_claims_for_policy(policy_number: str) -> List[Claim]:
    return [claim for claim in _DEMO_CLAIMS.values() if claim.policy_number == policy_number]


def get_all_claims() -> List[Claim]:
    return list(_DEMO_CLAIMS.values())


def get_claims_for_policy_holder(policy_holder_id: str) -> List[Claim]:
    return [
        claim for claim in _DEMO_CLAIMS.values()
        if claim.policy_holder_id == policy_holder_id
    ]
