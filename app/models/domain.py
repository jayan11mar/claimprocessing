import json
import sqlite3
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.config import get_settings


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
    approved_amount: Optional[float] = None
    policy_holder_id: Optional[str] = None
    diagnosis_code: Optional[str] = None
    hospital_name: Optional[str] = None
    admission_date: Optional[date] = None
    discharge_date: Optional[date] = None
    incident_date: Optional[date] = None
    supporting_documents: List[str] = Field(default_factory=list)
    extra_info: Dict[str, str] = Field(default_factory=dict)
    status: Optional[str] = None
    fraud_score: Optional[float] = None
    settlement_status: Optional[str] = None


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

_db_initialized = False


def _get_db_path() -> Path:
    settings = get_settings()
    db_path = Path(settings.SQLITE_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _connect():
    conn = sqlite3.connect(_get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def _serialize_json(value: Any) -> str:
    return json.dumps(value, default=str)


def _deserialize_json(value: Optional[str]) -> Any:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _row_to_policy(row: sqlite3.Row) -> Policy:
    return Policy(
        policy_number=row["policy_number"],
        policy_holder_id=row["policy_holder_id"],
        status=PolicyStatus(row["status"]),
        sum_insured=row["sum_insured"],
        deductible=row["deductible"],
        copay_percent=row["copay_percent"],
        sub_limits=_deserialize_json(row["sub_limits"]),
        depreciation_schedule=_deserialize_json(row["depreciation_schedule"]),
        start_date=date.fromisoformat(row["start_date"]),
        end_date=date.fromisoformat(row["end_date"]),
    )


def _row_to_claim(row: sqlite3.Row) -> Claim:
    return Claim(
        claim_id=row["claim_id"],
        policy_number=row["policy_number"],
        claim_amount=row["claim_amount"],
        policy_holder_id=row["policy_holder_id"],
        diagnosis_code=row["diagnosis_code"],
        hospital_name=row["hospital_name"],
        admission_date=date.fromisoformat(row["admission_date"]) if row["admission_date"] else None,
        discharge_date=date.fromisoformat(row["discharge_date"]) if row["discharge_date"] else None,
        incident_date=date.fromisoformat(row["incident_date"]) if row["incident_date"] else None,
        supporting_documents=_deserialize_json(row["supporting_documents"]) if row["supporting_documents"] else [],
        extra_info=_deserialize_json(row["extra_info"]) if row["extra_info"] else {},
        status=row["status"],
        fraud_score=row["fraud_score"],
        settlement_status=row["settlement_status"],
        approved_amount=row["approved_amount"],
    )


def _ensure_db() -> None:
    global _db_initialized
    if _db_initialized:
        return
    with sqlite3.connect(_get_db_path()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS policies (
                policy_number TEXT PRIMARY KEY,
                policy_holder_id TEXT,
                status TEXT NOT NULL,
                sum_insured REAL NOT NULL,
                deductible REAL NOT NULL,
                copay_percent REAL NOT NULL,
                sub_limits TEXT,
                depreciation_schedule TEXT,
                start_date TEXT,
                end_date TEXT,
                product_code TEXT,
                coverage_type TEXT,
                underwriting_class TEXT,
                risk_category TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                policy_number TEXT NOT NULL,
                policy_holder_id TEXT,
                claim_amount REAL NOT NULL,
                incident_date TEXT,
                admission_date TEXT,
                discharge_date TEXT,
                diagnosis_code TEXT,
                hospital_name TEXT,
                supporting_documents TEXT,
                extra_info TEXT,
                status TEXT,
                loss_type TEXT,
                reported_date TEXT,
                closed_date TEXT,
                approved_amount REAL,
                fraud_score REAL,
                settlement_status TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(policy_number) REFERENCES policies(policy_number)
            )
            """
        )
        conn.commit()
    _load_demo_policies_into_db()
    _load_demo_claims_into_db()
    _db_initialized = True


def _load_demo_policies_into_db() -> None:
    with _connect() as conn:
        now = datetime.utcnow().isoformat()
        for policy in _DEMO_POLICIES.values():
            conn.execute(
                """
                INSERT OR IGNORE INTO policies (
                    policy_number,
                    policy_holder_id,
                    status,
                    sum_insured,
                    deductible,
                    copay_percent,
                    sub_limits,
                    depreciation_schedule,
                    start_date,
                    end_date,
                    product_code,
                    coverage_type,
                    underwriting_class,
                    risk_category,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    policy.policy_number,
                    policy.policy_holder_id,
                    policy.status.value,
                    policy.sum_insured,
                    policy.deductible,
                    policy.copay_percent,
                    _serialize_json(policy.sub_limits),
                    _serialize_json(policy.depreciation_schedule),
                    policy.start_date.isoformat(),
                    policy.end_date.isoformat(),
                    "",
                    "",
                    "",
                    "",
                    now,
                    now,
                ),
            )
        conn.commit()


def _load_demo_claims_into_db() -> None:
    with _connect() as conn:
        now = datetime.utcnow().isoformat()
        for claim in _DEMO_CLAIMS.values():
            conn.execute(
                """
                INSERT OR IGNORE INTO claims (
                    claim_id,
                    policy_number,
                    policy_holder_id,
                    claim_amount,
                    incident_date,
                    admission_date,
                    discharge_date,
                    diagnosis_code,
                    hospital_name,
                    supporting_documents,
                    extra_info,
                    status,
                    loss_type,
                    reported_date,
                    closed_date,
                    approved_amount,
                    fraud_score,
                    settlement_status,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.claim_id,
                    claim.policy_number,
                    claim.policy_holder_id,
                    claim.claim_amount,
                    claim.incident_date.isoformat() if claim.incident_date else None,
                    claim.admission_date.isoformat() if claim.admission_date else None,
                    claim.discharge_date.isoformat() if claim.discharge_date else None,
                    claim.diagnosis_code,
                    claim.hospital_name,
                    _serialize_json(claim.supporting_documents),
                    _serialize_json(claim.extra_info),
                    claim.status,
                    None,
                    None,
                    None,
                    claim.approved_amount,
                    claim.fraud_score,
                    claim.settlement_status,
                    now,
                    now,
                ),
            )
        conn.commit()


def create_claim_id() -> str:
    return f"C{uuid4().hex[:8].upper()}"


def get_policy(policy_number: str) -> Optional[Policy]:
    _ensure_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM policies WHERE policy_number = ?",
            (policy_number,),
        ).fetchone()
        if row:
            return _row_to_policy(row)
    return _DEMO_POLICIES.get(policy_number)


def get_claim(claim_id: str) -> Optional[Claim]:
    _ensure_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM claims WHERE claim_id = ?",
            (claim_id,),
        ).fetchone()
        if row:
            return _row_to_claim(row)
    return _DEMO_CLAIMS.get(claim_id)


def save_claim(claim: Claim) -> None:
    _ensure_db()
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO claims (
                claim_id,
                policy_number,
                policy_holder_id,
                claim_amount,
                incident_date,
                admission_date,
                discharge_date,
                diagnosis_code,
                hospital_name,
                supporting_documents,
                extra_info,
                status,
                loss_type,
                reported_date,
                closed_date,
                approved_amount,
                fraud_score,
                settlement_status,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(claim_id) DO UPDATE SET
                policy_number=excluded.policy_number,
                policy_holder_id=excluded.policy_holder_id,
                claim_amount=excluded.claim_amount,
                incident_date=excluded.incident_date,
                admission_date=excluded.admission_date,
                discharge_date=excluded.discharge_date,
                diagnosis_code=excluded.diagnosis_code,
                hospital_name=excluded.hospital_name,
                supporting_documents=excluded.supporting_documents,
                extra_info=excluded.extra_info,
                status=excluded.status,
                loss_type=excluded.loss_type,
                reported_date=excluded.reported_date,
                closed_date=excluded.closed_date,
                approved_amount=excluded.approved_amount,
                fraud_score=excluded.fraud_score,
                settlement_status=excluded.settlement_status,
                updated_at=excluded.updated_at
            """,
            (
                claim.claim_id,
                claim.policy_number,
                claim.policy_holder_id,
                claim.claim_amount,
                claim.incident_date.isoformat() if claim.incident_date else None,
                claim.admission_date.isoformat() if claim.admission_date else None,
                claim.discharge_date.isoformat() if claim.discharge_date else None,
                claim.diagnosis_code,
                claim.hospital_name,
                _serialize_json(claim.supporting_documents),
                _serialize_json(claim.extra_info),
                claim.status,
                None,
                None,
                None,
                claim.approved_amount,
                claim.fraud_score,
                claim.settlement_status,
                now,
                now,
            ),
        )
        conn.commit()


def get_claims_for_policy(policy_number: str) -> List[Claim]:
    _ensure_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM claims WHERE policy_number = ?",
            (policy_number,),
        ).fetchall()
    return [_row_to_claim(row) for row in rows]


def get_all_claims() -> List[Claim]:
    _ensure_db()
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM claims").fetchall()
    return [_row_to_claim(row) for row in rows]


def get_claims_for_policy_holder(policy_holder_id: str) -> List[Claim]:
    _ensure_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM claims WHERE policy_holder_id = ?",
            (policy_holder_id,),
        ).fetchall()
    return [_row_to_claim(row) for row in rows]
