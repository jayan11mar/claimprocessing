import sqlite3
from datetime import date

from app.config import get_settings
from app.memory.sqlite_memory import SQLiteMemory
from app.models.domain import Claim, get_claim, get_claims_for_policy, save_claim
from app.tools.claims_intake import register_and_validate_claim
from app.tools.fraud_detector import compute_fraud_score


def reset_settings() -> None:
    get_settings.cache_clear()


def test_sqlite_memory_initializes_database_tables(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "claims.db"))
    reset_settings()

    memory = SQLiteMemory()
    assert memory.db_path.exists(), "SQLite database file should be created on startup"

    with sqlite3.connect(memory.db_path) as conn:
        table_names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"chat_history", "policies", "claims"}.issubset(table_names)

        claim_columns = {row[1] for row in conn.execute("PRAGMA table_info(claims)")}
        assert {"status", "approved_amount", "fraud_score", "settlement_status"}.issubset(claim_columns)


def test_register_claim_persists_only_when_fraud_is_acceptable(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "claims.db"))
    reset_settings()

    SQLiteMemory()

    initial_claims = len(get_claims_for_policy("P123456"))

    result = register_and_validate_claim(
        policy_number="P123456",
        claim_amount=1200.0,
        extra_info={"incident_date": "2024-04-10", "supporting_documents": ["invoice"]},
    )

    assert result.is_eligible is True
    assert result.claim_id
    persisted = get_claim(result.claim_id)
    assert persisted is not None
    assert persisted.status == "CREATED"
    assert persisted.fraud_score is not None
    assert persisted.approved_amount == round(max(0.0, 700.0), 2)

    existing_claim = Claim(
        claim_id="CEXIST01",
        policy_number="P123456",
        claim_amount=9000.0,
        approved_amount=8500.0,
        policy_holder_id="H1001",
        hospital_name="General Hospital",
        incident_date=date.fromisoformat("2024-02-05"),
        supporting_documents=["invoice"],
        extra_info={"sub_limit_category": "hospital"},
        status="CREATED",
        fraud_score=0.2,
    )
    save_claim(existing_claim)

    result_high_fraud = register_and_validate_claim(
        policy_number="P123456",
        claim_amount=9000.0,
        extra_info={
            "incident_date": "2024-01-10",
            "supporting_documents": ["invoice"],
            "policy_holder_id": "H1001",
            "hospital_name": "General Hospital",
        },
    )

    assert result_high_fraud.is_eligible is False
    assert result_high_fraud.claim_id == ""
    assert any("fraud" in msg.lower() for msg in result_high_fraud.validation_messages)
    assert len(get_claims_for_policy("P123456")) == initial_claims + 2


def test_fraud_detector_supports_saved_and_pre_save_claims(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "claims.db"))
    reset_settings()

    SQLiteMemory()

    saved_score = compute_fraud_score("C1001")
    assert saved_score.claim_id == "C1001"
    assert 0.0 <= saved_score.score <= 1.0

    claim = Claim(
        claim_id="C9999",
        policy_number="P123456",
        claim_amount=9000.0,
        policy_holder_id="H1001",
        hospital_name="General Hospital",
        incident_date=date.fromisoformat("2024-01-10"),
        extra_info={"sub_limit_category": "hospital"},
    )
    pre_save_score = compute_fraud_score(claim=claim)
    assert pre_save_score.claim_id == "C9999"
    assert 0.0 <= pre_save_score.score <= 1.0
