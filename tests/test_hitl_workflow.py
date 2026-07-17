"""End-to-end tests for the HITL (Human-In-The-Loop) workflow.

Validates:
1. Trigger rule evaluation (all five rules)
2. Pause / create task flow via the manager
3. Pending list and review (approve / reject) via API
4. ``/reset`` does NOT delete pending tasks
5. SQLite persistence survives a "restart" (new store instance)
6. Confidence / recall metrics meet thresholds (precision >=85%, recall >=95%)
"""

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.api import server
from app.config import get_settings
from app.hitl.manager import (
    get_hitl_manager,
    reset_hitl_manager_singleton,
)
from app.hitl.models import HITLReviewRequest, HITLTask
from app.hitl.store import (
    HITLTaskStore,
    get_task_store,
    reset_task_store_singleton,
)
from app.hitl.triggers import (
    clear_rules_cache,
    evaluate_triggers,
    load_rules,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_hitl_state():
    """Reset all HITL singletons and caches before each test."""
    reset_task_store_singleton()
    reset_hitl_manager_singleton()
    clear_rules_cache()
    # Use a temp DB path so tests don't collide
    os.environ["HITL_STORE_PATH"] = tempfile.mktemp(suffix=".db")
    # Ensure HITL is enabled
    os.environ["ENABLE_HITL"] = "true"
    get_settings.cache_clear()
    yield
    # Cleanup temp DB
    db_path = os.environ.get("HITL_STORE_PATH", "")
    if db_path and os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass
    # Also cleanup the default path if a test wrote there
    default_path = "data/hitl_tasks.db"
    if os.path.exists(default_path):
        try:
            os.remove(default_path)
        except OSError:
            pass


@pytest.fixture
def client():
    """Return a TestClient with a clean server state."""
    reset_task_store_singleton()
    reset_hitl_manager_singleton()
    clear_rules_cache()
    return TestClient(server.app)


# ── Helper: build a comprehensive context that triggers high_amount ────────


def _high_amount_context(session_id: str = "test-sess") -> dict:
    return {
        "session_id": session_id,
        "user_message": "Process claim for Rs 600,000",
        "agent_response": "Claim requires manual review due to high amount.",
        "claim_amount": 600000,
        "decision": "pending",
        "fraud_flag": False,
        "policy_exclusion": False,
        "reasoning_trace": (
            "Agent calculated settlement amount. "
            "Amount of Rs 600,000 exceeds threshold of Rs 500,000. "
            "Recommend escalating to human reviewer."
        ),
        "confidence": 0.72,
        "recommendation": {
            "action": "manual_review",
            "claim_id": "C-HIGH-001",
            "proposed_settlement": 480000,
            "reason": "High-value claim",
        },
        "retrieved_chunks": [
            {
                "chunk_id": "chunk-001",
                "text": "Claims exceeding Rs 500,000 require manual approval.",
                "source_id": "policy_manual_v2",
                "score": 0.92,
            },
            {
                "chunk_id": "chunk-002",
                "text": "Settlement amounts are calculated as sum insured minus deductible.",
                "source_id": "policy_manual_v2",
                "score": 0.88,
            },
        ],
    }


def _rejection_context(session_id: str = "test-sess") -> dict:
    return {
        "session_id": session_id,
        "user_message": "Reject claim for policy exclusion",
        "agent_response": "Claim is being rejected.",
        "decision": "reject",
        "fraud_flag": False,
        "policy_exclusion": False,
        "claim_amount": 10000,
        "reasoning_trace": "Claim falls outside coverage period.",
        "confidence": 0.95,
        "recommendation": {"action": "reject", "reason": "Policy lapsed"},
        "retrieved_chunks": [],
    }


def _partial_settlement_context(session_id: str = "test-sess") -> dict:
    return {
        "session_id": session_id,
        "user_message": "Settle partially",
        "agent_response": "Partial settlement approved at 60%.",
        "decision": "partial",
        "fraud_flag": False,
        "policy_exclusion": False,
        "claim_amount": 50000,
        "reasoning_trace": "Only 60% covered due to sub-limit.",
        "confidence": 0.80,
        "recommendation": {"action": "partial", "amount": 30000},
        "retrieved_chunks": [],
    }


def _fraud_context(session_id: str = "test-sess") -> dict:
    return {
        "session_id": session_id,
        "user_message": "Check claim for fraud",
        "agent_response": "Suspicious activity detected.",
        "fraud_flag": True,
        "decision": "pending",
        "policy_exclusion": False,
        "claim_amount": 10000,
        "reasoning_trace": "Duplicate claim detected.",
        "confidence": 0.99,
        "recommendation": {"action": "flag_for_review", "fraud_signals": ["duplicate"]},
        "retrieved_chunks": [],
    }


def _exclusion_context(session_id: str = "test-sess") -> dict:
    return {
        "session_id": session_id,
        "user_message": "Apply policy exclusion",
        "agent_response": "Exclusion applied for pre-existing condition.",
        "policy_exclusion": True,
        "decision": "pending",
        "fraud_flag": False,
        "claim_amount": 25000,
        "reasoning_trace": "Pre-existing condition exclusion applies.",
        "confidence": 0.88,
        "recommendation": {"action": "exclude", "exclusion_clause": "pre_existing"},
        "retrieved_chunks": [],
    }


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Trigger rule evaluation
# ══════════════════════════════════════════════════════════════════════════


class TestTriggerEvaluation:
    """Validate that each of the 5 rules triggers correctly."""

    def test_high_amount_triggers(self):
        """Claim amount > Rs 5L must trigger high_amount rule."""
        result = evaluate_triggers(_high_amount_context())
        assert result.triggered is True
        rule_ids = [r["rule_id"] for r in result.matched_rules]
        assert "high_amount" in rule_ids
        assert result.task is not None
        assert result.task.rule_id == "high_amount"

    def test_rejection_triggers(self):
        """Decision == 'reject' must trigger claim_rejection rule."""
        result = evaluate_triggers(_rejection_context())
        assert result.triggered is True
        rule_ids = [r["rule_id"] for r in result.matched_rules]
        assert "claim_rejection" in rule_ids

    def test_partial_settlement_triggers(self):
        """Decision == 'partial' must trigger partial_settlement rule."""
        result = evaluate_triggers(_partial_settlement_context())
        assert result.triggered is True
        rule_ids = [r["rule_id"] for r in result.matched_rules]
        assert "partial_settlement" in rule_ids

    def test_fraud_flag_triggers(self):
        """fraud_flag == True must trigger fraud_flag rule."""
        result = evaluate_triggers(_fraud_context())
        assert result.triggered is True
        rule_ids = [r["rule_id"] for r in result.matched_rules]
        assert "fraud_flag" in rule_ids

    def test_policy_exclusion_triggers(self):
        """policy_exclusion == True must trigger policy_exclusion rule."""
        result = evaluate_triggers(_exclusion_context())
        assert result.triggered is True
        rule_ids = [r["rule_id"] for r in result.matched_rules]
        assert "policy_exclusion" in rule_ids

    def test_no_trigger_for_normal_claim(self):
        """A normal low-value claim with no flags must NOT trigger."""
        context = {
            "claim_amount": 10000,
            "decision": "approve",
            "fraud_flag": False,
            "policy_exclusion": False,
        }
        result = evaluate_triggers(context)
        assert result.triggered is False
        assert result.task is None

    def test_multiple_triggers_can_fire(self):
        """Context matching multiple rules must list all matched rules."""
        context = {
            "claim_amount": 600000,
            "decision": "reject",
            "fraud_flag": True,
            "policy_exclusion": True,
        }
        result = evaluate_triggers(context)
        assert result.triggered is True
        rule_ids = [r["rule_id"] for r in result.matched_rules]
        assert "high_amount" in rule_ids
        assert "claim_rejection" in rule_ids
        assert "fraud_flag" in rule_ids
        assert "policy_exclusion" in rule_ids
        # First matched rule becomes the primary task rule
        assert result.task is not None


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Pause / Resume lifecycle via manager
# ══════════════════════════════════════════════════════════════════════════


class TestManagerPauseResume:
    """Validate the HITLManager pause/resume lifecycle."""

    def test_pause_creates_persisted_task(self):
        """Calling pause() with a triggering context must persist a task."""
        manager = get_hitl_manager()
        result = manager.pause(_high_amount_context())
        assert result.triggered is True
        assert result.task is not None
        assert result.task.status == "pending"
        assert result.task.task_id.startswith("hitl_")
        # Verify it's in the store
        fetched = manager.get_task(result.task.task_id)
        assert fetched is not None
        assert fetched.status == "pending"

    def test_pause_serialises_full_context(self):
        """The persisted task must contain retrieved chunks, reasoning, confidence, recommendation."""
        manager = get_hitl_manager()
        ctx = _high_amount_context()
        result = manager.pause(ctx)
        task = result.task
        assert task.retrieved_chunks == ctx["retrieved_chunks"]
        assert task.reasoning_trace == ctx["reasoning_trace"]
        assert task.confidence == ctx["confidence"]
        assert task.recommendation == ctx["recommendation"]
        assert task.user_message == ctx["user_message"]
        assert task.agent_response == ctx["agent_response"]
        assert task.session_id == ctx["session_id"]

    def test_resume_approve(self):
        """Resume with 'approved' must update the task status and decision."""
        manager = get_hitl_manager()
        result = manager.pause(_high_amount_context())
        task_id = result.task.task_id
        updated = manager.resume(task_id, "approved", "Looks good, proceed.")
        assert updated is not None
        assert updated.status == "approved"
        assert updated.decision == "approved"
        assert updated.reviewer_comments == "Looks good, proceed."
        assert updated.reviewed_at is not None

    def test_resume_reject(self):
        """Resume with 'rejected' must update the task."""
        manager = get_hitl_manager()
        result = manager.pause(_high_amount_context())
        updated = manager.resume(result.task.task_id, "rejected", "Insufficient documentation.")
        assert updated is not None
        assert updated.status == "rejected"
        assert updated.decision == "rejected"

    def test_resume_unknown_task_returns_none(self):
        """Resume on a non-existent task must return None."""
        manager = get_hitl_manager()
        task = manager.resume("nonexistent", "approved")
        assert task is None

    def test_resume_already_reviewed(self):
        """Resume on an already-reviewed task must return None (status != pending)."""
        manager = get_hitl_manager()
        result = manager.pause(_high_amount_context())
        manager.resume(result.task.task_id, "approved")
        # Second resume should fail because status is no longer pending
        second = manager.resume(result.task.task_id, "rejected")
        assert second is None

    def test_list_pending(self):
        """Pending tasks must appear in the pending list; reviewed tasks must not."""
        manager = get_hitl_manager()
        # Create two tasks
        r1 = manager.pause(_high_amount_context(session_id="sess-1"))
        r2 = manager.pause(_rejection_context(session_id="sess-2"))
        # Review one
        manager.resume(r1.task.task_id, "approved")

        pending = manager.list_pending()
        pending_ids = [t.task_id for t in pending]
        assert r2.task.task_id in pending_ids
        assert r1.task.task_id not in pending_ids

    def test_pause_when_hitl_disabled(self):
        """When ENABLE_HITL is False, pause() must short-circuit."""
        os.environ["ENABLE_HITL"] = "false"
        get_settings.cache_clear()
        manager = get_hitl_manager()
        result = manager.pause(_high_amount_context())
        assert result.triggered is False
        assert result.task is None

    def test_count_pending(self):
        """count_pending must return the correct number."""
        manager = get_hitl_manager()
        assert manager.count_pending() == 0
        manager.pause(_high_amount_context())
        assert manager.count_pending() == 1
        manager.pause(_rejection_context())
        assert manager.count_pending() == 2


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3 — API endpoints
# ══════════════════════════════════════════════════════════════════════════


class TestAPIEndpoints:
    """Validate the /hitl/pending and /hitl/review/{task_id} endpoints."""

    def test_hitl_pending_empty_when_disabled(self, client):
        """When HITL disabled, /hitl/pending must return empty with enabled=False."""
        os.environ["ENABLE_HITL"] = "false"
        get_settings.cache_clear()
        resp = client.get("/hitl/pending")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False
        assert data["count"] == 0

    def test_hitl_pending_returns_tasks(self, client):
        """With HITL enabled, /hitl/pending must return pending tasks."""
        # Pre-create a task via the manager
        manager = get_hitl_manager()
        manager.pause(_high_amount_context())
        resp = client.get("/hitl/pending")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["count"] >= 1
        assert len(data["tasks"]) >= 1
        task = data["tasks"][0]
        assert task["status"] == "pending"
        assert "task_id" in task
        assert "retrieved_chunks" in task
        assert "reasoning_trace" in task
        assert "confidence" in task

    def test_hitl_review_approve(self, client):
        """POST /hitl/review/{task_id} with approved must update the task."""
        manager = get_hitl_manager()
        result = manager.pause(_high_amount_context())
        task_id = result.task.task_id
        resp = client.post(
            f"/hitl/review/{task_id}",
            json={"decision": "approved", "comments": "Proceed with settlement"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["task"]["status"] == "approved"
        assert data["task"]["decision"] == "approved"

    def test_hitl_review_reject(self, client):
        """POST /hitl/review/{task_id} with rejected must update the task."""
        manager = get_hitl_manager()
        result = manager.pause(_high_amount_context())
        resp = client.post(
            f"/hitl/review/{result.task.task_id}",
            json={"decision": "rejected", "comments": "Need more docs"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"]["status"] == "rejected"
        assert data["task"]["reviewer_comments"] == "Need more docs"

    def test_hitl_review_invalid_decision(self, client):
        """POST with an invalid decision must return 422."""
        manager = get_hitl_manager()
        result = manager.pause(_high_amount_context())
        resp = client.post(
            f"/hitl/review/{result.task.task_id}",
            json={"decision": "invalid_decision"},
        )
        assert resp.status_code == 422

    def test_hitl_review_not_found(self, client):
        """POST /hitl/review on a non-existent task must return 404."""
        resp = client.post(
            "/hitl/review/nonexistent",
            json={"decision": "approved"},
        )
        assert resp.status_code == 404

    def test_hitl_review_disabled(self, client):
        """When HITL disabled, review must return 503."""
        os.environ["ENABLE_HITL"] = "false"
        get_settings.cache_clear()
        resp = client.post(
            "/hitl/review/some-task",
            json={"decision": "approved"},
        )
        assert resp.status_code == 503

    def test_hitl_get_task(self, client):
        """GET /hitl/task/{task_id} must return the task regardless of status."""
        manager = get_hitl_manager()
        result = manager.pause(_high_amount_context())
        task_id = result.task.task_id
        resp = client.get(f"/hitl/task/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"]["task_id"] == task_id
        assert data["task"]["status"] == "pending"

        # After review
        manager.resume(task_id, "approved")
        resp2 = client.get(f"/hitl/task/{task_id}")
        assert resp2.status_code == 200
        assert resp2.json()["task"]["status"] == "approved"

    def test_hitl_get_task_not_found(self, client):
        """GET /hitl/task for a non-existent task must return 404."""
        resp = client.get("/hitl/task/nonexistent")
        assert resp.status_code == 404


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4 — /reset does NOT delete pending tasks
# ══════════════════════════════════════════════════════════════════════════


class TestResetPreservesHITL:
    """The /reset endpoint must NOT clear HITL pending tasks."""

    def test_pending_tasks_survive_reset(self, client):
        """After /reset, pending HITL tasks must still be accessible."""
        # Use a known session for the chat
        session_id = "hitl-reset-test"
        manager = get_hitl_manager()
        result = manager.pause(_high_amount_context(session_id=session_id))
        task_id = result.task.task_id

        # Confirm pending
        resp = client.get("/hitl/pending")
        assert resp.status_code == 200
        assert any(t["task_id"] == task_id for t in resp.json()["tasks"])

        # Perform a reset
        resp2 = client.post("/reset", json={"session_id": session_id})
        assert resp2.status_code == 200

        # Pending tasks must still exist
        resp3 = client.get("/hitl/pending")
        assert resp3.status_code == 200
        pending_ids = [t["task_id"] for t in resp3.json()["tasks"]]
        assert task_id in pending_ids, "HITL pending tasks must survive /reset"

        # The task must still be reviewable
        resp4 = client.post(
            f"/hitl/review/{task_id}",
            json={"decision": "approved", "comments": "Still valid after reset"},
        )
        assert resp4.status_code == 200
        assert resp4.json()["task"]["status"] == "approved"


# ══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Persistence across "restart"
# ══════════════════════════════════════════════════════════════════════════


class TestPersistence:
    """Prove that HITL tasks survive a store re-initialisation (restart)."""

    def test_tasks_survive_store_restart(self):
        """Create a task, re-create the store (simulate restart), task must still exist."""
        db_path = os.environ["HITL_STORE_PATH"]
        # First store
        store1 = HITLTaskStore(db_path=db_path)
        task = HITLTask(
            session_id="persist-test",
            rule_id="high_amount",
            rule_reason="Test persistence",
            retrieved_chunks=[{"chunk_id": "c1", "text": "test"}],
            reasoning_trace="persistence test",
            confidence=0.95,
            recommendation={"action": "approve"},
            user_message="Test message",
            agent_response="Test response",
        )
        store1.create_task(task)
        task_id = task.task_id

        # "Restart" — create a new store instance pointing to the same DB
        store2 = HITLTaskStore(db_path=db_path)
        fetched = store2.get_task(task_id)
        assert fetched is not None
        assert fetched.task_id == task_id
        assert fetched.status == "pending"
        assert fetched.rule_id == "high_amount"
        assert len(fetched.retrieved_chunks) == 1
        assert fetched.retrieved_chunks[0]["chunk_id"] == "c1"
        assert fetched.confidence == 0.95
        assert fetched.recommendation == {"action": "approve"}
        assert fetched.user_message == "Test message"
        assert fetched.agent_response == "Test response"

    def test_tasks_survive_api_restart(self, client):
        """Create a task via API, then simulate restart by re-creating the store."""
        manager = get_hitl_manager()
        result = manager.pause(_high_amount_context(session_id="restart-api-test"))
        task_id = result.task.task_id

        # Simulate restart: reset singletons, create new store pointing to same DB
        db_path = os.environ["HITL_STORE_PATH"]
        reset_task_store_singleton()
        reset_hitl_manager_singleton()
        clear_rules_cache()
        # Point to the same DB
        os.environ["HITL_STORE_PATH"] = db_path
        get_settings.cache_clear()

        manager2 = get_hitl_manager()
        fetched = manager2.get_task(task_id)
        assert fetched is not None
        assert fetched.status == "pending"

        # Can still be reviewed
        updated = manager2.resume(task_id, "approved", "Survived restart")
        assert updated is not None
        assert updated.status == "approved"


# ══════════════════════════════════════════════════════════════════════════
# SECTION 6 — End-to-end pause / approve / resume flow
# ══════════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    """Full end-to-end: pause → list → review → verify final state."""

    def test_full_e2e_approve(self):
        """Complete flow: pause → pending has task → approve → task removed from pending."""
        manager = get_hitl_manager()

        # 1. Pause on a high-amount claim
        ctx = _high_amount_context(session_id="e2e-sess")
        pause_result = manager.pause(ctx)
        assert pause_result.triggered is True
        task_id = pause_result.task.task_id

        # 2. Pending list includes the task
        pending = manager.list_pending()
        assert any(t.task_id == task_id for t in pending)

        # 3. Approve the task
        updated = manager.resume(task_id, "approved", "E2E approval")
        assert updated.status == "approved"
        assert updated.decision == "approved"

        # 4. Task no longer in pending
        pending_after = manager.list_pending()
        assert not any(t.task_id == task_id for t in pending_after)

        # 5. Task is still retrievable via get_task
        final = manager.get_task(task_id)
        assert final is not None
        assert final.status == "approved"
        assert final.decision == "approved"
        assert final.reviewer_comments == "E2E approval"
        assert final.reviewed_at is not None

        # 6. All serialised context preserved
        assert final.retrieved_chunks == ctx["retrieved_chunks"]
        assert final.reasoning_trace == ctx["reasoning_trace"]
        assert final.confidence == ctx["confidence"]
        assert final.recommendation == ctx["recommendation"]

    def test_full_e2e_reject(self):
        """Complete flow: pause → reject → verify rejected state."""
        manager = get_hitl_manager()

        ctx = _rejection_context(session_id="e2e-reject")
        pause_result = manager.pause(ctx)
        task_id = pause_result.task.task_id

        manager.resume(task_id, "rejected", "Documents insufficient")

        final = manager.get_task(task_id)
        assert final.status == "rejected"
        assert final.decision == "rejected"
        assert final.reviewer_comments == "Documents insufficient"


# ══════════════════════════════════════════════════════════════════════════
# SECTION 7 — Precision / Recall validation
# ══════════════════════════════════════════════════════════════════════════


class TestPrecisionRecall:
    """Validate precision >=85% and recall >=95% on a labelled golden set.

    We define a golden set of scenarios and treat the trigger evaluation as
    a classifier:
      - True Positive (TP): trigger fired and rule matched the expected rule.
      - False Positive (FP): trigger fired when it should not have, or wrong rule.
      - False Negative (FN): trigger should have fired but did not.
      - True Negative (TN): trigger correctly did not fire.

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    """

    # List of (context, expected_rule_id_or_none)
    GOLDEN_SET = [
        (_high_amount_context(), "high_amount"),
        (_rejection_context(), "claim_rejection"),
        (_partial_settlement_context(), "partial_settlement"),
        (_fraud_context(), "fraud_flag"),
        (_exclusion_context(), "policy_exclusion"),
        # Normal claims should NOT trigger
        ({"claim_amount": 10000, "decision": "approve", "fraud_flag": False, "policy_exclusion": False}, None),
        ({"claim_amount": 250000, "decision": "approve", "fraud_flag": False, "policy_exclusion": False}, None),
        # Edge: high amount + decision = partial → both fire, primary should be high_amount
        (
            {"claim_amount": 600000, "decision": "partial", "fraud_flag": False, "policy_exclusion": False},
            "high_amount",
        ),
        # Edge: fraud + exclusion → both fire
        (
            {"claim_amount": 50000, "decision": "pending", "fraud_flag": True, "policy_exclusion": True},
            "fraud_flag",  # fraud_flag rule comes first in YAML
        ),
    ]

    def test_precision_and_recall(self):
        """Evaluate precision & recall against the golden set."""
        tp = fp = fn = tn = 0

        for ctx, expected_rule in self.GOLDEN_SET:
            result = evaluate_triggers(ctx)

            if expected_rule is not None:
                # Positive case: should trigger
                if result.triggered and any(r["rule_id"] == expected_rule for r in result.matched_rules):
                    tp += 1
                else:
                    fn += 1
                    print(f"FN: expected {expected_rule} but got {[r['rule_id'] for r in result.matched_rules]}")
            else:
                # Negative case: should NOT trigger
                if result.triggered:
                    fp += 1
                    print(f"FP: expected no trigger but got {[r['rule_id'] for r in result.matched_rules]}")
                else:
                    tn += 1

        total = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        print(f"\nTP={tp} FP={fp} FN={fn} TN={tn}  Total={total}")
        print(f"Precision={precision:.2%}  (target >=85%)")
        print(f"Recall={recall:.2%}  (target >=95%)")

        assert precision >= 0.85, f"Precision {precision:.2%} < 85%"
        assert recall >= 0.95, f"Recall {recall:.2%} < 95%"

    def test_all_five_rules_fire(self):
        """Verify that all 5 trigger rules can fire independently."""
        contexts = [
            ("high_amount", _high_amount_context()),
            ("claim_rejection", _rejection_context()),
            ("partial_settlement", _partial_settlement_context()),
            ("fraud_flag", _fraud_context()),
            ("policy_exclusion", _exclusion_context()),
        ]
        for rule_id, ctx in contexts:
            result = evaluate_triggers(ctx)
            assert result.triggered, f"Rule '{rule_id}' did not trigger"
            rule_ids = [r["rule_id"] for r in result.matched_rules]
            assert rule_id in rule_ids, f"Expected '{rule_id}' in {rule_ids}"