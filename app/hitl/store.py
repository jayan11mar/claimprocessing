"""SQLite-backed persistent task store for HITL tasks."""

import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.hitl.models import HITLTask
from app.logging.json_logger import get_logger

logger = get_logger("app.hitl.store")

# Default path — overridden by HITL_STORE_PATH env var
DEFAULT_HITL_STORE_PATH = "data/hitl_tasks.db"


class HITLTaskStore:
    """Persistent SQLite-backed store for HITL tasks.

    Thread-safe via a per-instance lock.  Tasks are serialised as JSON
    rows so the full pydantic model (including nested lists / dicts) is
    round-trippable.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or os.environ.get(
            "HITL_STORE_PATH", DEFAULT_HITL_STORE_PATH
        )
        self._lock = threading.Lock()
        self._init_db()

    # ── schema ──────────────────────────────────────────────────────────
    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hitl_tasks (
                    task_id         TEXT PRIMARY KEY,
                    session_id      TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL,
                    status          TEXT NOT NULL DEFAULT 'pending',
                    rule_id         TEXT NOT NULL,
                    rule_reason     TEXT NOT NULL DEFAULT '',
                    payload_json    TEXT NOT NULL DEFAULT '{}',
                    decision        TEXT,
                    reviewer_comments TEXT,
                    reviewed_at     TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hitl_status ON hitl_tasks(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hitl_session ON hitl_tasks(session_id)"
            )
            conn.commit()

    # ── CRUD ────────────────────────────────────────────────────────────

    def create_task(self, task: HITLTask) -> HITLTask:
        """Insert a new task into the store."""
        payload = self._serialise_payload(task)
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO hitl_tasks
                    (task_id, session_id, created_at, updated_at, status,
                     rule_id, rule_reason, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.session_id,
                    task.created_at.isoformat(),
                    task.updated_at.isoformat(),
                    task.status,
                    task.rule_id,
                    task.rule_reason,
                    payload,
                ),
            )
            conn.commit()
        logger.info("hitl_task_created", {"task_id": task.task_id, "rule_id": task.rule_id})
        return task

    def get_task(self, task_id: str) -> Optional[HITLTask]:
        """Retrieve a single task by ID."""
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM hitl_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def list_pending(self) -> List[HITLTask]:
        """Return all tasks with status 'pending', ordered by creation time."""
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM hitl_tasks WHERE status = 'pending' ORDER BY created_at ASC"
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def list_all(self) -> List[HITLTask]:
        """Return all tasks ordered by creation time (newest first)."""
        with self._lock, sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM hitl_tasks ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def update_decision(
        self,
        task_id: str,
        decision: str,
        reviewer_comments: Optional[str] = None,
    ) -> Optional[HITLTask]:
        """Approve or reject a pending task.

        Returns the updated task, or ``None`` if the task was not found or
        is not pending.
        """
        now = datetime.utcnow().isoformat()
        with self._lock, sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE hitl_tasks
                SET status = ?,
                    decision = ?,
                    reviewer_comments = ?,
                    reviewed_at = ?,
                    updated_at = ?
                WHERE task_id = ? AND status = 'pending'
                """,
                (decision, decision, reviewer_comments, now, now, task_id),
            )
            conn.commit()
            # If no rows were updated, the task was not pending or not found
            if cursor.rowcount == 0:
                return None
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM hitl_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        task = self._row_to_task(row)
        logger.info(
            "hitl_task_reviewed",
            {"task_id": task_id, "decision": decision},
        )
        return task

    def count_pending(self) -> int:
        """Return the number of pending tasks."""
        with self._lock, sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM hitl_tasks WHERE status = 'pending'"
            ).fetchone()
        return row[0] if row else 0

    # ── helpers ─────────────────────────────────────────────────────────

    def _serialise_payload(self, task: HITLTask) -> str:
        """Turn the rich fields of a HITLTask into a JSON string for storage."""
        return json.dumps(
            {
                "retrieved_chunks": task.retrieved_chunks,
                "reasoning_trace": task.reasoning_trace,
                "confidence": task.confidence,
                "recommendation": task.recommendation,
                "user_message": task.user_message,
                "agent_response": task.agent_response,
            }
        )

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> HITLTask:
        payload = json.loads(row["payload_json"] if row["payload_json"] else "{}")
        task = HITLTask(
            task_id=row["task_id"],
            session_id=row["session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            status=row["status"],
            rule_id=row["rule_id"],
            rule_reason=row["rule_reason"] if row["rule_reason"] else "",
            retrieved_chunks=payload.get("retrieved_chunks", []),
            reasoning_trace=payload.get("reasoning_trace", ""),
            confidence=payload.get("confidence", 0.0),
            recommendation=payload.get("recommendation", {}),
            user_message=payload.get("user_message", ""),
            agent_response=payload.get("agent_response", ""),
            decision=row["decision"],
            reviewer_comments=row["reviewer_comments"],
            reviewed_at=(
                datetime.fromisoformat(row["reviewed_at"])
                if row["reviewed_at"]
                else None
            ),
        )
        return task


# ── Module-level singleton (for convenience) ──────────────────────────────

_store_instance: Optional[HITLTaskStore] = None
_store_lock = threading.Lock()


def get_task_store() -> HITLTaskStore:
    """Return a singleton HITLTaskStore instance."""
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = HITLTaskStore()
    return _store_instance


def reset_task_store_singleton() -> None:
    """Reset the singleton (used in tests)."""
    global _store_instance
    with _store_lock:
        _store_instance = None