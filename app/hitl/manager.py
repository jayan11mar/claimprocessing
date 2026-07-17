"""HITL (Human-In-The-Loop) manager.

Orchestrates the pause / resume lifecycle:

1. **Pause** — evaluate trigger rules; if any match, serialise the
   recommendation + context into a persistent task and return a pause
   signal.
2. **Resume** — on ``/hitl/review/{task_id}``, update the task decision
   and return the decision so the caller can proceed.
"""

from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.hitl.models import HITLTask, HITLTriggerResult
from app.hitl.store import get_task_store
from app.hitl.triggers import evaluate_triggers, load_rules
from app.logging.json_logger import get_logger

logger = get_logger("app.hitl.manager")


class HITLManager:
    """Manages the HITL pause / resume lifecycle.

    Usage::

        mgr = HITLManager()
        result = mgr.pause(context={...})
        if result.triggered:
            # Return pause signal to the caller
            return {"hitl_paused": True, "task_id": result.task.task_id}
        # else proceed normally
    """

    def __init__(self) -> None:
        self._store = get_task_store()

    # ── Public API ──────────────────────────────────────────────────────

    def pause(self, context: Dict[str, Any]) -> HITLTriggerResult:
        """Evaluate trigger rules and, if matched, create a persistent task.

        Args:
            context: Dict with fields for rule evaluation (``claim_amount``,
                ``decision``, ``fraud_flag``, ``policy_exclusion``, etc.)
                plus context for the task (``retrieved_chunks``,
                ``reasoning_trace``, ``confidence``, ``recommendation``,
                ``user_message``, ``agent_response``, ``session_id``).

        Returns:
            A ``HITLTriggerResult``.  If ``triggered`` is ``True``, the
            ``task`` field contains the persisted task.
        """
        settings = get_settings()
        if not settings.ENABLE_HITL:
            return HITLTriggerResult(triggered=False)

        result = evaluate_triggers(context)
        if result.triggered and result.task is not None:
            persisted = self._store.create_task(result.task)
            result.task = persisted
            logger.info(
                "hitl_paused",
                {
                    "task_id": persisted.task_id,
                    "rule_id": persisted.rule_id,
                    "session_id": persisted.session_id,
                },
            )
        return result

    def resume(
        self,
        task_id: str,
        decision: str,
        comments: Optional[str] = None,
    ) -> Optional[HITLTask]:
        """Resume a paused task by recording the human decision.

        Args:
            task_id: The ID of the task to review.
            decision: ``"approved"`` or ``"rejected"``.
            comments: Optional reviewer comments.

        Returns:
            The updated task, or ``None`` if the task was not found or is
            not pending.
        """
        task = self._store.update_decision(task_id, decision, comments)
        if task is None:
            logger.warning(
                "hitl_resume_not_found",
                {"task_id": task_id, "decision": decision},
            )
        else:
            logger.info(
                "hitl_resumed",
                {
                    "task_id": task_id,
                    "decision": decision,
                    "session_id": task.session_id,
                },
            )
        return task

    def list_pending(self) -> List[HITLTask]:
        """Return all pending tasks."""
        return self._store.list_pending()

    def get_task(self, task_id: str) -> Optional[HITLTask]:
        """Get a single task by ID."""
        return self._store.get_task(task_id)

    def count_pending(self) -> int:
        """Return the number of pending tasks."""
        return self._store.count_pending()


# ── Module-level singleton ────────────────────────────────────────────────

_manager_instance: Optional[HITLManager] = None


def get_hitl_manager() -> HITLManager:
    """Return a singleton HITLManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = HITLManager()
    return _manager_instance


def reset_hitl_manager_singleton() -> None:
    """Reset the singleton (used in tests)."""
    global _manager_instance
    _manager_instance = None