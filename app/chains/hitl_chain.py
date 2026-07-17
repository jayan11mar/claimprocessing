"""HITL (Human-In-The-Loop) LCEL chain.

Integrates with ``app/hitl/manager`` to evaluate trigger rules and, when a
rule matches, pause execution by creating a persistent task.  The caller can
then inspect the task via ``/hitl/pending`` and resume via
``/hitl/review/{task_id}``.

Workflow
--------
1. ``_run_hitl`` is called as part of the LCEL chain.
2. If ``ENABLE_HITL`` is ``False`` → pass-through (no-op).
3. If HITL is enabled, build a context dict from the inputs and pass it to
   ``HITLManager.pause()``.
4. If a trigger rule matched, the output dict receives:
   - ``hitl_paused`` = ``True``
   - ``hitl_task_id`` = the persisted task ID
   - ``hitl_status`` = ``"pending"``
   The chain should stop processing and return this as its final output.
5. If no rule matched, the chain proceeds normally.

Resume path
-----------
After a human reviews the task via ``POST /hitl/review/{task_id}``, the
calling service can check ``task.decision`` to determine how to proceed.
This chain does *not* auto-retry; the decision is read from the store.
"""

import logging
from typing import Any, Dict, Optional

from langchain_core.runnables import Runnable, RunnableLambda

from app.config import get_settings
from app.hitl.manager import get_hitl_manager

logger = logging.getLogger(__name__)


def _build_hitl_config() -> Dict[str, Any]:
    """Read HITL-relevant settings and return a config dict."""
    settings = get_settings()
    return {
        "enabled": settings.ENABLE_HITL,
        "rules_path": settings.HITL_RULES_PATH,
    }


def _run_hitl(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate HITL trigger rules and pause if a rule matches.

    Expects ``inputs`` to contain at least ``session_id`` and
    ``user_message``.  Additional fields such as ``claim_amount``,
    ``decision``, ``fraud_flag``, ``policy_exclusion``,
    ``retrieved_chunks``, ``reasoning_trace``, ``confidence``, and
    ``recommendation`` are forwarded to the trigger evaluator.

    Returns the input dict augmented with HITL-status keys.
    """
    config = _build_hitl_config()
    enabled = config.get("enabled", False)

    if not enabled:
        inputs["hitl"] = {"status": "skipped", "reason": "HITL not enabled"}
        return inputs

    # Build context for the trigger evaluator
    context: Dict[str, Any] = {
        "session_id": inputs.get("session_id", ""),
        "user_message": inputs.get("user_message", ""),
        "agent_response": inputs.get("agent_response", ""),
        "claim_amount": inputs.get("claim_amount", 0.0),
        "decision": inputs.get("decision", "pending"),
        "fraud_flag": inputs.get("fraud_flag", False),
        "policy_exclusion": inputs.get("policy_exclusion", False),
        "reasoning_trace": inputs.get("reasoning_trace", ""),
        "confidence": inputs.get("confidence", 0.0),
        "recommendation": inputs.get("recommendation", {}),
        "retrieved_chunks": inputs.get("retrieved_chunks", []),
    }

    manager = get_hitl_manager()
    result = manager.pause(context)

    if result.triggered and result.task is not None:
        task = result.task
        inputs["hitl"] = {
            "status": "pending",
            "paused": True,
            "task_id": task.task_id,
            "rule_id": task.rule_id,
            "rule_reason": task.rule_reason,
        }
        inputs["hitl_paused"] = True
        inputs["hitl_task_id"] = task.task_id
        logger.info(
            "hitl_chain_paused",
            {"task_id": task.task_id, "rule_id": task.rule_id, "session_id": task.session_id},
        )
    else:
        inputs["hitl"] = {
            "status": "passed",
            "paused": False,
            "reason": "No trigger rule matched",
        }
        inputs["hitl_paused"] = False

    return inputs


# ── Public Runnable ─────────────────────────────────────────────────────────

hitl_lcel_chain: Runnable = RunnableLambda(_run_hitl)
"""HITL runnable that pauses the chain when a trigger rule matches.

Usage::

    result = hitl_lcel_chain.invoke({
        "session_id": "sess-123",
        "user_message": "Approve settlement for Rs 600,000",
        "claim_amount": 600000,
        "decision": "pending",
        "confidence": 0.72,
        "recommendation": {"action": "manual_review"},
        "retrieved_chunks": [...],
    })
    if result.get("hitl_paused"):
        print(f"Task {result['hitl_task_id']} is pending review")
"""