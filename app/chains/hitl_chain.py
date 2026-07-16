"""Placeholder HITL (Human-In-The-Loop) LCEL chain.

This module provides the ``hitl_lcel_chain`` runnable that will, in a future
iteration, integrate human review steps into the LCEL orchestration layer.

Current behaviour
-----------------
- Returns a pass-through response that delegates to the fallback path.
- The ``hitl_config`` dict can be extended with review rules, thresholds, etc.
"""

import logging
from typing import Any, Dict, Optional

from langchain_core.runnables import Runnable, RunnableLambda

from app.config import get_settings

logger = logging.getLogger(__name__)


def _build_hitl_config() -> Dict[str, Any]:
    """Read HITL-relevant settings and return a config dict.

    Extends gracefully: when ``ENABLE_HITL`` is ``False`` the chain acts as a
    no-op pass-through.
    """
    settings = get_settings()
    return {
        "enabled": settings.ENABLE_HITL,
        "rules_path": settings.HITL_RULES_PATH,
    }


def _run_hitl(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder HITL execution.

    When ``ENABLE_HITL`` is ``False`` (default), this function simply returns
    the input unchanged — effectively acting as a no-op pass-through node.

    When ``ENABLE_HITL`` is ``True`` (future), this node will:
    1. Evaluate input against HITL rules from ``config/hitl_rules.yaml``.
    2. If a rule matches, pause execution and notify the human-review queue.
    3. Resume with the human decision attached to the run context.

    Args:
        inputs: A dict with keys ``session_id``, ``user_message``, ``metadata``,
            and any intermediate chain output.

    Returns:
        The same dict (pass-through) with an added ``hitl`` key indicating
        the current HITL status.
    """
    config = _build_hitl_config()
    enabled = config.get("enabled", False)

    if not enabled:
        inputs["hitl"] = {"status": "skipped", "reason": "HITL not enabled"}
        return inputs

    # ── Future: actual HITL logic goes here ──────────────────────────────
    inputs["hitl"] = {
        "status": "placeholder",
        "reason": "HITL enabled but chain not yet implemented",
        "config": config,
    }

    logger.info("hitl_placeholder invoked for session %s", inputs.get("session_id"))

    # Pass through — do not modify the chain output yet.
    return inputs


# ── Public Runnable ─────────────────────────────────────────────────────────

hitl_lcel_chain: Runnable = RunnableLambda(_run_hitl)
"""Placeholder HITL runnable.

Currently acts as a pass-through. In future it will integrate with the
``app/hitl/`` module to pause and resume chains based on human review.

Usage::

    result = hitl_lcel_chain.invoke({
        "session_id": "sess-123",
        "user_message": "Approve settlement for C1001",
    })
    assert result["hitl"]["status"] == "skipped"
"""