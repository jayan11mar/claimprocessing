"""LCEL chain that wraps the existing tool-call logic from ``AgentChain`` as a
``Runnable``.

This chain does **not** rewrite or replace ``app/chains/agent_chain.py``. It
reuses the same tool functions and the same ``FAQChain`` for intent detection,
preserving existing behaviour while exposing an LCEL interface.
"""

import logging
import time
from typing import Any, Dict, Optional

from langchain_core.runnables import Runnable, RunnableLambda

from app.chains.faq_chain import FAQChain
from app.memory.sqlite_memory import SQLiteMemory
from app.models.faq import FAQIntent, FAQResponse
from app.chains.agent_chain import AgentChain

logger = logging.getLogger(__name__)


def _run_tool_chain(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the existing AgentChain logic for a tool-augmented query.

    Expected input keys:
        - ``session_id`` (str)
        - ``user_message`` (str)
        - ``metadata`` (dict, optional) — may carry ``correlation_id``,
          ``timings``, etc.

    Returns:
        A dict with ``answer_text``, ``intent``, ``confidence``, ``metadata``,
        ``citation``, ``retrieval_trace``, and ``tool_metadata`` keys — all
        shaped to match the ``/chat`` contract.
    """
    session_id: str = inputs.get("session_id", "")
    user_message: str = inputs.get("user_message", "")
    meta: Dict[str, Any] = inputs.get("metadata", {}) or {}

    # Build an AgentChain (with memory singleton) and invoke it
    memory = SQLiteMemory()
    agent = AgentChain(memory=memory)

    context = {
        "correlation_id": meta.get("correlation_id"),
        "timings": {"llm_ms": 0, "tools": []},
    }

    faq_response: FAQResponse = agent.invoke(
        session_id=session_id,
        user_message=user_message,
        context=context,
    )

    # Build output that matches the /chat ChatResponse contract
    timings: Dict[str, Any] = context.get("timings", {})
    tool_timings = timings.get("tools", [])
    is_tool_augmented = bool(tool_timings)

    citations = []
    retrieval_trace = []
    tool_metadata = {}

    if isinstance(faq_response.metadata, dict):
        citations = faq_response.metadata.get("citations", [])
        retrieval_trace = faq_response.metadata.get("retrieval_trace", [])
        tool_metadata = {
            "tool": faq_response.metadata.get("tool"),
            "tool_output": faq_response.metadata.get("tool_output"),
            "error": faq_response.metadata.get("error"),
            "hitl_paused": faq_response.metadata.get("hitl_paused"),
            "hitl_task_id": faq_response.metadata.get("hitl_task_id"),
            "hitl_rule_id": faq_response.metadata.get("hitl_rule_id"),
            "hitl_reason": faq_response.metadata.get("hitl_reason"),
        }

    # Flatten timings into chain_metadata-like structure
    chain_metadata = {
        "llm_ms": timings.get("llm_ms", 0),
        "tool_timings": tool_timings,
        "is_tool_augmented": is_tool_augmented,
    }

    return {
        "answer_text": faq_response.answer_text,
        "intent": faq_response.intent.value if hasattr(faq_response.intent, "value") else str(faq_response.intent),
        "category": faq_response.category,
        "confidence": faq_response.confidence,
        "reasoning": faq_response.reasoning,
        "metadata": faq_response.metadata,
        "citations": citations,
        "retrieval_trace": retrieval_trace,
        "tool_metadata": tool_metadata,
        "chain_metadata": chain_metadata,
    }


# ── Public Runnable ─────────────────────────────────────────────────────────

tool_lcel_chain: Runnable = RunnableLambda(_run_tool_chain)
"""A ``Runnable`` that wraps the existing ``AgentChain`` (FAQ + tool dispatch).

Usage::

    result = tool_lcel_chain.invoke({
        "session_id": "sess-123",
        "user_message": "Check status of claim C1001",
    })
    print(result["answer_text"])
"""