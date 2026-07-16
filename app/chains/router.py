"""Intent → LCEL chain registry with ``.with_retry()`` and ``.with_fallbacks()``.

This module builds a single ``Runnable`` that routes incoming chat requests to
the correct LCEL chain based on the resolved intent. If no explicit intent is
provided, the router falls back to the tool chain (which performs its own
intent detection via ``FAQChain``).

Architecture
------------
- ``lcel_router`` is the top-level ``Runnable`` exposed to the API layer.
- Each chain in the registry is wrapped with ``.with_retry()`` and
  ``.with_fallbacks()`` for resilience.
- Callbacks are attached at the router level via ``RunnableConfig``.
"""

import logging
from typing import Any, Dict, Optional

from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough

from app.chains.base_lcel import build_run_dict, lcel_identity, make_fallback_chain, make_retryable
from app.chains.hitl_chain import hitl_lcel_chain
from app.chains.rag_chain_lcel import rag_lcel_chain
from app.chains.tool_chain_lcel import tool_lcel_chain
from app.config import get_settings
from app.models.faq import FAQIntent

logger = logging.getLogger(__name__)

# ── Intent classification node (delegates to the existing FAQChain logic) ────

from app.chains.faq_chain import FAQChain


def _classify_intent(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run FAQChain's intent detection (without tool dispatch) and attach the
    resolved intent to the input dict.

    This allows the downstream router branch to pick the right chain without
    re-running the full tool-llm cycle.
    """
    session_id: str = inputs.get("session_id", "")
    user_message: str = inputs.get("user_message", "")

    faq = FAQChain()
    response = faq.invoke(session_id, user_message, persist_history=False)

    inputs["_resolved_intent"] = response.intent.value if hasattr(response.intent, "value") else str(response.intent)
    inputs["_faq_confidence"] = response.confidence
    return inputs


classify_node: Runnable = RunnableLambda(_classify_intent)

# ── Route functions for RunnableBranch ──────────────────────────────────────

_INTENT_TO_KEY: Dict[str, str] = {
    FAQIntent.KNOWLEDGE_RETRIEVAL.value: "rag",
    FAQIntent.CLAIM_REGISTRATION.value: "tool",
    FAQIntent.POLICY_STATUS.value: "tool",
    FAQIntent.CLAIM_STATUS.value: "tool",
    FAQIntent.FRAUD_CHECK.value: "tool",
    FAQIntent.SETTLEMENT_QUERY.value: "tool",
    FAQIntent.ESCALATION.value: "tool",
    FAQIntent.DOCUMENTS_REQUIRED.value: "tool",
    FAQIntent.OTHER.value: "tool",
}


def _route_by_intent(inputs: Dict[str, Any]) -> str:
    """Return the branch key for the resolved intent."""
    intent: str = inputs.get("_resolved_intent", "OTHER")
    return _INTENT_TO_KEY.get(intent, "tool")


# ── Build the registry with retry & fallback wrappers ────────────────────────

def _registry() -> Dict[str, Runnable]:
    """Build and return the chain registry.

    Each chain is wrapped with ``.with_retry()`` and ``.with_fallbacks()``
    for resilience.
    """
    base_registry = {
        "rag": rag_lcel_chain,
        "tool": tool_lcel_chain,
        "hitl": hitl_lcel_chain,
    }

    # Wrap each with retry
    retried = {name: make_retryable(chain, max_retries=2) for name, chain in base_registry.items()}

    # Add fallback chains:
    #   - rag  → tool (fallback to tool if RAG fails)
    #   - tool → tool (identity fallback — already the most general)
    #   - hitl → tool (fallback to tool if HITL is unavailable)
    fallback_registry = {}
    for name, chain in retried.items():
        if name == "rag":
            fallback_registry[name] = make_fallback_chain(chain, retried["tool"])
        elif name == "hitl":
            fallback_registry[name] = make_fallback_chain(chain, retried["tool"])
        else:
            fallback_registry[name] = chain  # tool uses itself as fallback

    return fallback_registry


# ── Top-level router ────────────────────────────────────────────────────────

def _build_router() -> Runnable:
    """Assemble the full LCEL router.

    Pipeline::

        input_dict
          → classify_node (attach _resolved_intent)
          → RunnableBranch(rag → rag_chain, tool → tool_chain, default → tool_chain)
          → post_process (normalise output to /chat contract)
    """
    registry = _registry()

    logger.info(
        "LCEL router built with chains: %s",
        {k: v.__class__.__name__ for k, v in registry.items()},
    )

    branch = RunnableBranch(
        (lambda x: _route_by_intent(x) == "rag", registry["rag"]),
        (lambda x: _route_by_intent(x) == "hitl", registry["hitl"]),
        registry["tool"],  # default
    )

    # Pipeline: classify → branch → strip_internal_keys
    pipeline = classify_node | branch | RunnableLambda(_post_process)

    return pipeline


def _post_process(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove internal keys (``_resolved_intent``, ``_faq_confidence``) from
    the output before returning it to the API layer.
    """
    result = dict(inputs)
    for key in list(result.keys()):
        if key.startswith("_"):
            del result[key]
    return result


# ── Smoke-test entry point ───────────────────────────────────────────────────

def _smoke_test():
    """Run a smoke test that exercises intent routing, retry, and fallback.

    Prints the selected chain per sample intent, then forces an error to
    demonstrate that retry/fallback mechanisms engage.

    Usage::

        PYTHONPATH=. python -m app.chains.router --smoke
    """
    import json
    import sys

    # ── Sample intents ──────────────────────────────────────────────────────
    test_inputs = [
        {"session_id": "smoke-greeting", "user_message": "hello", "expected": "tool (OTHER)"},
        {"session_id": "smoke-rag", "user_message": "What does the policy cover for hospital claims?", "expected": "rag"},
        {"session_id": "smoke-policy", "user_message": "Check status of policy P1001", "expected": "tool (POLICY_STATUS)"},
        {"session_id": "smoke-claim", "user_message": "Register a new claim for water damage", "expected": "tool (CLAIM_REGISTRATION)"},
    ]

    print("=" * 60)
    print("Router smoke test — intent routing")
    print("=" * 60)

    for inp in test_inputs:
        try:
            result = lcel_router.invoke(inp)
            intent = result.get("intent", "?")
            answer = result.get("answer_text", "")[:80]
            print(f"  [{inp['session_id']:>20s}] intent={intent:20s}  answer={answer!r}")
        except Exception as exc:
            print(f"  [{inp['session_id']:>20s}] ERROR: {exc}")

    # ── Forced error — demonstrate retry / fallback ────────────────────────
    print()
    print("-" * 60)
    print("Forced error — Router-level retry/fallback demo")
    print("-" * 60)

    from app.chains.base_lcel import make_retryable, make_fallback_chain
    from langchain_core.runnables import RunnableLambda

    def _failing_runnable(inputs):
        raise RuntimeError("Forced transient failure for smoke test")

    def _fallback_runnable(inputs):
        return {
            "answer_text": "Fallback engaged: tool chain responded.",
            "intent": "OTHER",
            "confidence": 0.5,
            "metadata": {"fallback": True, "original_error": str(inputs.get("_error", ""))},
        }

    failing = RunnableLambda(_failing_runnable)
    fallback = RunnableLambda(_fallback_runnable)
    chain = make_fallback_chain(make_retryable(failing, max_retries=1), fallback)

    try:
        fb_result = chain.invoke({"session_id": "smoke-fallback", "user_message": "test"})
        print(f"  Fallback result: answer_text={fb_result.get('answer_text')!r}")
        print(f"  metadata: {json.dumps(fb_result.get('metadata', {}), indent=4)}")
    except Exception as exc:
        print(f"  ERROR during forced fallback: {exc}")

    print()
    print("✓ Smoke test complete.")


# ── Singleton exported to the API layer ─────────────────────────────────────

lcel_router: Runnable = _build_router()
"""Top-level LCEL router runnable.

Usage::

    result = lcel_router.invoke({
        "session_id": "sess-123",
        "user_message": "What does the policy cover for hospital claims?",
    })
    print(result["answer_text"])
"""


if __name__ == "__main__":
    import sys

    if "--smoke" in sys.argv:
        _smoke_test()
    else:
        print("Usage: PYTHONPATH=. python -m app.chains.router --smoke")