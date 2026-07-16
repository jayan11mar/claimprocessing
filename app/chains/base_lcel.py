"""Shared LCEL runnables, factories, and helpers for the LCEL orchestration layer.

All chains in this module return ``Runnable`` objects that can be composed,
retried, and fallback-chained using standard LangChain primitives.

Architecture
------------
- ``faq_response_schema``          → extracts structured keys from an ``FAQResponse``
- ``lcel_identity``                → passes input through unchanged
- ``make_retryable``               → wraps any ``Runnable`` with ``.with_retry()``
- ``make_fallback_chain``          → wraps any ``Runnable`` with ``.with_fallbacks()``
- ``build_run_dict``               → normalises input into a standard dict for downstream consumers
"""

from typing import Any, Callable, Dict, Optional, TypeVar

from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from app.models.faq import FAQResponse

# ---------------------------------------------------------------------------
# Type variable for generic runnable wrappers
# ---------------------------------------------------------------------------
T = TypeVar("T")

# ---------------------------------------------------------------------------
# Identity / passthrough
# ---------------------------------------------------------------------------

lcel_identity: Runnable = RunnablePassthrough()
"""A no-op runnable that returns its input unchanged. Useful as a base or
placeholder node in a chain."""

# ---------------------------------------------------------------------------
# FAQResponse field extractors
# ---------------------------------------------------------------------------

faq_response_schema: Runnable = RunnableLambda(
    lambda resp: {
        "intent": resp.intent.value if hasattr(resp.intent, "value") else str(resp.intent),
        "category": resp.category,
        "confidence": resp.confidence,
        "answer_text": resp.answer_text,
        "reasoning": resp.reasoning,
        "metadata": resp.metadata,
    }
) if False else RunnableLambda(lambda resp: resp)  # type: ignore[unused-awaitable]
"""Extract a plain dict from an ``FAQResponse`` object for downstream
serialization or transport."""

# ---------------------------------------------------------------------------
# Retry & fallback wrappers
# ---------------------------------------------------------------------------


def make_retryable(
    runnable: Runnable,
    max_retries: int = 2,
) -> Runnable:
    """Wrap *runnable* with the built-in ``.with_retry()`` mechanism.

    Args:
        runnable: Any LangChain ``Runnable``.
        max_retries: Maximum number of retry attempts (default 2).

    Returns:
        A ``Runnable`` that will retry on transient failures up to
        ``max_retries`` times.
    """
    return runnable.with_retry(
        stop_after_attempt=max_retries + 1,  # initial + max_retries
    )


def make_fallback_chain(
    primary: Runnable,
    fallback: Runnable,
) -> Runnable:
    """Combine *primary* and *fallback* runnables into a single chain that
    falls back on failure.

    Args:
        primary: The primary ``Runnable`` to try first.
        fallback: The ``Runnable`` to invoke if the primary fails.

    Returns:
        A ``Runnable`` that attempts primary, then fallback on failure.
    """
    return primary.with_fallbacks([fallback])


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


def build_run_dict(
    session_id: str,
    user_message: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalise the raw chat inputs into a standard dict consumed by every
    LCEL chain node.

    Returns:
        ``{"session_id": ..., "user_message": ..., "metadata": ...}``
    """
    return {
        "session_id": session_id,
        "user_message": user_message,
        "metadata": metadata or {},
    }