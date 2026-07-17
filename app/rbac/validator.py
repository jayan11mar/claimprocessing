"""
Post-retrieval validator — guarantees no restricted chunk leaks through the
re-ranker or cache.

Spec 3.6 (Role-Based RAG):
  - After hybrid retrieval (including re-ranking and any caching layer), the
    post-retrieval validator inspects every result chunk and strips any that
    the authenticated user's role is not permitted to see.
  - This is the **last line of defence** against role leakage.  Even if the
    pre-retrieval filter is bypassed (e.g. due to a fallback path, a cache
    hit, or a future code change), the validator ensures 0% leakage.

Threshold: 0% leakage (mandatory).
Pitfall: role leakage through re-ranker — the validator runs *after* the
         re-ranker, so any chunk that the re-ranker promotes is still subject
         to validation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from app.logging.json_logger import get_logger
from app.rbac.models import (
    AnonymousContext,
    PermissionMatrix,
    RoleContext,
)

logger = get_logger("app.rbac.validator")


def validate_retrieval_results(
    results: List[Dict[str, Any]],
    role_context: RoleContext | AnonymousContext,
    query: str = "",
) -> List[Dict[str, Any]]:
    """Post-retrieval validation: strip any result whose doc_type is not
    allowed for the given role.

    This function is called **after** the hybrid retriever and re-ranker
    have produced their final ranked list.  It guarantees that no restricted
    chunk leaks through to the caller.

    Args:
        results: The list of result dicts from the retrieval pipeline
                 (each dict must contain a ``chunk`` key with a ``doc_type``
                 attribute, or a ``doc_type`` key at the top level).
        role_context: The authenticated (or anonymous) user context.
        query: The original query string (used for audit logging).

    Returns:
        Filtered list containing only results whose doc_type is in the
        role's allowed set.

    Note:
        Each result dict is expected to have either:
          - ``result["chunk"].doc_type`` (Chunk object), or
          - ``result["doc_type"]`` (string, for serialized results).
        If neither is found, the result is **kept** (fail-open for
        compatibility), but a warning is logged.
    """
    start_ns = time.perf_counter_ns()

    # Unrestricted role (e.g. service role when RBAC is disabled) — pass all
    if role_context.has_unrestricted_access:
        elapsed_ms = _elapsed_ms(start_ns)
        logger.info(
            "rbac_validator_unrestricted_pass",
            {
                "role": role_context.role,
                "user_id": role_context.user_id,
                "query": query,
                "result_count": len(results),
                "elapsed_ms": elapsed_ms,
            },
        )
        return results

    allowed = role_context.allowed_doc_types

    if not allowed:
        # Role has no allowed doc_types — strip everything
        elapsed_ms = _elapsed_ms(start_ns)
        logger.warning(
            "rbac_validator_no_allowed_types_stripped_all",
            {
                "role": role_context.role,
                "user_id": role_context.user_id,
                "query": query,
                "original_count": len(results),
                "final_count": 0,
                "elapsed_ms": elapsed_ms,
            },
        )
        return []

    validated: List[Dict[str, Any]] = []
    stripped_count = 0

    for result in results:
        doc_type = _extract_doc_type(result)
        if doc_type is None:
            # Cannot determine doc_type — keep the result but warn
            logger.warning(
                "rbac_validator_unknown_doc_type",
                {
                    "role": role_context.role,
                    "user_id": role_context.user_id,
                    "query": query,
                    "chunk_id": result.get("chunk_id", "?"),
                },
            )
            validated.append(result)
        elif doc_type in allowed:
            validated.append(result)
        else:
            stripped_count += 1
            logger.info(
                "rbac_validator_stripped_chunk",
                {
                    "role": role_context.role,
                    "user_id": role_context.user_id,
                    "query": query,
                    "doc_type": doc_type,
                    "chunk_id": result.get("chunk_id", "?"),
                    "source_id": result.get("source_id", "?"),
                },
            )

    elapsed_ms = _elapsed_ms(start_ns)

    if stripped_count > 0:
        logger.warning(
            "rbac_validator_stripped_summary",
            {
                "role": role_context.role,
                "user_id": role_context.user_id,
                "query": query,
                "original_count": len(results),
                "stripped_count": stripped_count,
                "final_count": len(validated),
                "elapsed_ms": elapsed_ms,
            },
        )
    else:
        logger.info(
            "rbac_validator_passed",
            {
                "role": role_context.role,
                "user_id": role_context.user_id,
                "query": query,
                "result_count": len(validated),
                "elapsed_ms": elapsed_ms,
            },
        )

    return validated


def _extract_doc_type(result: Dict[str, Any]) -> Optional[str]:
    """Extract the doc_type from a result dict.

    Tries, in order:
      1. ``result["chunk"].doc_type`` (Chunk object attribute)
      2. ``result["doc_type"]`` (top-level string key)
      3. ``result.get("chunk", {}).get("doc_type")`` (nested dict)
    """
    chunk = result.get("chunk")
    if chunk is not None:
        if hasattr(chunk, "doc_type"):
            return chunk.doc_type
        if isinstance(chunk, dict):
            return chunk.get("doc_type")

    doc_type = result.get("doc_type")
    if doc_type is not None:
        return doc_type

    return None


def _elapsed_ms(start_ns: int) -> float:
    """Return elapsed time in milliseconds since *start_ns*."""
    return round((time.perf_counter_ns() - start_ns) / 1_000_000, 3)