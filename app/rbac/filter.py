"""
Pre-retrieval metadata filter — injected before hybrid retrieval.

Spec 3.6 (Role-Based RAG):
  - Before the hybrid retriever runs, the pre-retrieval filter consults the
    authenticated user's RoleContext and builds a metadata_filter dict that
    restricts retrieval to only the doc_types the role is permitted to see.
  - The filter is injected as the ``metadata_filter`` parameter of
    ``hybrid_retrieve()``.
  - If the role has no allowed doc_types, retrieval is short-circuited
    (empty result) to avoid wasted computation.

NFR: Filtering overhead < 200ms (the filter itself is O(1) — just a dict
     construction — so this is trivially satisfied).
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

logger = get_logger("app.rbac.filter")


def build_role_metadata_filter(
    role_context: RoleContext | AnonymousContext,
    query: str = "",
    requested_k: int = 5,
) -> Optional[Dict[str, Any]]:
    """Build a metadata filter dict that restricts retrieval to the role's
    allowed doc_types.

    The returned dict is suitable for passing as ``metadata_filter`` to
    ``hybrid_retrieve()``.

    Args:
        role_context: The authenticated (or anonymous) user context.
        query: The original query string (used for audit logging).
        requested_k: The top-k the user requested (will be clamped to
                     the role's max_retrieval_k).

    Returns:
        A dict like ``{"doc_type": "policy_wording"}`` if the role has
        exactly one allowed doc_type, or ``None`` if the role has no
        restrictions (all doc_types allowed).  Returns an empty dict
        ``{}`` if the role has zero allowed doc_types (short-circuit).

    Note:
        - A role with **unrestricted access** (``has_unrestricted_access``)
          returns ``None`` (no filter), allowing all doc_types through.
        - The hybrid retriever's ``_apply_metadata_filter`` does an exact
          match on each key-value pair.  Since a chunk has a single doc_type,
          we can only filter on one doc_type at a time with the current
          implementation.  For roles with multiple allowed doc_types, we
          return ``None`` (no filter) and rely on the **post-retrieval
          validator** to strip any disallowed chunks after retrieval.
    """
    start_ns = time.perf_counter_ns()

    # Unrestricted role (e.g. service role when RBAC is disabled) — no filter
    if role_context.has_unrestricted_access:
        elapsed_ms = _elapsed_ms(start_ns)
        logger.info(
            "rbac_filter_unrestricted",
            {
                "role": role_context.role,
                "user_id": role_context.user_id,
                "query": query,
                "elapsed_ms": elapsed_ms,
            },
        )
        return None

    allowed = role_context.allowed_doc_types
    max_k = role_context.max_k

    # Clamp requested_k to the role's maximum
    effective_k = min(requested_k, max_k) if max_k > 0 else requested_k

    # Short-circuit: no allowed doc_types → empty result
    if not allowed:
        elapsed_ms = _elapsed_ms(start_ns)
        logger.warning(
            "rbac_filter_no_allowed_types",
            {
                "role": role_context.role,
                "user_id": role_context.user_id,
                "query": query,
                "requested_k": requested_k,
                "effective_k": effective_k,
                "elapsed_ms": elapsed_ms,
            },
        )
        return {}  # signals "return empty"

    # If the role has exactly one allowed doc_type, we can filter at the
    # metadata level (exact match in _apply_metadata_filter).
    if len(allowed) == 1:
        metadata_filter = {"doc_type": allowed[0]}
        elapsed_ms = _elapsed_ms(start_ns)
        logger.info(
            "rbac_filter_single_type",
            {
                "role": role_context.role,
                "user_id": role_context.user_id,
                "query": query,
                "allowed_doc_type": allowed[0],
                "effective_k": effective_k,
                "elapsed_ms": elapsed_ms,
            },
        )
        return metadata_filter

    # Multiple allowed doc_types: we cannot express an OR filter with the
    # current _apply_metadata_filter (which does AND).  Return None to
    # skip pre-filtering; the post-retrieval validator will enforce the
    # restriction.
    elapsed_ms = _elapsed_ms(start_ns)
    logger.info(
        "rbac_filter_multi_type_skip",
        {
            "role": role_context.role,
            "user_id": role_context.user_id,
            "query": query,
            "allowed_doc_types": allowed,
            "effective_k": effective_k,
            "reason": "multiple allowed types — post-retrieval validator will enforce",
            "elapsed_ms": elapsed_ms,
        },
    )
    return None


def clamp_top_k(
    role_context: RoleContext | AnonymousContext,
    requested_k: int,
) -> int:
    """Clamp the requested top-k to the role's maximum allowed value.

    Args:
        role_context: The user's role context.
        requested_k: The top-k the user requested.

    Returns:
        The clamped value (never exceeds the role's max_retrieval_k).
    """
    max_k = role_context.max_k
    if max_k > 0 and requested_k > max_k:
        logger.info(
            "rbac_clamp_top_k",
            {
                "role": role_context.role,
                "user_id": role_context.user_id,
                "requested_k": requested_k,
                "max_k": max_k,
                "clamped_to": max_k,
            },
        )
        return max_k
    return requested_k


def _elapsed_ms(start_ns: int) -> float:
    """Return elapsed time in milliseconds since *start_ns*."""
    return round((time.perf_counter_ns() - start_ns) / 1_000_000, 3)