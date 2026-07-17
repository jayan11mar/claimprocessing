"""
Audit logging for every role-filtered retrieval.

Spec 3.6 (Role-Based RAG):
  - Every role-filtered retrieval must be audited.
  - The audit log records: timestamp, user_id, role, query, requested_k,
    effective_k, allowed_doc_types, pre-filter result count, post-validator
    result count, stripped count, elapsed_ms, and any fallback events.
  - The audit log is written via the structured JSON logger (app/logging/json_logger.py)
    so it integrates with the existing logging pipeline and can be shipped to
    log aggregation systems.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from app.logging.json_logger import get_logger
from app.rbac.models import AnonymousContext, RoleContext

logger = get_logger("app.rbac.audit")

# Audit event types (used as the log message)
AUDIT_EVENT_RETRIEVAL = "rbac_audit_retrieval"
AUDIT_EVENT_CONSENT = "rbac_audit_consent"
AUDIT_EVENT_LEAKAGE = "rbac_audit_leakage_blocked"
AUDIT_EVENT_CLAMP = "rbac_audit_top_k_clamped"


def audit_retrieval(
    role_context: RoleContext | AnonymousContext,
    query: str,
    requested_k: int,
    effective_k: int,
    pre_filter_count: int,
    post_validator_count: int,
    stripped_count: int,
    elapsed_ms: float,
    metadata_filter_used: Optional[Dict[str, Any]] = None,
    fallback_triggered: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Audit a single role-filtered retrieval operation.

    All retrievals that go through the RBAC layer are logged here,
    regardless of whether any chunks were stripped.

    Args:
        role_context: The user's role context.
        query: The original query string.
        requested_k: The top-k originally requested.
        effective_k: The top-k after clamping to the role's maximum.
        pre_filter_count: Number of results before the post-retrieval validator.
        post_validator_count: Number of results after the validator.
        stripped_count: Number of results stripped by the validator.
        elapsed_ms: Total elapsed time for the RBAC layer (filter + validator).
        metadata_filter_used: The metadata filter dict passed to hybrid_retrieve, if any.
        fallback_triggered: Whether a filter fallback was triggered.
        extra: Additional context to include in the audit log.
    """
    payload: Dict[str, Any] = {
        "event": AUDIT_EVENT_RETRIEVAL,
        "user_id": role_context.user_id,
        "role": role_context.role,
        "is_authenticated": role_context.is_authenticated,
        "query": query,
        "requested_k": requested_k,
        "effective_k": effective_k,
        "pre_filter_count": pre_filter_count,
        "post_validator_count": post_validator_count,
        "stripped_count": stripped_count,
        "elapsed_ms": round(elapsed_ms, 3),
        "metadata_filter_used": metadata_filter_used,
        "fallback_triggered": fallback_triggered,
    }
    if extra:
        payload.update(extra)

    logger.info(AUDIT_EVENT_RETRIEVAL, payload)


def audit_consent_event(
    role_context: RoleContext | AnonymousContext,
    action: str,
    doc_type: str,
    query: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Audit an explicit consent event.

    For roles that require explicit consent (e.g. fraud_investigator accessing
    investigation files), every retrieval of restricted content is logged
    as a consent event for compliance purposes.

    Args:
        role_context: The user's role context.
        action: The action taken (e.g. "retrieve_investigation").
        doc_type: The document type that required consent.
        query: The original query string.
        extra: Additional context.
    """
    payload: Dict[str, Any] = {
        "event": AUDIT_EVENT_CONSENT,
        "user_id": role_context.user_id,
        "role": role_context.role,
        "is_authenticated": role_context.is_authenticated,
        "action": action,
        "doc_type": doc_type,
        "query": query,
    }
    if extra:
        payload.update(extra)

    logger.info(AUDIT_EVENT_CONSENT, payload)


def audit_leakage_blocked(
    role_context: RoleContext | AnonymousContext,
    doc_type: str,
    chunk_id: str,
    source_id: str,
    query: str,
    stage: str = "post_retrieval_validator",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Audit a leakage event that was successfully blocked.

    This is called whenever the post-retrieval validator strips a chunk
    that the role should not have access to.  It serves as both a security
    audit trail and a signal for potential misconfiguration.

    Args:
        role_context: The user's role context.
        doc_type: The document type that was blocked.
        chunk_id: The ID of the blocked chunk.
        source_id: The source document ID.
        query: The original query string.
        stage: The stage at which the leakage was blocked.
        extra: Additional context.
    """
    payload: Dict[str, Any] = {
        "event": AUDIT_EVENT_LEAKAGE,
        "user_id": role_context.user_id,
        "role": role_context.role,
        "is_authenticated": role_context.is_authenticated,
        "doc_type": doc_type,
        "chunk_id": chunk_id,
        "source_id": source_id,
        "query": query,
        "stage": stage,
    }
    if extra:
        payload.update(extra)

    logger.warning(AUDIT_EVENT_LEAKAGE, payload)


def audit_top_k_clamp(
    role_context: RoleContext | AnonymousContext,
    requested_k: int,
    clamped_to: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Audit when a requested top-k is clamped to the role's maximum."""
    payload: Dict[str, Any] = {
        "event": AUDIT_EVENT_CLAMP,
        "user_id": role_context.user_id,
        "role": role_context.role,
        "is_authenticated": role_context.is_authenticated,
        "requested_k": requested_k,
        "clamped_to": clamped_to,
    }
    if extra:
        payload.update(extra)

    logger.info(AUDIT_EVENT_CLAMP, payload)