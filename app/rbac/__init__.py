"""
RBAC (Role-Based Access Control) module for Spec 3.6 (Role-Based RAG).

Components:
  - auth:       JWT middleware (HS256 via JWT_SECRET); decode token -> role + permissions
  - models:     Role-permission matrix (role -> allowed doc_type / metadata filters)
  - filter:     Pre-retrieval metadata filter (injected before hybrid retrieval)
  - validator:  Post-retrieval validator (guarantees 0% leakage after re-ranker)
  - audit:      Audit logging for every role-filtered retrieval
"""

from app.rbac.auth import (
    JWTConfig,
    create_access_token,
    decode_access_token,
    extract_role_context_from_request,
    get_service_role_context,
    SERVICE_ROLE_NAME,
    SERVICE_ROLE_PERMISSIONS,
)
from app.rbac.models import (
    Role,
    RolePermissions,
    RoleContext,
    AnonymousContext,
    PermissionMatrix,
)
from app.rbac.filter import build_role_metadata_filter, clamp_top_k
from app.rbac.validator import validate_retrieval_results
from app.rbac.audit import (
    audit_retrieval,
    audit_consent_event,
    audit_leakage_blocked,
    audit_top_k_clamp,
    AUDIT_EVENT_RETRIEVAL,
    AUDIT_EVENT_CONSENT,
    AUDIT_EVENT_LEAKAGE,
    AUDIT_EVENT_CLAMP,
)

__all__ = [
    # auth
    "JWTConfig",
    "create_access_token",
    "decode_access_token",
    "extract_role_context_from_request",
    "get_service_role_context",
    "SERVICE_ROLE_NAME",
    "SERVICE_ROLE_PERMISSIONS",
    # models
    "Role",
    "RolePermissions",
    "RoleContext",
    "AnonymousContext",
    "PermissionMatrix",
    # filter
    "build_role_metadata_filter",
    "clamp_top_k",
    # validator
    "validate_retrieval_results",
    # audit
    "audit_retrieval",
    "audit_consent_event",
    "audit_leakage_blocked",
    "audit_top_k_clamp",
    "AUDIT_EVENT_RETRIEVAL",
    "AUDIT_EVENT_CONSENT",
    "AUDIT_EVENT_LEAKAGE",
    "AUDIT_EVENT_CLAMP",
]