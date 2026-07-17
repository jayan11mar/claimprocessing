"""
JWT middleware (HS256 via JWT_SECRET); decode token -> role + permissions.

Spec 3.6 (Role-Based RAG):
  - JWT tokens encode the authenticated user's role.
  - The middleware decodes the token on every request, resolves permissions
    from the PermissionMatrix, and attaches a RoleContext to request.state.
  - If ENABLE_RBAC is False, a SERVICE_ROLE with full access is used so that
    existing tests continue to work unchanged.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import jwt as pyjwt

from app.config import get_settings
from app.logging.json_logger import get_logger
from app.rbac.models import (
    AnonymousContext,
    PermissionMatrix,
    RoleContext,
    RolePermissions,
)

logger = get_logger("app.rbac.auth")

# ── Constants ────────────────────────────────────────────────────────────────

# Service role for when RBAC is disabled — full access to all doc_types.
# This ensures existing tests that don't set ENABLE_RBAC continue to work.
SERVICE_ROLE_NAME = "service"
SERVICE_ROLE_PERMISSIONS = RolePermissions(
    display_name="Service Account",
    description="Full-access service role used when RBAC is disabled",
    allowed_doc_types=[],  # empty = all types allowed (validated by filter/validator logic)
    allowed_insurance_types=None,
    max_retrieval_k=100,
    requires_explicit_consent=False,
)


# ── JWT config ───────────────────────────────────────────────────────────────


@dataclass
class JWTConfig:
    """JWT configuration loaded from config/roles.yaml."""

    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    default_role: str = "claims_processor"
    header_prefix: str = "Bearer "


def _load_jwt_config() -> JWTConfig:
    """Load JWT config from the permission matrix singleton."""
    matrix = PermissionMatrix.get_instance()
    return matrix.jwt_config


def _get_jwt_secret() -> str:
    """Return the JWT signing secret from the environment.

    Uses JWT_SECRET_KEY env var.  Falls back to a dev-only secret.
    In production, set JWT_SECRET_KEY to a strong 32+ byte random value.
    """
    return os.getenv("JWT_SECRET_KEY", "dev-secret-change-in-production")


# ── Token creation / decoding ────────────────────────────────────────────────


def create_access_token(
    subject: str,
    role: str,
    extra_claims: Optional[Dict[str, Any]] = None,
    expires_minutes: Optional[int] = None,
) -> str:
    """Create a signed JWT access token (HS256).

    Args:
        subject: The user identifier (e.g. user_id or email).
        role: The user's role (must be a valid Role enum value).
        extra_claims: Optional additional claims to embed.
        expires_minutes: Token expiry in minutes (default from config).

    Returns:
        Encoded JWT string.
    """
    jwt_cfg = _load_jwt_config()
    secret = _get_jwt_secret()
    expiry = expires_minutes or jwt_cfg.access_token_expire_minutes

    payload: Dict[str, Any] = {
        "sub": subject,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + expiry * 60,
    }
    if extra_claims:
        payload.update(extra_claims)

    token = pyjwt.encode(payload, secret, algorithm=jwt_cfg.algorithm)
    return token


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a JWT access token (HS256).

    Args:
        token: The raw JWT string (without the "Bearer " prefix).

    Returns:
        Decoded payload dict, or None if the token is invalid/expired.
    """
    jwt_cfg = _load_jwt_config()
    secret = _get_jwt_secret()

    try:
        payload = pyjwt.decode(token, secret, algorithms=[jwt_cfg.algorithm])
        return payload
    except pyjwt.ExpiredSignatureError:
        logger.warning("jwt_expired")
        return None
    except pyjwt.InvalidTokenError as exc:
        logger.warning("jwt_invalid", {"error": str(exc)})
        return None


# ── Request context extraction ───────────────────────────────────────────────


def extract_role_context_from_request(request: Any) -> RoleContext | AnonymousContext:
    """Extract a RoleContext from a FastAPI request.

    Looks for the ``Authorization`` header, decodes the JWT, and returns
    a ``RoleContext``.  If no valid token is found, returns an
    ``AnonymousContext`` with the default role.

    This is designed to be called from a FastAPI middleware or dependency.
    """
    auth_header = request.headers.get("Authorization", "")
    jwt_cfg = _load_jwt_config()

    if not auth_header.startswith(jwt_cfg.header_prefix):
        logger.info("auth_no_token", {"path": request.url.path})
        return AnonymousContext()

    token = auth_header[len(jwt_cfg.header_prefix):].strip()
    if not token:
        logger.info("auth_empty_token", {"path": request.url.path})
        return AnonymousContext()

    payload = decode_access_token(token)
    if payload is None:
        logger.warning("auth_invalid_token", {"path": request.url.path})
        return AnonymousContext()

    role = payload.get("role", jwt_cfg.default_role)
    user_id = payload.get("sub", "unknown")

    logger.info(
        "auth_context_extracted",
        {"user_id": user_id, "role": role, "path": request.url.path},
    )

    return RoleContext(
        user_id=user_id,
        role=role,
        token_payload=payload,
    )


def get_service_role_context() -> RoleContext:
    """Return a RoleContext with the full-access service role.

    Used when ENABLE_RBAC is False — the service role has unrestricted
    access to all document types, so existing tests and operations
    continue to work without modification.
    """
    return RoleContext(
        user_id=SERVICE_ROLE_NAME,
        role=SERVICE_ROLE_NAME,
        token_payload={"role": SERVICE_ROLE_NAME, "sub": SERVICE_ROLE_NAME},
        permissions=SERVICE_ROLE_PERMISSIONS,
    )