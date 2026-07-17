"""
Role-permission matrix (role -> allowed doc_type / metadata filters).

Spec 3.6 (Role-Based RAG):
  - The PermissionMatrix singleton loads role definitions from config/roles.yaml.
  - RoleContext and AnonymousContext carry the authenticated user's role and
    permissions through the request lifecycle.
  - A special SERVICE_ROLE (used when ENABLE_RBAC is False) has unrestricted
    access to all document types, ensuring existing tests continue to work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import yaml

from app.config import get_settings
from app.logging.json_logger import get_logger

logger = get_logger("app.rbac.models")

# ── Role enum ────────────────────────────────────────────────────────────────


class Role(str, Enum):
    CLAIMS_PROCESSOR = "claims_processor"
    SENIOR_ADJUSTER = "senior_adjuster"
    CLAIMS_MANAGER = "claims_manager"
    FRAUD_INVESTIGATOR = "fraud_investigator"
    SERVICE = "service"  # full-access role when RBAC is disabled


# ── Permission matrix (loaded from config/roles.yaml) ────────────────────────


@dataclass
class RolePermissions:
    """Permissions granted to a single role.

    When ``allowed_doc_types`` is **empty**, the role is treated as having
    unrestricted access to all document types (used by the service role).
    """

    display_name: str
    description: str
    allowed_doc_types: List[str]
    allowed_insurance_types: Optional[List[str]] = None
    max_retrieval_k: int = 10
    requires_explicit_consent: bool = False
    restricted_doc_types: List[str] = field(default_factory=list)

    @property
    def has_unrestricted_access(self) -> bool:
        """Return True if this role can access all document types.

        A role with an empty allowed_doc_types list is treated as
        unrestricted (used by the service role).
        """
        return len(self.allowed_doc_types) == 0


@dataclass
class JWTConfig:
    """JWT configuration loaded from config/roles.yaml."""

    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    default_role: str = "claims_processor"
    header_prefix: str = "Bearer "


@dataclass
class MetadataFieldMapping:
    """Maps role concepts to Chunk attribute names."""

    doc_type_field: str = "doc_type"
    insurance_type_field: str = "insurance_type"


class PermissionMatrix:
    """Singleton permission matrix loaded from config/roles.yaml.

    Thread-safe after initialisation (read-only at runtime).
    """

    _instance: Optional["PermissionMatrix"] = None

    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path is None:
            config_path = get_settings().ROLES_PATH
        self.config_path = config_path
        self.roles: Dict[str, RolePermissions] = {}
        self.jwt_config: JWTConfig = JWTConfig()
        self.field_mapping: MetadataFieldMapping = MetadataFieldMapping()
        self._load()

    def _load(self) -> None:
        """Parse config/roles.yaml into the permission matrix."""
        path = self.config_path
        if not os.path.exists(path):
            logger.warning("roles_config_not_found", {"path": path})
            self._load_defaults()
            return

        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        if not raw or "roles" not in raw:
            logger.warning("roles_config_missing_roles_key", {"path": path})
            self._load_defaults()
            return

        # Parse roles
        for role_name, role_cfg in raw["roles"].items():
            self.roles[role_name] = RolePermissions(
                display_name=role_cfg.get("display_name", role_name),
                description=role_cfg.get("description", ""),
                allowed_doc_types=role_cfg.get("allowed_doc_types", []),
                allowed_insurance_types=role_cfg.get("allowed_insurance_types"),
                max_retrieval_k=role_cfg.get("max_retrieval_k", 10),
                requires_explicit_consent=role_cfg.get("requires_explicit_consent", False),
                restricted_doc_types=role_cfg.get("restricted_doc_types", []),
            )

        # Parse JWT config
        jwt_cfg = raw.get("jwt", {})
        self.jwt_config = JWTConfig(
            algorithm=jwt_cfg.get("algorithm", "HS256"),
            access_token_expire_minutes=jwt_cfg.get("access_token_expire_minutes", 60),
            default_role=jwt_cfg.get("default_role", "claims_processor"),
            header_prefix=jwt_cfg.get("header_prefix", "Bearer "),
        )

        # Parse metadata field mapping
        mf = raw.get("metadata_fields", {})
        self.field_mapping = MetadataFieldMapping(
            doc_type_field=mf.get("doc_type_field", "doc_type"),
            insurance_type_field=mf.get("insurance_type_field", "insurance_type"),
        )

        logger.info(
            "permission_matrix_loaded",
            {
                "role_count": len(self.roles),
                "roles": list(self.roles.keys()),
                "config_path": path,
            },
        )

    def _load_defaults(self) -> None:
        """Fallback defaults when config file is missing."""
        self.roles = {
            "claims_processor": RolePermissions(
                display_name="Claims Processor",
                description="Process standard claims with policy wordings and SOPs",
                allowed_doc_types=["policy_wording", "sop"],
                max_retrieval_k=10,
            ),
            "senior_adjuster": RolePermissions(
                display_name="Senior Adjuster",
                description="Adjust complex claims with access to prior memos",
                allowed_doc_types=["policy_wording", "sop", "memo"],
                max_retrieval_k=20,
            ),
            "claims_manager": RolePermissions(
                display_name="Claims Manager",
                description="Oversee claims lifecycle with access to prior memos",
                allowed_doc_types=["policy_wording", "sop", "memo"],
                max_retrieval_k=20,
            ),
            "fraud_investigator": RolePermissions(
                display_name="Fraud Investigator",
                description="Investigate suspicious claims with access to investigation files",
                allowed_doc_types=["policy_wording", "sop", "memo", "investigation"],
                max_retrieval_k=30,
                requires_explicit_consent=True,
            ),
        }

    def get_permissions(self, role: str) -> Optional[RolePermissions]:
        """Return the permissions for a given role, or None if unknown."""
        return self.roles.get(role)

    def is_doc_type_allowed(self, role: str, doc_type: str) -> bool:
        """Check whether *role* is permitted to retrieve *doc_type*.

        A role with unrestricted access (empty allowed_doc_types) is
        allowed to retrieve any doc_type.
        """
        perms = self.get_permissions(role)
        if perms is None:
            return False
        if perms.has_unrestricted_access:
            return True
        return doc_type in perms.allowed_doc_types

    def get_allowed_doc_types(self, role: str) -> List[str]:
        """Return the list of doc_types a role may retrieve.

        Returns None (represented as empty list) for unrestricted roles.
        The filter/validator treat empty list as "all types allowed".
        """
        perms = self.get_permissions(role)
        if perms is None:
            return []
        if perms.has_unrestricted_access:
            return []  # empty = all types allowed
        return list(perms.allowed_doc_types)

    def get_max_k(self, role: str) -> int:
        """Return the maximum top-k a role may request."""
        perms = self.get_permissions(role)
        if perms is None:
            return 5
        return perms.max_retrieval_k

    def requires_consent(self, role: str) -> bool:
        """Return whether the role requires explicit consent logging."""
        perms = self.get_permissions(role)
        if perms is None:
            return False
        return perms.requires_explicit_consent

    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> "PermissionMatrix":
        """Return the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config_path=config_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for tests)."""
        cls._instance = None


# ── RoleContext — attached to every authenticated request ────────────────────


@dataclass
class RoleContext:
    """Authenticated user context attached to every request.

    This is stored on ``request.state.role_context`` by the auth middleware
    and consumed by the pre-retrieval filter and post-retrieval validator.
    """

    user_id: str
    role: str
    token_payload: Dict[str, Any] = field(default_factory=dict)
    permissions: Optional[RolePermissions] = None

    def __post_init__(self) -> None:
        if self.permissions is None:
            matrix = PermissionMatrix.get_instance()
            self.permissions = matrix.get_permissions(self.role)

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def has_unrestricted_access(self) -> bool:
        """Return True if this role can access all document types."""
        if self.permissions is None:
            return False
        return self.permissions.has_unrestricted_access

    @property
    def allowed_doc_types(self) -> List[str]:
        """Return allowed doc_types.

        Returns an empty list for unrestricted roles, which the filter
        and validator interpret as "all types allowed".
        """
        if self.permissions is None:
            return []
        if self.permissions.has_unrestricted_access:
            return []  # empty = all types allowed
        return list(self.permissions.allowed_doc_types)

    @property
    def max_k(self) -> int:
        if self.permissions is None:
            return 5
        return self.permissions.max_retrieval_k

    @property
    def requires_consent(self) -> bool:
        if self.permissions is None:
            return False
        return self.permissions.requires_explicit_consent


# ── Unauthenticated context (fallback) ───────────────────────────────────────


@dataclass
class AnonymousContext:
    """Fallback context when no valid JWT is provided.

    The anonymous context has the most restrictive permissions (claims_processor).
    """

    user_id: str = "anonymous"
    role: str = "claims_processor"
    token_payload: Dict[str, Any] = field(default_factory=dict)
    permissions: Optional[RolePermissions] = None

    def __post_init__(self) -> None:
        if self.permissions is None:
            matrix = PermissionMatrix.get_instance()
            self.permissions = matrix.get_permissions(self.role)

    @property
    def is_authenticated(self) -> bool:
        return False

    @property
    def has_unrestricted_access(self) -> bool:
        if self.permissions is None:
            return False
        return self.permissions.has_unrestricted_access

    @property
    def allowed_doc_types(self) -> List[str]:
        if self.permissions is None:
            return []
        if self.permissions.has_unrestricted_access:
            return []
        return list(self.permissions.allowed_doc_types)

    @property
    def max_k(self) -> int:
        if self.permissions is None:
            return 5
        return self.permissions.max_retrieval_k

    @property
    def requires_consent(self) -> bool:
        if self.permissions is None:
            return False
        return self.permissions.requires_explicit_consent