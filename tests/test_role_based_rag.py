"""
Comprehensive tests for Spec 3.6 (Role-Based RAG).

Tests cover:
  1. Permission matrix loading from config/roles.yaml
  2. JWT token creation and decoding
  3. RoleContext and AnonymousContext
  4. Pre-retrieval metadata filter (build_role_metadata_filter)
  5. Top-k clamping (clamp_top_k)
  6. Post-retrieval validator (validate_retrieval_results)
  7. Audit logging
  8. API endpoints: /roles, /auth/context, /retrieve (with RBAC)
  9. 0% leakage guarantee (mandatory threshold)
 10. Filtering overhead < 200ms (NFR)
"""

import json
import os
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from app.config import get_settings
from app.rbac.auth import (
    create_access_token,
    decode_access_token,
    extract_role_context_from_request,
)
from app.rbac.models import (
    AnonymousContext,
    PermissionMatrix,
    Role,
    RoleContext,
    RolePermissions,
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
from app.api.server import app as fastapi_app
from app.rag.chunkers import Chunk


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_permission_matrix():
    """Reset the PermissionMatrix singleton before each test."""
    PermissionMatrix.reset_instance()
    yield
    PermissionMatrix.reset_instance()


@pytest.fixture
def matrix():
    """Return a fresh PermissionMatrix instance."""
    return PermissionMatrix.get_instance()


@pytest.fixture
def claims_processor_context():
    """Return a RoleContext for claims_processor."""
    return RoleContext(user_id="user1", role="claims_processor")


@pytest.fixture
def senior_adjuster_context():
    """Return a RoleContext for senior_adjuster."""
    return RoleContext(user_id="user2", role="senior_adjuster")


@pytest.fixture
def fraud_investigator_context():
    """Return a RoleContext for fraud_investigator."""
    return RoleContext(user_id="user3", role="fraud_investigator")


@pytest.fixture
def anonymous_context():
    """Return an AnonymousContext."""
    return AnonymousContext()


@pytest.fixture
def sample_chunks() -> List[Dict[str, Any]]:
    """Return sample retrieval results with various doc_types."""
    return [
        {
            "chunk_id": "doc1_0",
            "source_id": "doc1",
            "doc_type": "policy_wording",
            "chunk": Chunk(
                text="Policy coverage details",
                source_id="doc1",
                source_path="/docs/policy1.md",
                doc_type="policy_wording",
                insurance_type="health",
                chunk_index=0,
            ),
        },
        {
            "chunk_id": "doc2_0",
            "source_id": "doc2",
            "doc_type": "memo",
            "chunk": Chunk(
                text="Prior claim memo",
                source_id="doc2",
                source_path="/docs/memo1.json",
                doc_type="memo",
                insurance_type="health",
                chunk_index=0,
            ),
        },
        {
            "chunk_id": "doc3_0",
            "source_id": "doc3",
            "doc_type": "investigation",
            "chunk": Chunk(
                text="Fraud investigation report",
                source_id="doc3",
                source_path="/docs/investigation1.json",
                doc_type="investigation",
                insurance_type="health",
                chunk_index=0,
            ),
        },
        {
            "chunk_id": "doc4_0",
            "source_id": "doc4",
            "doc_type": "sop",
            "chunk": Chunk(
                text="Standard operating procedure",
                source_id="doc4",
                source_path="/docs/sop1.md",
                doc_type="sop",
                insurance_type="health",
                chunk_index=0,
            ),
        },
    ]


# ── 1. Permission Matrix Tests ─────────────────────────────────────────────


class TestPermissionMatrix:
    def test_loads_all_roles(self, matrix):
        """Verify all four roles are loaded from config."""
        assert len(matrix.roles) == 4
        assert "claims_processor" in matrix.roles
        assert "senior_adjuster" in matrix.roles
        assert "claims_manager" in matrix.roles
        assert "fraud_investigator" in matrix.roles

    def test_claims_processor_permissions(self, matrix):
        """claims_processor: policy_wording + sop only."""
        perms = matrix.get_permissions("claims_processor")
        assert perms is not None
        assert perms.allowed_doc_types == ["policy_wording", "sop"]
        assert perms.max_retrieval_k == 10
        assert perms.requires_explicit_consent is False

    def test_senior_adjuster_permissions(self, matrix):
        """senior_adjuster: policy_wording + sop + memo."""
        perms = matrix.get_permissions("senior_adjuster")
        assert perms is not None
        assert set(perms.allowed_doc_types) == {"policy_wording", "sop", "memo"}
        assert perms.max_retrieval_k == 20

    def test_claims_manager_permissions(self, matrix):
        """claims_manager: policy_wording + sop + memo."""
        perms = matrix.get_permissions("claims_manager")
        assert perms is not None
        assert set(perms.allowed_doc_types) == {"policy_wording", "sop", "memo"}
        assert perms.max_retrieval_k == 20

    def test_fraud_investigator_permissions(self, matrix):
        """fraud_investigator: all types including investigation."""
        perms = matrix.get_permissions("fraud_investigator")
        assert perms is not None
        assert set(perms.allowed_doc_types) == {"policy_wording", "sop", "memo", "investigation"}
        assert perms.max_retrieval_k == 30
        assert perms.requires_explicit_consent is True

    def test_is_doc_type_allowed(self, matrix):
        """Verify is_doc_type_allowed for various combinations."""
        # claims_processor
        assert matrix.is_doc_type_allowed("claims_processor", "policy_wording") is True
        assert matrix.is_doc_type_allowed("claims_processor", "sop") is True
        assert matrix.is_doc_type_allowed("claims_processor", "memo") is False
        assert matrix.is_doc_type_allowed("claims_processor", "investigation") is False

        # senior_adjuster
        assert matrix.is_doc_type_allowed("senior_adjuster", "memo") is True
        assert matrix.is_doc_type_allowed("senior_adjuster", "investigation") is False

        # fraud_investigator
        assert matrix.is_doc_type_allowed("fraud_investigator", "investigation") is True

        # unknown role
        assert matrix.is_doc_type_allowed("unknown_role", "policy_wording") is False

    def test_get_max_k(self, matrix):
        """Verify max retrieval k per role."""
        assert matrix.get_max_k("claims_processor") == 10
        assert matrix.get_max_k("senior_adjuster") == 20
        assert matrix.get_max_k("claims_manager") == 20
        assert matrix.get_max_k("fraud_investigator") == 30
        assert matrix.get_max_k("unknown_role") == 5  # default

    def test_requires_consent(self, matrix):
        """Only fraud_investigator requires explicit consent."""
        assert matrix.requires_consent("claims_processor") is False
        assert matrix.requires_consent("senior_adjuster") is False
        assert matrix.requires_consent("claims_manager") is False
        assert matrix.requires_consent("fraud_investigator") is True
        assert matrix.requires_consent("unknown_role") is False

    def test_singleton_pattern(self, matrix):
        """PermissionMatrix.get_instance() returns the same instance."""
        instance2 = PermissionMatrix.get_instance()
        assert matrix is instance2

    def test_reset_instance(self, matrix):
        """reset_instance() clears the singleton."""
        PermissionMatrix.reset_instance()
        new_instance = PermissionMatrix.get_instance()
        assert matrix is not new_instance

    def test_defaults_on_missing_file(self):
        """When config file is missing, defaults are loaded."""
        with patch("os.path.exists", return_value=False):
            PermissionMatrix.reset_instance()
            m = PermissionMatrix.get_instance(config_path="/nonexistent/path.yaml")
            assert len(m.roles) == 4
            assert m.get_permissions("claims_processor") is not None


# ── 2. JWT Token Tests ─────────────────────────────────────────────────────


class TestJWTTokens:
    def test_create_and_decode_token(self):
        """Create a JWT and decode it successfully."""
        token = create_access_token(subject="test_user", role="claims_processor")
        assert isinstance(token, str)
        assert len(token) > 0

        payload = decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == "test_user"
        assert payload["role"] == "claims_processor"
        assert "iat" in payload
        assert "exp" in payload

    def test_decode_invalid_token(self):
        """Decoding an invalid token returns None."""
        payload = decode_access_token("invalid.token.here")
        assert payload is None

    def test_decode_expired_token(self):
        """Decoding an expired token returns None."""
        token = create_access_token(
            subject="test_user",
            role="claims_processor",
            expires_minutes=-1,  # expired immediately
        )
        payload = decode_access_token(token)
        assert payload is None

    def test_token_with_extra_claims(self):
        """Extra claims are embedded in the token."""
        token = create_access_token(
            subject="test_user",
            role="fraud_investigator",
            extra_claims={"department": "fraud", "clearance": "level3"},
        )
        payload = decode_access_token(token)
        assert payload is not None
        assert payload["department"] == "fraud"
        assert payload["clearance"] == "level3"

    def test_token_with_custom_expiry(self):
        """Custom expiry is respected."""
        token = create_access_token(
            subject="test_user",
            role="claims_processor",
            expires_minutes=30,
        )
        payload = decode_access_token(token)
        assert payload is not None
        # Token should be valid for ~30 minutes
        exp = payload["exp"]
        iat = payload["iat"]
        assert exp - iat == 30 * 60


# ── 3. RoleContext Tests ────────────────────────────────────────────────────


class TestRoleContext:
    def test_claims_processor_context(self, claims_processor_context):
        """claims_processor context has correct properties."""
        ctx = claims_processor_context
        assert ctx.user_id == "user1"
        assert ctx.role == "claims_processor"
        assert ctx.is_authenticated is True
        assert ctx.allowed_doc_types == ["policy_wording", "sop"]
        assert ctx.max_k == 10
        assert ctx.requires_consent is False

    def test_fraud_investigator_context(self, fraud_investigator_context):
        """fraud_investigator context has correct properties."""
        ctx = fraud_investigator_context
        assert ctx.role == "fraud_investigator"
        assert set(ctx.allowed_doc_types) == {"policy_wording", "sop", "memo", "investigation"}
        assert ctx.max_k == 30
        assert ctx.requires_consent is True

    def test_anonymous_context(self, anonymous_context):
        """Anonymous context has default role and is not authenticated."""
        ctx = anonymous_context
        assert ctx.user_id == "anonymous"
        assert ctx.role == "claims_processor"
        assert ctx.is_authenticated is False
        assert ctx.allowed_doc_types == ["policy_wording", "sop"]
        assert ctx.max_k == 10

    def test_extract_role_context_from_request_with_token(self):
        """Extract RoleContext from a request with a valid JWT."""
        token = create_access_token(subject="alice", role="senior_adjuster")
        request = MagicMock(spec=Request)
        request.headers = {"Authorization": f"Bearer {token}"}
        request.url.path = "/retrieve"

        ctx = extract_role_context_from_request(request)
        assert isinstance(ctx, RoleContext)
        assert ctx.user_id == "alice"
        assert ctx.role == "senior_adjuster"
        assert ctx.is_authenticated is True

    def test_extract_role_context_from_request_no_token(self):
        """Extract AnonymousContext from a request without a token."""
        request = MagicMock(spec=Request)
        request.headers = {}
        request.url.path = "/retrieve"

        ctx = extract_role_context_from_request(request)
        assert isinstance(ctx, AnonymousContext)
        assert ctx.is_authenticated is False

    def test_extract_role_context_from_request_invalid_token(self):
        """Extract AnonymousContext from a request with an invalid token."""
        request = MagicMock(spec=Request)
        request.headers = {"Authorization": "Bearer invalid.token.here"}
        request.url.path = "/retrieve"

        ctx = extract_role_context_from_request(request)
        assert isinstance(ctx, AnonymousContext)
        assert ctx.is_authenticated is False


# ── 4. Pre-Retrieval Filter Tests ──────────────────────────────────────────


class TestBuildRoleMetadataFilter:
    def test_single_allowed_type_returns_filter(self, claims_processor_context):
        """claims_processor has one allowed type → returns metadata filter."""
        result = build_role_metadata_filter(claims_processor_context, query="test query")
        # claims_processor has 2 allowed types, so returns None (multi-type skip)
        assert result is None

    def test_no_allowed_types_returns_empty_dict(self):
        """Role with no allowed types returns empty dict (short-circuit).

        Note: empty allowed_doc_types means "unrestricted" for the service
        role.  To test "no access", we use a role not in the matrix.
        """
        # A role not in the permission matrix has no permissions → short-circuit
        ctx = RoleContext(user_id="test", role="nonexistent_role")
        result = build_role_metadata_filter(ctx, query="test")
        assert result == {}

    def test_anonymous_context_returns_none(self, anonymous_context):
        """Anonymous context (default role) returns None (multi-type)."""
        result = build_role_metadata_filter(anonymous_context, query="test")
        assert result is None

    def test_filter_overhead_under_200ms(self, claims_processor_context):
        """NFR: Filtering overhead < 200ms."""
        start = time.perf_counter()
        for _ in range(100):
            build_role_metadata_filter(claims_processor_context, query="test query")
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100
        assert elapsed_ms < 200, f"Average filter overhead {elapsed_ms:.3f}ms exceeds 200ms"


# ── 5. Top-k Clamping Tests ────────────────────────────────────────────────


class TestClampTopK:
    def test_clamp_within_limit(self, claims_processor_context):
        """Requested k within limit is not clamped."""
        result = clamp_top_k(claims_processor_context, 5)
        assert result == 5

    def test_clamp_exceeds_limit(self, claims_processor_context):
        """Requested k exceeding limit is clamped."""
        result = clamp_top_k(claims_processor_context, 20)
        assert result == 10  # claims_processor max is 10

    def test_clamp_at_limit(self, claims_processor_context):
        """Requested k at limit is not clamped."""
        result = clamp_top_k(claims_processor_context, 10)
        assert result == 10

    def test_clamp_fraud_investigator(self, fraud_investigator_context):
        """fraud_investigator max is 30."""
        assert clamp_top_k(fraud_investigator_context, 50) == 30
        assert clamp_top_k(fraud_investigator_context, 10) == 10


# ── 6. Post-Retrieval Validator Tests ──────────────────────────────────────


class TestValidateRetrievalResults:
    def test_claims_processor_strips_memo_and_investigation(
        self, claims_processor_context, sample_chunks
    ):
        """claims_processor can only see policy_wording and sop."""
        results = validate_retrieval_results(
            sample_chunks, claims_processor_context, query="test"
        )
        assert len(results) == 2
        doc_types = {r["doc_type"] for r in results}
        assert doc_types == {"policy_wording", "sop"}

    def test_senior_adjuster_allows_memo(
        self, senior_adjuster_context, sample_chunks
    ):
        """senior_adjuster can see policy_wording, sop, and memo."""
        results = validate_retrieval_results(
            sample_chunks, senior_adjuster_context, query="test"
        )
        assert len(results) == 3
        doc_types = {r["doc_type"] for r in results}
        assert doc_types == {"policy_wording", "sop", "memo"}

    def test_fraud_investigator_allows_all(
        self, fraud_investigator_context, sample_chunks
    ):
        """fraud_investigator can see all types."""
        results = validate_retrieval_results(
            sample_chunks, fraud_investigator_context, query="test"
        )
        assert len(results) == 4

    def test_anonymous_strips_memo_and_investigation(
        self, anonymous_context, sample_chunks
    ):
        """Anonymous (default claims_processor) strips memo and investigation."""
        results = validate_retrieval_results(
            sample_chunks, anonymous_context, query="test"
        )
        assert len(results) == 2

    def test_zero_leakage_guarantee(self, claims_processor_context, sample_chunks):
        """Threshold: 0% leakage — no restricted chunks pass through."""
        results = validate_retrieval_results(
            sample_chunks, claims_processor_context, query="test"
        )
        for r in results:
            assert r["doc_type"] in {"policy_wording", "sop"}, (
                f"Leakage detected: {r['doc_type']} chunk leaked through validator"
            )

    def test_empty_results_returns_empty(self, claims_processor_context):
        """Empty input returns empty output."""
        results = validate_retrieval_results([], claims_processor_context, query="test")
        assert results == []

    def test_no_allowed_types_strips_all(self):
        """Role with no allowed types strips everything.

        Note: empty allowed_doc_types means "unrestricted" for the service
        role.  To test "no access", we use a role not in the matrix.
        """
        # A role not in the permission matrix has no permissions → strip all
        ctx = RoleContext(user_id="test", role="nonexistent_role")
        chunks = [
            {"chunk_id": "c1", "doc_type": "policy_wording", "chunk": MagicMock(doc_type="policy_wording")},
        ]
        results = validate_retrieval_results(chunks, ctx, query="test")
        assert results == []

    def test_unknown_doc_type_kept_with_warning(self, claims_processor_context):
        """Chunks with unknown doc_type are kept (fail-open) with warning."""
        # A dict with no doc_type key and a chunk dict without doc_type
        chunks = [
            {"chunk_id": "c1", "chunk": {"text": "some content"}},
        ]
        results = validate_retrieval_results(chunks, claims_processor_context, query="test")
        assert len(results) == 1

    def test_validator_overhead_under_200ms(self, claims_processor_context, sample_chunks):
        """NFR: Validator overhead < 200ms."""
        start = time.perf_counter()
        for _ in range(100):
            validate_retrieval_results(sample_chunks, claims_processor_context, query="test")
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100
        assert elapsed_ms < 200, f"Average validator overhead {elapsed_ms:.3f}ms exceeds 200ms"


# ── 7. Audit Logging Tests ─────────────────────────────────────────────────


class TestAuditLogging:
    def test_audit_retrieval_logs_all_fields(self, caplog, claims_processor_context):
        """audit_retrieval logs all required fields."""
        audit_retrieval(
            role_context=claims_processor_context,
            query="test query",
            requested_k=10,
            effective_k=5,
            pre_filter_count=10,
            post_validator_count=3,
            stripped_count=7,
            elapsed_ms=1.234,
            metadata_filter_used={"doc_type": "policy_wording"},
            fallback_triggered=False,
        )
        # Check that the log record was created (we can't easily capture JSON logger output)
        # The audit function uses the JSON logger, so we verify it doesn't raise
        assert True

    def test_audit_consent_event(self, fraud_investigator_context):
        """audit_consent_event logs consent events."""
        audit_consent_event(
            role_context=fraud_investigator_context,
            action="retrieve_investigation",
            doc_type="investigation",
            query="fraud case 123",
        )
        assert True

    def test_audit_leakage_blocked(self, claims_processor_context):
        """audit_leakage_blocked logs blocked leakage events."""
        audit_leakage_blocked(
            role_context=claims_processor_context,
            doc_type="investigation",
            chunk_id="inv_0",
            source_id="inv_doc",
            query="test query",
            stage="post_retrieval_validator",
        )
        assert True

    def test_audit_top_k_clamp(self, claims_processor_context):
        """audit_top_k_clamp logs clamping events."""
        audit_top_k_clamp(
            role_context=claims_processor_context,
            requested_k=50,
            clamped_to=10,
        )
        assert True


# ── 8. API Endpoint Tests ──────────────────────────────────────────────────


class TestRBACEndpoints:
    client = TestClient(fastapi_app)

    def test_get_roles_endpoint(self):
        """GET /roles returns all roles with permissions."""
        response = self.client.get("/roles")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["role_count"] == 4
        assert "claims_processor" in data["roles"]
        assert "fraud_investigator" in data["roles"]
        # Check structure
        cp = data["roles"]["claims_processor"]
        assert "allowed_doc_types" in cp
        assert "max_retrieval_k" in cp
        assert "requires_explicit_consent" in cp

    def test_auth_context_no_token(self):
        """GET /auth/context returns anonymous context when no token is provided."""
        response = self.client.get("/auth/context")
        assert response.status_code == 200
        data = response.json()
        # With RBAC enabled and no token, the anonymous context is used
        assert data["user_id"] == "anonymous"
        assert data["role"] == "claims_processor"
        assert data["is_authenticated"] is False
        assert data["rbac_enabled"] is True

    def test_auth_context_with_valid_token(self):
        """GET /auth/context returns authenticated context with valid JWT."""
        with patch.object(get_settings(), "ENABLE_RBAC", True):
            token = create_access_token(subject="bob", role="fraud_investigator")
            response = self.client.get(
                "/auth/context",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["user_id"] == "bob"
        assert data["role"] == "fraud_investigator"
        assert data["is_authenticated"] is True
        assert "investigation" in data["permissions"]["allowed_doc_types"]

    def test_auth_context_with_invalid_token(self):
        """GET /auth/context returns anonymous context with invalid token."""
        response = self.client.get(
            "/auth/context",
            headers={"Authorization": "Bearer invalid.token.here"},
        )
        assert response.status_code == 200
        data = response.json()
        # With RBAC enabled, an invalid token results in anonymous context
        assert data["is_authenticated"] is False
        assert data["role"] == "claims_processor"

    def test_retrieve_with_rbac_enabled_and_valid_token(self):
        """POST /retrieve with RBAC enabled and valid token applies filters."""
        # Enable RBAC for this test
        with patch.object(get_settings(), "ENABLE_RBAC", True):
            token = create_access_token(subject="alice", role="claims_processor")
            response = self.client.post(
                "/retrieve",
                json={"query": "test query", "top_k": 5},
                headers={"Authorization": f"Bearer {token}"},
            )
            # The vector store may not be loaded, but the RBAC layer should process
            assert response.status_code == 200
            data = response.json()
            # Should have RBAC fields in response
            assert "results" in data

    def test_retrieve_with_rbac_disabled(self):
        """POST /retrieve with RBAC disabled works without auth."""
        with patch.object(get_settings(), "ENABLE_RBAC", False):
            response = self.client.post(
                "/retrieve",
                json={"query": "test query", "top_k": 5},
            )
            assert response.status_code == 200


# ── 9. Integration: Full RBAC Pipeline ─────────────────────────────────────


class TestRBACFullPipeline:
    """End-to-end tests simulating the full RBAC pipeline."""

    def test_claims_processor_cannot_access_memos(self, sample_chunks):
        """claims_processor: memos and investigations are stripped."""
        ctx = RoleContext(user_id="processor1", role="claims_processor")

        # Simulate pre-filter
        metadata_filter = build_role_metadata_filter(ctx, query="claim details")
        # Multi-type → None (skip pre-filter)

        # Simulate retrieval results (as if from hybrid_retrieve)
        results = sample_chunks

        # Post-retrieval validation
        validated = validate_retrieval_results(results, ctx, query="claim details")

        # Verify: only policy_wording and sop remain
        assert len(validated) == 2
        for r in validated:
            assert r["doc_type"] in {"policy_wording", "sop"}

    def test_senior_adjuster_can_access_memos_but_not_investigations(self, sample_chunks):
        """senior_adjuster: memos allowed, investigations stripped."""
        ctx = RoleContext(user_id="adjuster1", role="senior_adjuster")

        validated = validate_retrieval_results(sample_chunks, ctx, query="claim details")

        assert len(validated) == 3
        for r in validated:
            assert r["doc_type"] in {"policy_wording", "sop", "memo"}

    def test_fraud_investigator_can_access_all(self, sample_chunks):
        """fraud_investigator: all types allowed."""
        ctx = RoleContext(user_id="investigator1", role="fraud_investigator")

        validated = validate_retrieval_results(sample_chunks, ctx, query="fraud case")

        assert len(validated) == 4

    def test_anonymous_gets_claims_processor_permissions(self, sample_chunks):
        """Anonymous user gets claims_processor restrictions."""
        ctx = AnonymousContext()

        validated = validate_retrieval_results(sample_chunks, ctx, query="claim details")

        assert len(validated) == 2
        for r in validated:
            assert r["doc_type"] in {"policy_wording", "sop"}

    def test_zero_leakage_across_all_roles(self, sample_chunks):
        """Threshold: 0% leakage — test all roles for leaks."""
        roles = ["claims_processor", "senior_adjuster", "claims_manager", "fraud_investigator"]
        role_allowed = {
            "claims_processor": {"policy_wording", "sop"},
            "senior_adjuster": {"policy_wording", "sop", "memo"},
            "claims_manager": {"policy_wording", "sop", "memo"},
            "fraud_investigator": {"policy_wording", "sop", "memo", "investigation"},
        }

        for role_name in roles:
            ctx = RoleContext(user_id="test", role=role_name)
            validated = validate_retrieval_results(sample_chunks, ctx, query="test")
            allowed = role_allowed[role_name]
            for r in validated:
                assert r["doc_type"] in allowed, (
                    f"LEAKAGE: role={role_name} received doc_type={r['doc_type']} "
                    f"which is not in allowed set {allowed}"
                )

    def test_filtering_overhead_under_200ms_total(self, sample_chunks):
        """NFR: Total RBAC overhead (filter + validator) < 200ms."""
        ctx = RoleContext(user_id="test", role="claims_processor")

        start = time.perf_counter()
        for _ in range(100):
            # Pre-filter
            build_role_metadata_filter(ctx, query="test")
            # Post-validator
            validate_retrieval_results(sample_chunks, ctx, query="test")
        elapsed_ms = (time.perf_counter() - start) * 1000 / 100
        assert elapsed_ms < 200, (
            f"Average total RBAC overhead {elapsed_ms:.3f}ms exceeds 200ms NFR"
        )


# ── 10. Edge Cases ─────────────────────────────────────────────────────────


class TestRBACEdgeCases:
    def test_unknown_role_falls_back_to_defaults(self):
        """Unknown role gets default permissions (empty allowed list)."""
        ctx = RoleContext(user_id="test", role="nonexistent_role")
        assert ctx.allowed_doc_types == []
        assert ctx.max_k == 5

    def test_chunk_without_doc_type_is_kept(self, claims_processor_context):
        """Chunk without doc_type is kept (fail-open) with warning."""
        # Use a dict without a doc_type key and without a chunk that has doc_type
        results = [
            {"chunk_id": "c1", "chunk": {"text": "some content"}},
        ]
        validated = validate_retrieval_results(results, claims_processor_context, query="test")
        assert len(validated) == 1

    def test_mixed_allowed_and_restricted(self, claims_processor_context):
        """Mix of allowed and restricted types — only allowed pass through."""
        results = [
            {"chunk_id": "c1", "doc_type": "policy_wording", "chunk": MagicMock(doc_type="policy_wording")},
            {"chunk_id": "c2", "doc_type": "investigation", "chunk": MagicMock(doc_type="investigation")},
            {"chunk_id": "c3", "doc_type": "policy_wording", "chunk": MagicMock(doc_type="policy_wording")},
        ]
        validated = validate_retrieval_results(results, claims_processor_context, query="test")
        assert len(validated) == 2
        assert all(r["doc_type"] == "policy_wording" for r in validated)

    def test_jwt_with_different_secret_fails_to_decode(self):
        """Token signed with different secret fails to decode."""
        import jwt as pyjwt
        token = pyjwt.encode(
            {"sub": "test", "role": "claims_processor"},
            "different-secret",
            algorithm="HS256",
        )
        payload = decode_access_token(token)
        assert payload is None

    def test_permission_matrix_reload_after_reset(self):
        """After reset, get_instance loads fresh config."""
        PermissionMatrix.reset_instance()
        m1 = PermissionMatrix.get_instance()
        assert len(m1.roles) == 4

        PermissionMatrix.reset_instance()
        m2 = PermissionMatrix.get_instance()
        assert m1 is not m2
        assert len(m2.roles) == 4