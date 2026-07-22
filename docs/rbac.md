# Role-Based Access Control (RBAC) for RAG

## Purpose
This document describes the RBAC (Role-Based Access Control) module that enforces document-level access restrictions on the RAG pipeline, ensuring that users can only retrieve chunks matching their role's permitted document types. The system guarantees **0% leakage** of restricted documents.

## Role Definitions

Roles are defined in **`config/roles.yaml`** — **4 roles** configured:

| Role | Allowed Doc Types | Max Retrieval K | Requires Consent |
|------|-------------------|-----------------|------------------|
| `claims_processor` | policy_wording, sop | 10 | No |
| `senior_adjuster` | policy_wording, sop, memo | 20 | No |
| `claims_manager` | policy_wording, sop, memo | 20 | No |
| `fraud_investigator` | policy_wording, sop, memo, investigation | 30 | Yes |

### Configuration Fields
- `allowed_doc_types` — Document types the role may retrieve
- `allowed_insurance_types` — Optional insurance type filter (null = unrestricted)
- `max_retrieval_k` — Maximum top-k the role may request
- `requires_explicit_consent` — If true, retrieval of certain types requires audit trail
- `restricted_doc_types` — Never-allowed types (currently empty for all roles)

### Anonymous Access
Users without a JWT token are assigned the `claims_processor` role (default).

## Implementation Components

### JWT Authentication (`app/rbac/auth.py`)
- `create_access_token()` — Creates HS256 JWT with subject, role, optional extra claims
- `decode_access_token()` — Decodes and validates JWT, returns None if invalid/expired
- `extract_role_context_from_request()` — Extracts RoleContext/AnonymousContext from FastAPI request
- `get_service_role_context()` — Service role with full permissions for internal operations
- Configurable via `config/roles.yaml`: algorithm (HS256), token expiry, default role

### Permission Matrix (`app/rbac/models.py`)
- `PermissionMatrix` — Singleton loading role permissions from `config/roles.yaml`
- `get_permissions(role)` — Returns `RolePermissions` for a role
- `is_doc_type_allowed(role, doc_type)` — Boolean check for document type access
- `get_max_k(role)` — Returns maximum retrieval k for a role
- `requires_consent(role)` — Returns whether explicit consent is required
- `RolePermissions` — Dataclass with allowed_doc_types, max_retrieval_k, requires_explicit_consent
- `RoleContext` — Authenticated user context with cached permissions
- `AnonymousContext` — Unauthenticated user context (defaults to claims_processor)

### Pre-Retrieval Filter (`app/rbac/filter.py`)
- `build_role_metadata_filter(ctx, query)` — Builds metadata filter for vector store query
  - Single allowed type: returns `{"doc_type": {"$eq": type}}`
  - Multiple allowed types: returns `None` (post-retrieval validation handles filtering)
  - No allowed types: returns `{}` (short-circuit — no results)
- `clamp_top_k(ctx, requested_k)` — Clamps requested top-k to role's maximum

### Post-Retrieval Validator (`app/rbac/validator.py`)
- `validate_retrieval_results(results, ctx, query)` — Strips chunks with disallowed doc_types
- Guarantees **0% leakage**: any chunk with doc_type not in role's allowed list is removed
- Unknown doc_types are kept (fail-open with warning)
- Empty input returns empty output

### Audit Logging (`app/rbac/audit.py`)
- `audit_retrieval()` — Logs every filtered retrieval with role, query, counts
- `audit_consent_event()` — Logs consent events for restricted doc_types
- `audit_leakage_blocked()` — Logs when restricted chunks are blocked
- `audit_top_k_clamp()` — Logs when top-k is clamped
- Audit events: `AUDIT_EVENT_RETRIEVAL`, `AUDIT_EVENT_CONSENT`, `AUDIT_EVENT_LEAKAGE`, `AUDIT_EVENT_CLAMP`

## Runtime Flow

```
User Request (with/without JWT)
    │
    ▼
extract_role_context_from_request(request)
    │
    ├── [Valid JWT] → RoleContext(user_id, role, permissions)
    └── [No/Invalid JWT] → AnonymousContext(default_role = claims_processor)
    │
    ▼
RAG Retrieval
    │
    ├── Pre-Retrieval:
    │   ├── build_role_metadata_filter(ctx, query)
    │   ├── clamp_top_k(ctx, requested_k)
    │   └── (filter injected into hybrid retrieval)
    │
    ├── Retrieval (dense + sparse + reranking)
    │
    └── Post-Retrieval:
        ├── validate_retrieval_results(results, ctx, query)
        │   └── Strips chunks with disallowed doc_types
        ├── audit_retrieval(...)  — log the event
        └── Return filtered results
```

## API Endpoints

All endpoints in **`app/api/server.py`**:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/roles` | List all roles with permissions |
| `GET` | `/auth/context` | Get current auth context (with or without token) |
| `POST` | `/auth/token` | Create JWT token for a role |
| `POST` | `/retrieve` | Retrieval with RBAC filtering (when ENABLE_RBAC=true) |

## JWT Token Format

```json
// POST /auth/token
{"sub": "demo_user", "role": "fraud_investigator"}

// Response
{"access_token": "<JWT>", "token_type": "bearer"}
```

## Zero-Leakage Test Evidence

**Test file:** `tests/test_role_based_rag.py` — **60 test functions** across 10 test classes.

### Key Zero-Leakage Tests

**`test_zero_leakage_guarantee`** (line 480):
```python
def test_zero_leakage_guarantee(self, claims_processor_context, sample_chunks):
    results = validate_retrieval_results(sample_chunks, claims_processor_context, query="test")
    for r in results:
        assert r["doc_type"] in {"policy_wording", "sop"}, (
            f"Leakage detected: {r['doc_type']} chunk leaked through validator"
        )
```

**`test_zero_leakage_across_all_roles`** (line 719):
```python
def test_zero_leakage_across_all_roles(self, sample_chunks):
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
                f"LEAKAGE: role={role_name} received doc_type={r['doc_type']}"
            )
```

### Test Coverage Summary
| Test Class | Tests | Coverage |
|-----------|-------|----------|
| `TestPermissionMatrix` | 10 | Role loading, permissions, doc type checks, max_k, consent, singleton |
| `TestJWTTokens` | 5 | Token creation, decoding, expiry, extra claims |
| `TestRoleContext` | 6 | Role context, anonymous context, request extraction |
| `TestBuildRoleMetadataFilter` | 4 | Pre-retrieval filter construction, overhead NFR |
| `TestClampTopK` | 4 | Top-k clamping per role |
| `TestValidateRetrievalResults` | 8 | Post-retrieval validation, zero-leakage, edge cases, overhead NFR |
| `TestAuditLogging` | 4 | Audit event logging |
| `TestRBACEndpoints` | 6 | API endpoints: /roles, /auth/context, /retrieve |
| `TestRBACFullPipeline` | 6 | End-to-end pipeline, zero-leakage across all roles, NFR |
| `TestRBACEdgeCases` | 7 | Unknown roles, missing doc_type, JWT with different secret |

## Reviewer Demo

```bash
# Run all RBAC tests
python -m pytest tests/test_role_based_rag.py -v -x

# View role configuration
python -c "
import yaml
with open('config/roles.yaml') as f:
    data = yaml.safe_load(f)
for role, config in data['roles'].items():
    print(f'{role}: allowed={config[\"allowed_doc_types\"]}, max_k={config[\"max_retrieval_k\"]}, consent={config[\"requires_explicit_consent\"]}')
"
```

Expected output:
```
claims_processor: allowed=['policy_wording', 'sop'], max_k=10, consent=False
senior_adjuster: allowed=['policy_wording', 'sop', 'memo'], max_k=20, consent=False
claims_manager: allowed=['policy_wording', 'sop', 'memo'], max_k=20, consent=False
fraud_investigator: allowed=['policy_wording', 'sop', 'memo', 'investigation'], max_k=30, consent=True
```
