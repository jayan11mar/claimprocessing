API :  RAG & Retrieval Endpoints Testing Guide

## Swagger UI Testing

**Access Swagger UI:** `http://localhost:8000/docs` (after running `python -m app.main`)

## Endpoint Testing Matrix

| # | Endpoint | Method | Purpose | Request Body (JSON) | Expected Response |
|---|----------|--------|---------|-------------------|-----------------|
| **1** | `/ingest` | POST | Upload and index documents for RAG | `{"documents": [{"id": "doc1", "path": "policy.md", "doc_type": "policy", "insurance_type": "health", "content": "Full document text..."}]}` | `{"status": "accepted", "job_id": "uuid", "message": "Ingestion accepted", "job": {...}}` |
| **2** | `/ingest/status/{job_id}` | GET | Poll document ingestion job progress | *Path parameter: job_id from /ingest response* | `{"job_id": "uuid", "status": "completed\|running\|failed", "progress": 0-100, "document_count": N}` |
| **3** | `/retrieve` | POST | Hybrid search without LLM (pure retrieval) | `{"query": "hospital coverage", "top_k": 5}` | `{"results": [{"chunk": {"text": "...", "source_id": "doc1"}, "score": 0.95}, ...]}` |
| **4** | `/evaluate` | POST | Run RAG evaluation against golden dataset | `{}` (no body required) | `{"summary": {"total_cases": 10, "passed_cases": 8, "thresholds": {...}}, "cases": [...]}` |
| **5** | `/sources` | GET | List all indexed documents | *No body* | `{"documents": [{"source_id": "doc1", "source_path": "policy.md", "doc_type": "policy", "text_preview": "..."}]}` |
| **6** | `/sources/{doc_id}` | DELETE | Remove a document and rebuild index | *Path parameter: doc_id from /sources* | `{"status": "deleted", "doc_id": "health-policy-001"}` |
| **6b** | `/sources/{doc_id}/reload` | POST | Re-ingest a single source document from manifest | *Path parameter: doc_id from /sources* | `{"status": "reloaded", "doc_id": "health-policy-001", "document_count": N}` |
| **7** | `/chat` | POST | Chat with retrieval context (enhanced) | `{"session_id": "user123", "message": "What's my coverage?"}` | `{"intent": "...", "answer": "...", "retrieval_trace": [{"tool": "knowledge_retrieval", "query": "..."}], "citations": [...]}` |
| **8** | `/history/{session_id}` | GET | Retrieve prior turns for a chat session | *Path parameter: session_id* | `{"session_id": "user123", "message_count": 2, "turn_count": 1, "history": [...]}` |
| **9** | `/reset` | POST | Clear chat history for a session | `{"session_id": "user123"}` | `{"status": "ok", "session_id": "user123"}` |
| **10** | `/health` | GET | System health with vector store status | *No body* | `{"status": "healthy", "memory": "ok", "vector_store_status": "indexed", "document_count": 5}` |
| **11** | `/sources/{doc_id}/download` | GET | Download the original source document file | *Path parameter: doc_id from /sources* | `FileResponse` with attachment headers or `{"status": "not_available", "message": "..."}` |
| **12** | `/roles` | GET | List all roles and their permissions from RBAC matrix | *No body* | `{"roles": {...}, "enabled": true}` |
| **13** | `/auth/token` | POST | Issue a JWT access token for a given subject and role | `{"sub": "user1", "role": "claims_processor"}` | `{"access_token": "eyJ...", "token_type": "bearer", "sub": "user1", "role": "claims_processor"}` |
| **14** | `/auth/context` | GET | Return the current authentication context for the request | *No body* | `{"user_id": "...", "role": "...", "is_authenticated": true, "permissions": {...}}` |
| **15** | `/hitl/pending` | GET | List all pending HITL tasks | *No body* | `{"tasks": [{"task_id": "...", "status": "pending", ...}]}` |
| **16** | `/hitl/review/{task_id}` | POST | Review (approve or reject) a pending HITL task | `{"decision": "approved", "comments": "Looks good"}` | `{"task": {...}, "message": "Task '...' has been approved."}` |
| **17** | `/hitl/task/{task_id}` | GET | Get a single HITL task by ID (regardless of status) | *Path parameter: task_id* | `{"status": "ok", "task": {...}}` |
| **18** | `/mcp/tools` | GET | List all discovered MCP tools | *No body* | `{"tools": [{"name": "...", "description": "...", "input_schema": {...}}]}` |
| **19** | `/mcp/invoke` | POST | Invoke an MCP tool by name | `{"tool": "tool_name", "arguments": {...}}` | `{"tool": "...", "result": {...}, "success": true, "latency_ms": 123}` |
| **20** | `/eval/regression` | POST | Run a full regression evaluation against the golden set | `{}` or with optional fields | `{"summary": {...}, "comparison": {...}, "report_path": "..."}` |
| **21** | `/eval/drift` | POST | Run drift detection (read-only monitoring) | `{}` or with optional fields | `{"status": "ok", "message": "Drift detection completed.", "scores": {...}, "breaches": [...], "any_breach": false}` |
| **22** | `/prompts` | GET | List all registered prompts with their active versions | *No body* | `{"prompts": {"faq_chain": {"name": "faq_chain", "active_version": "1.0", "available_versions": ["1.0", "1.1"]}}}` |
| **23** | `/prompts/{name}/history` | GET | Get version history for a specific prompt | *Path parameter: name* | `{"name": "...", "active_version": "1.0", "versions": [...]}` |
| **24** | `/prompts/{name}/activate` | POST | Activate a specific version of a prompt (rollback) | `{"version": "1.0"}` | `{"name": "...", "active_version": "1.0", "message": "Activated version '1.0' for prompt '...'"}` |

---

## Testing Workflow (Recommended Order)

1. **Verify Server Health** → `GET /health`
2. **Obtain Auth Token** → `POST /auth/token`
3. **Validate Auth Context** → `GET /auth/context`
4. **List Roles** → `GET /roles`
5. **Ingest Sample Documents** → `POST /ingest`
6. **Check Ingestion Status** → `GET /ingest/status/{job_id}`
7. **List Indexed Documents** → `GET /sources`
8. **Test Pure Retrieval** → `POST /retrieve`
9. **Test Chat with Retrieval** → `POST /chat`
10. **List MCP Tools** → `GET /mcp/tools`
11. **Invoke MCP Tool** → `POST /mcp/invoke`
12. **List Pending HITL Tasks** → `GET /hitl/pending`
13. **Get HITL Task Details** → `GET /hitl/task/{task_id}`
14. **Review HITL Task** → `POST /hitl/review/{task_id}`
15. **List Prompts** → `GET /prompts`
16. **View Prompt History** → `GET /prompts/{name}/history`
17. **Activate Prompt Version** → `POST /prompts/{name}/activate`
18. **Run Evaluation** → `POST /evaluate`
19. **Run Regression** → `POST /eval/regression`
20. **Check Drift** → `POST /eval/drift`
21. **Inspect Chat History** → `GET /history/{session_id}`
22. **Reset Session History** → `POST /reset`
23. **Delete a Source Document** → `DELETE /sources/{doc_id}`
24. **Reload a Source Document** → `POST /sources/{doc_id}/reload`
25. **Download a Source Document** → `GET /sources/{doc_id}/download`

---

## Sample Request Bodies (Copy-Paste Ready)

### 1. POST /ingest - Ingest Documents

**Purpose:** Upload and chunk documents for RAG indexing

```json
{
  "documents": [
    {
      "id": "health-policy-001",
      "path": "health_policy.md",
      "doc_type": "policy_wording",
      "insurance_type": "health",
      "content": "This policy covers hospital stays up to 30 days per year. Emergency room visits are covered at 80% after $500 deductible. Preventive care is covered at 100%. Deductible resets annually on January 1st."
    },
    {
      "id": "claims-guide-001",
      "path": "claims_guide.md",
      "doc_type": "guide",
      "insurance_type": "health",
      "content": "To file a claim, submit form CL-100 within 30 days of service. Include itemized receipts and provider details. Claims are processed within 5-7 business days. Appeal window is 90 days from denial."
    }
  ]
}
```

**Expected Response:**
```json
{
  "status": "accepted",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Ingestion accepted",
  "job": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "progress": 100,
    "message": "Ingestion complete",
    "document_count": 2
  }
}
```

---

### 2. GET /ingest/status/{job_id} - Check Ingestion Progress

**Purpose:** Poll the status of an ingestion job

**Path Parameter:** `job_id` = `550e8400-e29b-41d4-a716-446655440000` (from /ingest response)

**Expected Response (In Progress):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": 50,
  "message": "Building embeddings and index"
}
```

**Expected Response (Completed):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "message": "Ingestion complete",
  "document_count": 2
}
```

---

### 3. POST /retrieve - Pure Retrieval (No LLM)

**Purpose:** Search documents using hybrid semantic + lexical matching

```json
{
  "query": "What is covered under emergency room visits?",
  "top_k": 3
}
```

**Expected Response:**
```json
{
  "query": "What is covered under emergency room visits?",
  "results": [
    {
      "chunk": {
        "text": "Emergency room visits are covered at 80% after $500 deductible.",
        "source_id": "health-policy-001",
        "source_path": "health_policy.md",
        "doc_type": "policy_wording"
      },
      "combined_score": 0.92,
      "rerank_score": 0.88
    },
    {
      "chunk": {
        "text": "Preventive care is covered at 100%.",
        "source_id": "health-policy-001",
        "source_path": "health_policy.md",
        "doc_type": "policy_wording"
      },
      "combined_score": 0.65,
      "rerank_score": 0.58
    }
  ]
}
```

---

### 4. POST /evaluate - Run RAG Evaluation

**Purpose:** Validate RAG quality against golden dataset (eval/golden_set.json)

**Request Body:**
```json
{}
```

**Expected Response:**
```json
{
  "summary": {
    "total_cases": 10,
    "passed_cases": 8,
    "failed_cases": 2,
    "pass_rate": 0.8,
    "thresholds": {
      "hit_rate": 0.85,
      "mrr": 0.65,
      "faithfulness": 0.9,
      "answer_correctness": 0.8
    }
  },
  "cases": [
    {
      "case_id": "case-001",
      "query": "What is the deductible?",
      "expected_docs": ["health-policy-001"],
      "retrieved_docs": ["health-policy-001"],
      "hit": true,
      "mrr": 1.0,
      "generated_answer": "The deductible is $500...",
      "faithfulness": 0.95,
      "answer_correctness": 0.9,
      "status": "passed"
    }
  ]
}
```

---

### 5. GET /sources - List All Indexed Documents

**Purpose:** View all documents currently indexed in the vector store

**Request Body:** *(none)*

**Expected Response:**
```json
{
  "document_count": 2,
  "documents": [
    {
      "source_id": "health-policy-001",
      "source_path": "health_policy.md",
      "doc_type": "policy_wording",
      "insurance_type": "health",
      "text_preview": "This policy covers hospital stays up to 30 days per year..."
    },
    {
      "source_id": "claims-guide-001",
      "source_path": "claims_guide.md",
      "doc_type": "guide",
      "insurance_type": "health",
      "text_preview": "To file a claim, submit form CL-100 within 30 days..."
    }
  ]
}
```

---

### 6. DELETE /sources/{doc_id} - Remove a Document

**Purpose:** Delete a document and rebuild the RAG index

**Path Parameter:** `doc_id` = `health-policy-001` (from /sources response)

**Expected Response:**
```json
{
  "status": "deleted",
  "doc_id": "health-policy-001"
}
```

---

### 6b. POST /sources/{doc_id}/reload - Re-ingest a Source Document

**Purpose:** Re-load a single source document from the manifest, deleting old chunks and re-ingesting fresh content

**Path Parameter:** `doc_id` = `health_policy_hdfcergo` (a manifest source ID from /sources)

**Expected Response:**
```json
{
  "status": "reloaded",
  "doc_id": "health_policy_hdfcergo",
  "message": "Document 'health_policy_hdfcergo' has been re-ingested.",
  "document_count": 9
}
```

---

### 7. POST /chat - Chat with Retrieval Context

**Purpose:** Conversational query with RAG context, citations, and retrieval trace

```json
{
  "session_id": "test-user-123",
  "message": "Does my policy cover emergency room visits?"
}
```

**Expected Response:**
```json
{
  "session_id": "test-user-123",
  "message": "Does my policy cover emergency room visits?",
  "intent": "FAQIntent.POLICY_COVERAGE",
  "category": "policy",
  "confidence": 0.92,
  "answer": "Yes, your policy covers emergency room visits at 80% after a $500 deductible.",
  "reasoning": "Retrieved from policy document",
  "retrieval_trace": [
    {
      "tool": "knowledge_retrieval",
      "query": "emergency room coverage",
      "status": "success",
      "results_count": 2
    }
  ],
  "citations": [
    {
      "source_id": "health-policy-001",
      "source_path": "health_policy.md",
      "text": "Emergency room visits are covered at 80% after $500 deductible."
    }
  ]
}
```

---

### 8. GET /history/{session_id} - View Session History

**Purpose:** Retrieve the prior chat turns stored for a session

**Path Parameter:** `session_id` = `test-user-123`

**Expected Response:**
```json
{
  "session_id": "test-user-123",
  "message_count": 2,
  "turn_count": 1,
  "history": [
    {
      "role": "user",
      "content": "Does my policy cover emergency room visits?"
    },
    {
      "role": "assistant",
      "content": "Yes, your policy covers emergency room visits at 80% after a $500 deductible."
    }
  ]
}
```

---

### 9. POST /reset - Clear Session History

**Purpose:** Remove stored conversation history for a specific session

```json
{
  "session_id": "test-user-123"
}
```

**Expected Response:**
```json
{
  "status": "ok",
  "session_id": "test-user-123"
}
```

---

### 10. GET /health - System Health Check

**Purpose:** Verify server is running and check vector store status

**Request Body:** *(none)*

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-07-05T14:30:00Z",
  "memory_status": "ok",
  "vector_store_status": "indexed",
  "document_count": 2,
  "vector_store_type": "faiss",
  "vector_dimension": 1536
}
```

---

### 11. GET /sources/{doc_id}/download - Download Source Document

**Purpose:** Download the original source document file as an attachment

**Path Parameter:** `doc_id` = `health-policy-001` (from /sources response)

**Headers:** *(none required)*

**Expected Success Response:**
- HTTP 200 with `Content-Disposition: attachment; filename="health_policy.md"`
- Binary file content streamed as `FileResponse`

**Expected Response (Text-Only Document):**
```json
{
  "status": "not_available",
  "message": "This document was uploaded as text content and has no file to download."
}
```

**Validation Checklist:**
- [ ] HTTP 200 with file attachment when source_path points to a real file
- [ ] `Content-Disposition` header includes the correct filename
- [ ] HTTP 200 with JSON `not_available` for text-only uploads
- [ ] HTTP 404 when doc_id does not exist

---

### 12. GET /roles - List Roles and Permissions

**Purpose:** List all roles and their permissions from the RBAC permission matrix

**Request Body:** *(none)*

**Expected Response:**
```json
{
  "roles": {
    "claims_processor": {
      "display_name": "Claims Processor",
      "description": "Process standard claims with policy wordings and SOPs",
      "allowed_doc_types": ["policy_wording", "sop"],
      "allowed_insurance_types": null,
      "max_retrieval_k": 10,
      "requires_explicit_consent": false
    },
    "senior_adjuster": {
      "display_name": "Senior Adjuster",
      "description": "Adjust complex claims with access to prior memos",
      "allowed_doc_types": ["policy_wording", "sop", "memo"],
      "allowed_insurance_types": null,
      "max_retrieval_k": 20,
      "requires_explicit_consent": false
    },
    "claims_manager": {
      "display_name": "Claims Manager",
      "description": "Oversee claims lifecycle with access to prior memos",
      "allowed_doc_types": ["policy_wording", "sop", "memo"],
      "allowed_insurance_types": null,
      "max_retrieval_k": 20,
      "requires_explicit_consent": false
    },
    "fraud_investigator": {
      "display_name": "Fraud Investigator",
      "description": "Investigate suspicious claims with access to investigation files",
      "allowed_doc_types": ["policy_wording", "sop", "memo", "investigation"],
      "allowed_insurance_types": null,
      "max_retrieval_k": 30,
      "requires_explicit_consent": true
    }
  },
  "enabled": true
}
```

**Validation Checklist:**
- [ ] HTTP 200 with roles object
- [ ] `enabled` field reflects `ENABLE_RBAC` setting
- [ ] Each role includes `display_name`, `description`, `allowed_doc_types`, `max_retrieval_k`
- [ ] Roles match config/roles.yaml definitions

---

### 13. POST /auth/token - Issue JWT Access Token

**Purpose:** Issue a JWT access token for a given subject and role (for RBAC testing)

**Request Body:**
```json
{
  "sub": "test-user-001",
  "role": "claims_processor"
}
```

**Expected Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "sub": "test-user-001",
  "role": "claims_processor"
}
```

**Validation Checklist:**
- [ ] HTTP 200 with `access_token` (JWT string)
- [ ] `token_type` is `bearer`
- [ ] `sub` and `role` match the request
- [ ] Token is a valid HS256 JWT (decode at jwt.io)
- [ ] HTTP 422 for missing required fields

---

### 14. GET /auth/context - Get Authentication Context

**Purpose:** Return the current authentication context for the request, including active role and permissions

**Headers:** `Authorization: Bearer <token>` (optional)

**Request Body:** *(none)*

**Expected Response (Authenticated):**
```json
{
  "user_id": "test-user-001",
  "role": "claims_processor",
  "is_authenticated": true,
  "permissions": {
    "allowed_doc_types": ["policy_wording", "sop"],
    "max_retrieval_k": 10,
    "requires_explicit_consent": false
  },
  "rbac_enabled": true
}
```

**Expected Response (Anonymous / No Token):**
```json
{
  "user_id": "anonymous",
  "role": "claims_processor",
  "is_authenticated": false,
  "permissions": {
    "allowed_doc_types": ["policy_wording", "sop"],
    "max_retrieval_k": 10,
    "requires_explicit_consent": false
  },
  "rbac_enabled": true
}
```

**Validation Checklist:**
- [ ] HTTP 200 with context object
- [ ] `is_authenticated` is `true` when valid Bearer token provided
- [ ] `is_authenticated` is `false` when no token provided
- [ ] `role` and `permissions` reflect the token's role
- [ ] `rbac_enabled` matches `ENABLE_RBAC` setting

---

### 15. GET /hitl/pending - List Pending HITL Tasks

**Purpose:** List all pending HITL (Human-In-The-Loop) tasks awaiting review

**Request Body:** *(none)*

**Expected Response (HITL Enabled):**
```json
{
  "tasks": [
    {
      "task_id": "hitl-task-001",
      "session_id": "test-user-123",
      "status": "pending",
      "created_at": "2026-07-22T12:00:00Z",
      "context": {
        "message": "Does my policy cover emergency room visits?",
        "response": "Yes, your policy covers emergency room visits at 80% after a $500 deductible."
      },
      "decision": null,
      "comments": null,
      "reviewed_at": null,
      "reviewed_by": null
    }
  ]
}
```

**Expected Response (HITL Disabled):**
```json
{
  "tasks": []
}
```

**Validation Checklist:**
- [ ] HTTP 200 with `tasks` array
- [ ] Each task has `task_id`, `status`, `created_at`, `context`
- [ ] Only `pending` tasks are returned
- [ ] Empty array when HITL is disabled

---

### 16. POST /hitl/review/{task_id} - Review HITL Task

**Purpose:** Review (approve or reject) a pending HITL task

**Path Parameter:** `task_id` = `hitl-task-001` (from /hitl/pending response)

**Request Body (Approval):**
```json
{
  "decision": "approved",
  "comments": "Response looks accurate based on policy document."
}
```

**Request Body (Rejection):**
```json
{
  "decision": "rejected",
  "comments": "Incorrect coverage percentage. Should be 70% not 80%."
}
```

**Expected Response (Approval):**
```json
{
  "task": {
    "task_id": "hitl-task-001",
    "session_id": "test-user-123",
    "status": "approved",
    "created_at": "2026-07-22T12:00:00Z",
    "context": {
      "message": "Does my policy cover emergency room visits?",
      "response": "Yes, your policy covers emergency room visits at 80% after a $500 deductible."
    },
    "decision": "approved",
    "comments": "Response looks accurate based on policy document.",
    "reviewed_at": "2026-07-22T12:05:00Z",
    "reviewed_by": "reviewer"
  },
  "message": "Task 'hitl-task-001' has been approved."
}
```

**Expected Response (Rejection):**
```json
{
  "task": {
    "task_id": "hitl-task-001",
    "session_id": "test-user-123",
    "status": "rejected",
    "created_at": "2026-07-22T12:00:00Z",
    "context": {
      "message": "Does my policy cover emergency room visits?",
      "response": "Yes, your policy covers emergency room visits at 80% after a $500 deductible."
    },
    "decision": "rejected",
    "comments": "Incorrect coverage percentage. Should be 70% not 80%.",
    "reviewed_at": "2026-07-22T12:05:00Z",
    "reviewed_by": "reviewer"
  },
  "message": "Task 'hitl-task-001' has been rejected."
}
```

**Validation Checklist:**
- [ ] HTTP 200 with updated task and message
- [ ] `status` changes from `pending` to `approved` or `rejected`
- [ ] `decision` and `comments` are persisted
- [ ] `reviewed_at` timestamp is populated
- [ ] HTTP 404 when task_id not found
- [ ] HTTP 400 when HITL is disabled
- [ ] Reviewed task no longer appears in `/hitl/pending`

---

### 17. GET /hitl/task/{task_id} - Get HITL Task Details

**Purpose:** Get a single HITL task by ID (regardless of status)

**Path Parameter:** `task_id` = `hitl-task-001`

**Request Body:** *(none)*

**Expected Response:**
```json
{
  "status": "ok",
  "task": {
    "task_id": "hitl-task-001",
    "session_id": "test-user-123",
    "status": "pending",
    "created_at": "2026-07-22T12:00:00Z",
    "context": {
      "message": "Does my policy cover emergency room visits?",
      "response": "Yes, your policy covers emergency room visits at 80% after a $500 deductible."
    },
    "decision": null,
    "comments": null,
    "reviewed_at": null,
    "reviewed_by": null
  }
}
```

**Validation Checklist:**
- [ ] HTTP 200 with `status: "ok"` and `task` object
- [ ] Returns task regardless of status (pending, approved, rejected)
- [ ] HTTP 404 when task_id not found
- [ ] HTTP 400 when HITL is disabled

---

### 18. GET /mcp/tools - List MCP Tools

**Purpose:** List all discovered MCP (Model Context Protocol) tools

**Request Body:** *(none)*

**Expected Response (MCP Enabled):**
```json
{
  "tools": [
    {
      "name": "get_claim_status",
      "description": "Retrieve the status of a claim by claim ID",
      "input_schema": {
        "type": "object",
        "properties": {
          "claim_id": {
            "type": "string",
            "description": "The claim ID to look up"
          }
        },
        "required": ["claim_id"]
      }
    }
  ]
}
```

**Expected Response (MCP Disabled):**
```json
{
  "detail": "MCP is disabled"
}
```

**Validation Checklist:**
- [ ] HTTP 200 with `tools` array
- [ ] Each tool has `name`, `description`, `input_schema`
- [ ] `input_schema` follows JSON Schema format
- [ ] HTTP 400 when MCP is disabled

---

### 19. POST /mcp/invoke - Invoke MCP Tool

**Purpose:** Invoke an MCP tool by name with provided arguments

**Request Body:**
```json
{
  "tool": "get_claim_status",
  "arguments": {
    "claim_id": "CL-100-2026-001"
  }
}
```

**Expected Response (Success):**
```json
{
  "tool": "get_claim_status",
  "result": {
    "claim_id": "CL-100-2026-001",
    "status": "in_review",
    "amount": 5000.00,
    "filed_date": "2026-07-15"
  },
  "success": true,
  "latency_ms": 234
}
```

**Expected Response (Tool Not Found):**
```json
{
  "tool": "nonexistent_tool",
  "success": false,
  "error": "Tool 'nonexistent_tool' not found. Use /mcp/tools to list available tools.",
  "latency_ms": 5
}
```

**Validation Checklist:**
- [ ] HTTP 200 with `success: true` for valid tool invocations
- [ ] `result` contains the tool's output
- [ ] `latency_ms` is a positive integer
- [ ] HTTP 200 with `success: false` for unknown tools
- [ ] Error message includes guidance to use `/mcp/tools`
- [ ] HTTP 400 when MCP is disabled

---

### 20. POST /eval/regression - Run Regression Evaluation

**Purpose:** Run a full regression evaluation against the golden set, with optional baseline comparison

**Request Body (Minimal):**
```json
{}
```

**Request Body (With Baseline Comparison):**
```json
{
  "golden_set_path": "eval/golden_set.json",
  "project_filter": null,
  "thresholds": {
    "hit_rate": 0.85,
    "mrr": 0.65,
    "faithfulness": 0.9,
    "answer_correctness": 0.8
  },
  "baseline_path": "reports/regression_report.json"
}
```

**Expected Response:**
```json
{
  "summary": {
    "total_cases": 10,
    "passed_cases": 8,
    "failed_cases": 2,
    "pass_rate": 0.8,
    "thresholds": {
      "hit_rate": 0.85,
      "mrr": 0.65,
      "faithfulness": 0.9,
      "answer_correctness": 0.8
    }
  },
  "comparison": {
    "previous_pass_rate": 0.75,
    "current_pass_rate": 0.8,
    "regressions": 1,
    "improvements": 2,
    "unchanged": 7
  },
  "report_path": "reports/regression_report.json"
}
```

**Validation Checklist:**
- [ ] HTTP 200 with `summary` containing pass/fail metrics
- [ ] `comparison` present when `baseline_path` is provided
- [ ] `comparison` includes `regressions`, `improvements`, `unchanged` counts
- [ ] `report_path` points to a valid file in `reports/`
- [ ] Results are written to `reports/regression_report.json`

---

### 21. POST /eval/drift - Run Drift Detection

**Purpose:** Run drift detection (read-only monitoring) on evaluation cases

**Request Body (Minimal):**
```json
{}
```

**Request Body (With Custom Paths):**
```json
{
  "baseline_path": "reports/drift_baseline.json",
  "thresholds_path": "config/drift_thresholds.yaml"
}
```

**Expected Response (Drift Detected):**
```json
{
  "status": "ok",
  "message": "Drift detection completed.",
  "scores": {
    "embedding_drift": 0.12,
    "prompt_drift": 0.08,
    "response_drift": 0.15
  },
  "breaches": [
    {
      "metric": "response_drift",
      "score": 0.15,
      "threshold": 0.1,
      "severity": "warning"
    }
  ],
  "any_breach": true
}
```

**Expected Response (No Drift):**
```json
{
  "status": "ok",
  "message": "Drift detection completed.",
  "scores": {
    "embedding_drift": 0.03,
    "prompt_drift": 0.02,
    "response_drift": 0.04
  },
  "breaches": [],
  "any_breach": false
}
```

**Expected Response (Drift Disabled):**
```json
{
  "status": "ok",
  "message": "drift disabled"
}
```

**Validation Checklist:**
- [ ] HTTP 200 with `status: "ok"`
- [ ] `scores` contains drift metrics (embedding, prompt, response)
- [ ] `breaches` array lists any metrics exceeding thresholds
- [ ] `any_breach` is `true` when at least one threshold is exceeded
- [ ] `any_breach` is `false` when all scores are within thresholds
- [ ] Graceful message when drift is disabled (`ENABLE_DRIFT=false`)

---

### 22. GET /prompts - List Registered Prompts

**Purpose:** List all registered prompts with their active versions

**Request Body:** *(none)*

**Expected Response:**
```json
{
  "prompts": {
    "faq_chain": {
      "name": "faq_chain",
      "active_version": "1.0",
      "available_versions": ["1.0", "1.1"]
    },
    "rag_chain": {
      "name": "rag_chain",
      "active_version": "2.0",
      "available_versions": ["1.0", "2.0"]
    }
  }
}
```

**Validation Checklist:**
- [ ] HTTP 200 with `prompts` object
- [ ] Each prompt has `name`, `active_version`, `available_versions`
- [ ] `available_versions` is sorted ascending
- [ ] `active_version` matches one of the available versions

---

### 23. GET /prompts/{name}/history - Get Prompt Version History

**Purpose:** Get version history for a specific prompt

**Path Parameter:** `name` = `faq_chain` (from /prompts response)

**Request Body:** *(none)*

**Expected Response:**
```json
{
  "name": "faq_chain",
  "active_version": "1.0",
  "versions": [
    {
      "name": "faq_chain",
      "version": "1.0",
      "author": "admin",
      "last_updated": "2026-07-01T10:00:00Z",
      "changelog": "Initial version",
      "model_compatibility": ["gpt-4o-mini"],
      "input_variables": ["question", "context"],
      "template": "Answer the following question based on the context...",
      "templates": null,
      "activated_at": "2026-07-01T10:00:00Z"
    },
    {
      "name": "faq_chain",
      "version": "1.1",
      "author": "admin",
      "last_updated": "2026-07-15T14:00:00Z",
      "changelog": "Improved prompt formatting for better accuracy",
      "model_compatibility": ["gpt-4o-mini", "gpt-4o"],
      "input_variables": ["question", "context"],
      "template": "Using the provided context, answer the question accurately...",
      "templates": null,
      "activated_at": null
    }
  ]
}
```

**Validation Checklist:**
- [ ] HTTP 200 with `name`, `active_version`, `versions` array
- [ ] Each version includes `version`, `author`, `last_updated`, `changelog`
- [ ] `activated_at` is populated only for the active version
- [ ] Versions are ordered ascending
- [ ] HTTP 404 when prompt name does not exist

---

### 24. POST /prompts/{name}/activate - Activate Prompt Version

**Purpose:** Activate a specific version of a prompt (rollback)

**Path Parameter:** `name` = `faq_chain` (from /prompts response)

**Request Body:**
```json
{
  "version": "1.0"
}
```

**Expected Response:**
```json
{
  "name": "faq_chain",
  "active_version": "1.0",
  "message": "Activated version '1.0' for prompt 'faq_chain'"
}
```

**Validation Checklist:**
- [ ] HTTP 200 with `name`, `active_version`, `message`
- [ ] `active_version` matches the requested version
- [ ] Subsequent `GET /prompts/{name}/history` shows `activated_at` for the new active version
- [ ] HTTP 404 when prompt name does not exist
- [ ] HTTP 404 when version does not exist for that prompt
- [ ] Rollback is O(1) — completes in < 10ms

---

## Testing Tips

### Check Response Status Codes
- **2xx (Success):** Expected for all endpoints
- **400 (Bad Request):** Invalid JSON or missing required fields
- **401 (Unauthorized):** Missing or invalid Bearer token (auth endpoints)
- **404 (Not Found):** Document/job_id/task_id/prompt not found
- **422 (Unprocessable Entity):** Validation error in request body
- **500 (Internal Error):** Server error (check logs)

### Monitor Vector Store
- After `/ingest`, documents are chunked and embedded
- Vector store is persisted at `data/faiss_index`
- `/retrieve` performs hybrid search (BM25 + semantic similarity)

### Session Management
- `/chat` uses `session_id` for context continuity
- Each session maintains conversation history in SQLite
- `/reset` endpoint clears all sessions

### Performance
- `/retrieve` (pure retrieval): ~100-200ms
- `/chat` (with LLM): ~2-3 seconds (gpt-4o-mini)
- `/evaluate` (golden dataset): ~30-60 seconds (depends on dataset size)
- `/eval/regression`: ~30-120 seconds (depends on dataset size)
- `/eval/drift`: ~5-30 seconds
- `/mcp/invoke`: depends on external MCP server latency

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Document not found" on /sources/{doc_id} | Use document ID from /sources list, not file name |
| Empty retrieval results | Ensure documents were ingested with POST /ingest |
| Job stuck in "running" state | Check /ingest/status with correct job_id |
| Vector store not indexed | Run /ingest with sample documents first |
| Chat returns no citations | Enable retrieval in agent config |
| HITL endpoint returns 400 | Set `ENABLE_HITL=true` in environment |
| MCP endpoint returns 400 | Set `ENABLE_MCP=true` in environment |
| Auth token invalid | Ensure `JWT_SECRET_KEY` is consistent between token creation and validation |
| Drift detection returns "drift disabled" | Set `ENABLE_DRIFT=true` in environment |

---

## Unit Tests

All developed endpoints are covered by automated tests:

```bash
# Run all RAG and retrieval tests
pytest tests/test_api_rag_and_retrieval.py -v

# Run specific test
pytest tests/test_api_rag_and_retrieval.py::test_ingest_and_retrieve_round_trip -v

# Run HITL workflow tests
pytest tests/test_hitl_workflow.py -v

# Run MCP integration tests
pytest tests/test_mcp_integration.py -v

# Run prompt versioning tests
pytest tests/test_prompt_versioning.py -v

# Run drift detection tests
pytest tests/test_drift_detection.py -v

# Run with coverage
pytest tests/test_api_rag_and_retrieval.py --cov=app.api --cov=app.rag
```

**Test Coverage:**
- ✅ Chat endpoint returns retrieval trace and citations
- ✅ Ingest → Retrieve round trip workflow
- ✅ Health endpoint reports vector store status
- ✅ No regressions in existing endpoints
- ✅ HITL workflow (pending → review → completion)
- ✅ MCP tool discovery and invocation
- ✅ Prompt versioning (list, history, activate)
- ✅ Drift detection metrics and breach reporting

---

## Authentication Testing

### Obtaining a Token

Use `POST /auth/token` to obtain a JWT access token for testing RBAC-protected endpoints.

**Request:**
```json
{
  "sub": "test-user-001",
  "role": "claims_processor"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "sub": "test-user-001",
  "role": "claims_processor"
}
```

### Using the Bearer Token

Include the token in the `Authorization` header for subsequent requests:

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Validating Auth Context

Use `GET /auth/context` to verify the current authentication state:

**With valid token:**
```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/auth/context
```
→ `is_authenticated: true`, `role` matches token role

**Without token:**
```bash
curl http://localhost:8000/auth/context
```
→ `is_authenticated: false`, `role: "claims_processor"` (anonymous fallback)

### Expected 401 Behavior

When RBAC is enabled (`ENABLE_RBAC=true`) and a request is made to a protected endpoint without a valid token:

- The anonymous context is applied with the most restrictive role (`claims_processor`)
- Retrieval results are filtered to only include allowed document types
- `top_k` is clamped to the role's `max_retrieval_k` limit

**Test Steps:**
1. Obtain a token for `fraud_investigator` role
2. Call `/auth/context` with the token → verify `is_authenticated: true`, `role: "fraud_investigator"`
3. Call `/auth/context` without token → verify `is_authenticated: false`, `role: "claims_processor"`
4. Call `/auth/token` with invalid role → verify HTTP 422 validation error
5. Call `/auth/context` with expired/invalid token → verify `is_authenticated: false`

---

## HITL Workflow Testing

### End-to-End Flow

The HITL (Human-In-The-Loop) workflow follows three steps:

1. **List pending tasks** → `GET /hitl/pending`
2. **Get task details** → `GET /hitl/task/{task_id}`
3. **Review the task** → `POST /hitl/review/{task_id}`

### Step 1: List Pending Tasks

```bash
curl http://localhost:8000/hitl/pending
```

**Expected:** Array of pending HITL tasks with `status: "pending"`.

### Step 2: Get Task Details

```bash
curl http://localhost:8000/hitl/task/hitl-task-001
```

**Expected:** Full task object including `context` (the message and response awaiting review).

### Step 3a: Approve a Task

```json
{
  "decision": "approved",
  "comments": "Response is accurate based on policy document section 3.2"
}
```

**Expected Response:**
```json
{
  "task": {
    "task_id": "hitl-task-001",
    "status": "approved",
    "decision": "approved",
    "comments": "Response is accurate based on policy document section 3.2",
    "reviewed_at": "2026-07-22T12:05:00Z",
    "reviewed_by": "reviewer"
  },
  "message": "Task 'hitl-task-001' has been approved."
}
```

### Step 3b: Reject a Task

```json
{
  "decision": "rejected",
  "comments": "Incorrect deductible amount. Policy states $1,000 not $500."
}
```

**Expected Response:**
```json
{
  "task": {
    "task_id": "hitl-task-001",
    "status": "rejected",
    "decision": "rejected",
    "comments": "Incorrect deductible amount. Policy states $1,000 not $500.",
    "reviewed_at": "2026-07-22T12:05:00Z",
    "reviewed_by": "reviewer"
  },
  "message": "Task 'hitl-task-001' has been rejected."
}
```

### Verification

- After review, the task no longer appears in `GET /hitl/pending`
- `GET /hitl/task/{task_id}` still returns the task with updated status
- The `reviewed_at` timestamp is populated
- The `reviewed_by` field identifies the reviewer

---

## Prompt Versioning Testing

### End-to-End Flow

1. **List all prompts** → `GET /prompts`
2. **View version history** → `GET /prompts/{name}/history`
3. **Activate a different version** → `POST /prompts/{name}/activate`

### Step 1: List All Prompts

```bash
curl http://localhost:8000/prompts
```

**Expected:** Object with prompt names as keys, each containing `active_version` and `available_versions`.

### Step 2: View Version History

```bash
curl http://localhost:8000/prompts/faq_chain/history
```

**Expected:** Array of versions with metadata (author, changelog, template, activated_at).

### Step 3: Activate a Different Version

```json
{
  "version": "1.0"
}
```

**Expected Response:**
```json
{
  "name": "faq_chain",
  "active_version": "1.0",
  "message": "Activated version '1.0' for prompt 'faq_chain'"
}
```

### Validation: Confirm Active Version Change

1. Call `GET /prompts/faq_chain/history` before activation → note `active_version`
2. Call `POST /prompts/faq_chain/activate` with a different version
3. Call `GET /prompts/faq_chain/history` after activation → verify `active_version` changed
4. Verify the newly activated version now has `activated_at` populated
5. Verify the previously active version's `activated_at` is now `null`

---

## MCP Tool Testing

### End-to-End Flow

1. **List available tools** → `GET /mcp/tools`
2. **Invoke a tool** → `POST /mcp/invoke`

### Step 1: List MCP Tools

```bash
curl http://localhost:8000/mcp/tools
```

**Expected:** Array of tool objects with `name`, `description`, and `input_schema`.

### Step 2: Invoke a Tool

```json
{
  "tool": "get_claim_status",
  "arguments": {
    "claim_id": "CL-100-2026-001"
  }
}
```

**Expected Response:**
```json
{
  "tool": "get_claim_status",
  "result": {
    "claim_id": "CL-100-2026-001",
    "status": "in_review",
    "amount": 5000.00
  },
  "success": true,
  "latency_ms": 234
}
```

### Verification

- `success: true` indicates the tool executed successfully
- `result` contains the tool's output (structure depends on the tool)
- `latency_ms` provides execution time for performance monitoring
- Invoking a non-existent tool returns `success: false` with an error message
- The error message includes guidance to use `/mcp/tools` to list available tools

---

## Regression & Drift Monitoring

### Regression Testing

`POST /eval/regression` runs a full regression evaluation against the golden set.

**Minimal invocation:**
```json
{}
```

**With baseline comparison:**
```json
{
  "golden_set_path": "eval/golden_set.json",
  "baseline_path": "reports/regression_report.json",
  "thresholds": {
    "hit_rate": 0.85,
    "mrr": 0.65,
    "faithfulness": 0.9,
    "answer_correctness": 0.8
  }
}
```

**Expected Pass/Fail Metrics:**
- `pass_rate >= threshold` → **PASS**
- `pass_rate < threshold` → **FAIL**
- `regressions > 0` → Investigate regressed cases
- `improvements > 0` → Positive change from baseline

### Drift Detection

`POST /eval/drift` runs read-only drift monitoring on evaluation cases.

**Minimal invocation:**
```json
{}
```

**Drift Indicators:**
| Metric | Normal Range | Warning Threshold | Action Required |
|--------|-------------|-------------------|-----------------|
| `embedding_drift` | 0.0 – 0.05 | > 0.10 | Review embedding model or data distribution |
| `prompt_drift` | 0.0 – 0.05 | > 0.10 | Review prompt template changes |
| `response_drift` | 0.0 – 0.05 | > 0.10 | Review LLM response quality changes |

**Interpretation:**
- `any_breach: false` → System is stable, no action needed
- `any_breach: true` → Investigate breached metrics
- Each breach includes `metric`, `score`, `threshold`, and `severity`
- Breaches with `severity: "critical"` require immediate attention

---

## Reviewer Evidence Capture Template

| Endpoint | HTTP Status | Screenshot Captured | Request Verified | Response Verified | Pass/Fail | Notes |
|----------|------------|---------------------|-----------------|-------------------|-----------|-------|
| `GET /health` | 200 | ✅ | ✅ | ✅ | ✅ PASS | Server healthy, vector store indexed, 2 documents |

---

## Coverage Summary

Total OpenAPI Paths: 24
Documented Endpoints: 24
Swagger Coverage: 100%