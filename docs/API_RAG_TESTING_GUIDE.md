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
| **6** | `/sources/{doc_id}` | DELETE | Remove a document and rebuild index | *Path parameter: doc_id from /sources* | `{"status": "success", "message": "Document deleted", "remaining_docs": N}` |
| **7** | `/chat` | POST | Chat with retrieval context (enhanced) | `{"session_id": "user123", "message": "What's my coverage?"}` | `{"intent": "...", "answer": "...", "retrieval_trace": [{"tool": "knowledge_retrieval", "query": "..."}], "citations": [...]}` |
| **8** | `/history/{session_id}` | GET | Retrieve prior turns for a chat session | *Path parameter: session_id* | `{"session_id": "user123", "message_count": 2, "turn_count": 1, "history": [...]}` |
| **9** | `/reset` | POST | Clear chat history for a session | `{"session_id": "user123"}` | `{"status": "ok", "session_id": "user123"}` |
| **10** | `/health` | GET | System health with vector store status | *No body* | `{"status": "healthy", "memory": "ok", "vector_store_status": "indexed", "document_count": 5}` |

---

## Testing Workflow (Recommended Order)

1. **Verify Server Health** → `GET /health`
2. **Ingest Sample Documents** → `POST /ingest`
3. **Check Ingestion Status** → `GET /ingest/status/{job_id}`
4. **List Indexed Documents** → `GET /sources`
5. **Test Pure Retrieval** → `POST /retrieve`
6. **Test Chat with Retrieval** → `POST /chat`
7. **Run Evaluation** → `POST /evaluate`
8. **Inspect Chat History** → `GET /history/{session_id}`
9. **Reset Session History** → `POST /reset`
10. **Clean Up** → `DELETE /sources/{doc_id}`

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
  "status": "success",
  "message": "Document deleted and index rebuilt",
  "deleted_doc_id": "health-policy-001",
  "remaining_docs": 1
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

## Testing Tips

### Check Response Status Codes
- **2xx (Success):** Expected for all endpoints
- **400 (Bad Request):** Invalid JSON or missing required fields
- **404 (Not Found):** Document/job_id not found
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

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Document not found" on /sources/{doc_id} | Use document ID from /sources list, not file name |
| Empty retrieval results | Ensure documents were ingested with POST /ingest |
| Job stuck in "running" state | Check /ingest/status with correct job_id |
| Vector store not indexed | Run /ingest with sample documents first |
| Chat returns no citations | Enable retrieval in agent config |

---

## Unit Tests

All developed endpoints are covered by automated tests:

```bash
# Run all RAG and retrieval tests
pytest tests/test_api_rag_and_retrieval.py -v

# Run specific test
pytest tests/test_api_rag_and_retrieval.py::test_ingest_and_retrieve_round_trip -v

# Run with coverage
pytest tests/test_api_rag_and_retrieval.py --cov=app.api --cov=app.rag
```

**Test Coverage:**
- ✅ Chat endpoint returns retrieval trace and citations
- ✅ Ingest → Retrieve round trip workflow
- ✅ Health endpoint reports vector store status
- ✅ No regressions in existing endpoints
