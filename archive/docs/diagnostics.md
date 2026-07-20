# RAG Diagnostics Report

## Query: "What is covered under day care procedures in this policy?"

**Trace Provided:**
```json
{
  "query":"What is covered under day care procedures in this policy",
  "metadata_filter":{"insurance_type":"motor"},
  "result_count":0
}
```

---

## 1. Ingestion Validation

### Source Policy Documents

All health policy PDFs contain references to day-care treatment:

| Source ID | File | Contains "day care" | Contains "day care procedure" | Text Length |
|---|---|---|---|---|
| `health_policy_hdfcergo` | `health_hdfcergo_wording.pdf` | ✅ YES | ✅ YES | 121,770 chars |
| `health_policy_kotak` | `health_kotakmahindra_wording.pdf` | ✅ YES | ❌ NO | 148,776 chars |
| `health_policy_sbi` | `health_sbihealth_wording.pdf` | ✅ YES | ✅ YES | 238,369 chars |
| `motor_policy_sbi_private` | `motor_sbi_private_wording.pdf` | ❌ NO | ❌ NO | 159,618 chars |
| `motor_policy_sbi` | `motor_sbi_wording.pdf` | ❌ NO | ❌ NO | 54,829 chars |

**Verdict:** The source documents are present and contain the relevant content. Day care procedures are mentioned in 3 health policies.

### Generated Chunks (from `data/faiss_index.meta.json`)

```
Total chunks in persisted index: 2 (TWO)
Embedding model version: text-embedding-3-small
```

Only 2 chunks exist in the persisted FAISS index. Both are from `insurance_type=health`, `doc_type=policy_wording`.

**Chunk Contents:**
```
Chunk 0: "Coverage for hospital claims is available after the deductible."
Chunk 1: "Hospital claims are covered after the deductible and prior authorization is required for elective procedures."
```

**Neither chunk contains any reference to "day care", "daycare", or "day care procedure".**

### Embedded Check

The FAISS index (`data/faiss_index`) file is 12,333 bytes and contains 2 vectors of dimension 1536 (2 × 1536 × 4 bytes = ~12,288 bytes matches). This confirms only 2 embeddings were stored.

**Issue 1: The persisted index contains only 2 placeholder chunks instead of the full knowledge base.** When the manifest is loaded from scratch, it produces 52 documents (2 health policies + 2 motor policies + 1 regulation + 2 network + 1 exclusion + 44 CSV adjudication memos containing 50 entries). These should generate hundreds of chunks, but only 2 were persisted.

---

## 2. Vector Store Validation

| Property | Value |
|---|---|
| **Backend** | **FAISS** (`data/faiss_index`) |
| **Index type** | `IndexFlatIP` (inner product) |
| **Dimension** | 1536 |
| **Total indexed vectors** | **2** |
| **Chunks in metadata** | **2** |

The `data/faiss_index.meta.json` confirms exactly 2 chunks. Both correspond to dummy/policy-like placeholder text, not the actual PDF contents.

```
source_id: policy-1
source_id: policy-week6
```

These source IDs do NOT match any source ID in `data/knowledge_base/manifest.yaml` (which uses IDs like `health_policy_hdfcergo`, `health_policy_kotak`, etc.)

**Issue 2: The persisted index was built from a different (test/dummy) dataset, not from the manifest-defined knowledge base files.**

---

## 3. Metadata Validation

### Fields stored per chunk (from Chunk dataclass)

| Field | Type | Present |
|---|---|---|
| `text` | str | ✅ |
| `source_id` | str | ✅ |
| `source_path` | str | ✅ |
| `doc_type` | str | ✅ |
| `insurance_type` | str | ✅ |
| `insurer` | str/None | ✅ |
| `product_code` | str/None | ✅ |
| `product_name` | str/None | ✅ |
| `claim_type` | str/None | ✅ |
| `section` | str/None | ✅ |
| `clause_id` | str/None | ✅ |
| `chunk_index` | int | ✅ |
| `raw_metadata` | dict | ✅ |

### Unique values of `insurance_type` in persisted index

```
{ "health" }
```

**No `motor` chunks exist in the index at all.**

### Unique values of `doc_type` in persisted index

```
{ "policy_wording" }
```

### Case sensitivity

All metadata values in the persisted index are lowercase and match the expected format:
- `insurance_type`: `"health"` (matches manifest values)
- `doc_type`: `"policy_wording"` (matches manifest values)

**Issue 3: The metadata_filter `{"insurance_type":"motor"}` used in the query cannot match any chunk because NO chunks in the index are tagged with `insurance_type=motor`.**

---

## 4. Retrieval Validation

### 4A. No filters (query = "day care procedures")

```python
hybrid_retrieve(chunks, "day care procedures", k=5, metadata_filter=None)
```

The persisted index has only 2 chunks:
1. `"Coverage for hospital claims is available after the deductible."`
2. `"Hospital claims are covered after the deductible and prior authorization is required for elective procedures."`

Neither contains "day", "care", "procedure", or "daycare". 

**Expected result count: 0**

**Reason:** Chunks don't contain matching terms. BM25 would score 0. Dense search would find low cosine similarity with "day care procedures".

### 4B. Current filter: `insurance_type = motor`

```python
hybrid_retrieve(chunks, "day care procedures", k=5, metadata_filter={"insurance_type": "motor"})
```

The metadata filter is applied EARLY in `hybrid_retrieve()` at line 144:
```python
if metadata_filter:
    chunks = _apply_metadata_filter(chunks, metadata_filter)
    if not chunks:
        return []
```

Since no chunks have `insurance_type=motor`, the filter immediately returns an empty list.

**Expected result count: 0** — exactly matches the observed behavior.

### 4C. Health filter: `insurance_type = health`

```python
hybrid_retrieve(chunks, "day care procedures", k=5, metadata_filter={"insurance_type": "health"})
```

The filter would pass both chunks (both have `insurance_type=health`). Then BM25 and dense retrieval would run, but neither chunk mentions "day care".

**Expected result count: 0** (or very low scores with no semantic match)

### 4D. Exact keyword search

A substring search through the 2 chunks for "day care", "daycare", or "day care procedure" would return 0 results.

**Expected: 0 matches**

---

## 5. Collection Validation

### Collection used during ingestion

The ingestion pipeline (`app/rag/ingest_basic.py` or `app/rag/__main__.py`) writes to:
```python
store = get_vector_store(backend=vector_backend, dimension=dimension)
```
Where `vector_backend` defaults to `"faiss"` and the persist path is `data/faiss_index` (from `VECTOR_PERSIST_PATH` env var).

### Collection used during retrieval

The retrieval pipeline (`app/rag/qa_chain.py`) reads from:
```python
persist_path = get_settings().VECTOR_PERSIST_PATH  # "data/faiss_index"
store = FAISSStore.load(persist_path)
```

**Both use the same path: `data/faiss_index`.** The collection name matches. However, the content at that path was built from a different (test) dataset, not the full manifest.

**Issue 4: Collection path is consistent, but the index at that path was populated with test/placeholder data rather than the actual knowledge base documents.**

---

## 6. Query Rewriting Validation

### Is query rewriting enabled?

The function `build_query_variants()` in `app/rag/query_transform.py` is called in `hybrid_retrieve()` only for the token-overlap fallback path (`_dense_token_overlap`), not for the primary BM25+FAISS path.

### Query variants generated (for token-overlap fallback)

For query `"day care procedures"`:
```python
["day care procedures", "day care procedures policy clause", "day care procedures exclusions", "day care procedures evidence"]
```

### Retrieval query path

The query is passed through unchanged:
- `knowledge_retrieval()` → `run_qa_chain()` → `_build_qa_payload()` → `hybrid_retrieve()`
- The original query string is used for both BM25 and dense search.
- **No query rewriting happens for the primary retrieval path.**

---

## 7. Retriever Validation

### Current implementation: `hybrid` (BM25 + Dense FAISS)

From `app/config.py`:
```python
RETRIEVER_MODE: str = Field("hybrid", env="RETRIEVER_MODE")
```

### Code path
1. `_build_qa_payload()` in `qa_chain.py` calls `hybrid_retrieve()`
2. `hybrid_retrieve()` in `retriever_hybrid.py` does:
   - BM25 retrieval (top 20 candidates)
   - Dense retrieval from persistent FAISS store (top 20 candidates)
   - Merge → deduplicate → normalize → combine scores (0.65 BM25 + 0.35 dense)
   - Optional cross-encoder reranking

### The dense path
```python
def _dense_from_vector_store(vector_store, query, candidate_k, embedding_fn):
    query_embedding = embedding_fn([query])[0]  # Embed query only (1 API call)
    results = vector_store.search(query=query, query_embedding=query_embedding, k=candidate_k)
```

### The dense search in FAISSStore.search()
```python
results = self.index.search(query_array, min(k, len(self._chunks)))
# Post-filters by metadata
```

**With only 2 chunks, the inner product search computes similarity against both, finds low scores, and then the metadata filter (`motor`) rejects the only 2 results (both health).**

---

## 8. Root Cause Analysis

### Primary Root Cause

**The persisted FAISS index (`data/faiss_index`) contains only 2 dummy/placeholder chunks instead of the full knowledge base with hundreds of chunks from the actual PDF documents.**

Evidence:
| Check | Result |
|---|---|
| Chunks in persisted index | 2 |
| Source IDs in index | `policy-1`, `policy-week6` (DO NOT match manifest) |
| Documents in manifest | 10+ source files across 6 categories |
| Total text from health policies | ~508,915 chars across 3 PDFs |
| Day care content in source PDFs | ✅ Present in `health_hdfcergo_wording.pdf` and `health_sbihealth_wording.pdf` |
| Day care content in persisted chunks | ❌ NOT PRESENT |

The index at `data/faiss_index` appears to have been built from a test script with artificial placeholder text rather than from the full manifest ingestion pipeline (`python -m app.rag.ingest_basic`).

### Secondary Contributing Cause 1: Incorrect metadata_filter

The retrieval trace shows `metadata_filter={"insurance_type":"motor"}`.

The `_handle_knowledge_retrieval()` method in `app/chains/agent_chain.py` (lines 595-648) infers `insurance_type` from keywords:
```python
if "health" in message_lower or "medical" in message_lower:
    insurance_type = "health"
elif "motor" in message_lower or "car" in message_lower or "vehicle" in message_lower:
    insurance_type = "motor"
```

For the query "What is covered under day care procedures in this policy":
- None of the keywords ("health", "medical", "motor", "car", "vehicle") appear in the query
- Therefore `insurance_type` should be `None` and `metadata_filter` should be `None`

**The presence of `{"insurance_type":"motor"}` in the trace suggests either:**
1. The query was preprocessed/rewritten earlier to include "motor" 
2. The trace is from a different query than stated
3. The FAQ intent classification layer incorrectly identified this as a motor-related query

### Secondary Contributing Cause 2: Early metadata filter rejection

In `hybrid_retrieve()` (line 144), the metadata filter is applied BEFORE BM25 and dense search:
```python
if metadata_filter:
    chunks = _apply_metadata_filter(chunks, metadata_filter)
    if not chunks:
        return []
```

Even if the FAISS index had chunks, a wrong filter would cause early exit with 0 results.

### Secondary Contributing Cause 3: Keyword-based insurance detection is fragile

The keyword detection in `_handle_knowledge_retrieval()` uses simple substring matching:
- `"health"` or `"medical"` → health
- `"motor"`, `"car"`, or `"vehicle"` → motor

Many queries about health insurance don't explicitly say "health" or "medical" (e.g., "day care procedures", "pre-existing diseases", "hospitalization"). These queries would get `metadata_filter=None`, which is correct behavior but highlights the brittleness.

---

## Recommended Fixes

### 1. Re-ingest the knowledge base (HIGH PRIORITY)
```bash
cd /path/to/claimprocessing
python -m app.rag.ingest_basic --vector-backend faiss --embedding-model text-embedding-3-small
```
This will:
- Load all documents from `data/knowledge_base/manifest.yaml`
- Chunk all PDFs, DOCX, CSV files
- Embed all chunks using the configured embedding model
- Persist to `data/faiss_index`

After ingestion, verify:
```bash
python scripts/verify_ingestion.py
```
Expected: hundreds of chunks across health, motor, regulation, network, exclusion_summary, and memo doc types.

### 2. Fix metadata filter inference (MEDIUM PRIORITY)
Improve insurance type detection in `_handle_knowledge_retrieval()`:

**Option A:** Use LLM-based intent extraction rather than keyword matching.

**Option B:** Expand keyword list for health queries:
```python
health_keywords = {"health", "medical", "hospital", "surgery", "treatment", 
                   "day care", "daycare", "pre-existing", "pre existing",
                   "disease", "illness", "medication", "prescription",
                   "doctor", "physician", "patient", "insurance claim"}
```

**Option C:** When no insurance type is detected, run retrieval without a filter (metadata_filter=None). This is currently what should happen for the subject query, but verify the inference path doesn't incorrectly default to "motor".

### 3. Verify ingestion after rebuild
Run the existing verification script:
```bash
python scripts/verify_ingestion.py
```

And the diagnostic script:
```bash
PYTHONPATH=. python scripts/diagnose_retrieval_path.py
```

### 4. Add defensive logging
- Log the `metadata_filter` value at the point of creation in `_handle_knowledge_retrieval()`
- Log the number of chunks that survive the metadata filter in `hybrid_retrieve()`
- This would make future diagnostics immediate without requiring code reading

---

## Summary

| Factor | Finding | Severity |
|---|---|---|
| Source docs contain "day care" content | ✅ YES (in 3 health PDFs) | — |
| Persisted FAISS index has 2 chunks only | ❌ Only 2 placeholder chunks | **CRITICAL** |
| Chunks contain "day care" | ❌ Neither chunk mentions day care | **FATAL** |
| metadata_filter is correct | ❌ Shows `motor` instead of `health`/`None` | **HIGH** |
| Collection name consistency | ✅ Both use `data/faiss_index` | OK |
| Query rewriting | ✅ Original query preserved | OK |
| Retriever configuration | ✅ Hybrid (BM25 + Dense) | OK |

**The `result_count=0` occurs because:**
1. The persisted FAISS index has only 2 placeholder chunks (not the full KB)
2. Neither chunk mentions "day care procedures"
3. The metadata_filter `{"insurance_type":"motor"}` excludes the 2 health-tagged chunks anyway
4. Result: 0 chunks after filtering → 0 results

**The fix is to re-run full ingestion and verify the metadata filter inference path.**