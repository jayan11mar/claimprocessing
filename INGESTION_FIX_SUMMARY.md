# Ingestion Fix Summary

## 1. Files Reviewed

| File | Purpose |
|---|---|
| `app/rag/ingest_basic.py` | Main CLI ingestion entry point |
| `app/rag/__main__.py` | Alternative CLI entry point |
| `app/rag/retriever_basic.py` | Retriever builder with stats |
| `app/rag/vectorstores/faiss_store.py` | FAISS vector store implementation |
| `app/rag/loaders.py` | Document loaders for manifest sources |
| `app/rag/chunkers.py` | Chunking strategies |
| `app/rag/embeddings.py` | Embedding functions |
| `app/config.py` | Configuration settings |
| `data/knowledge_base/manifest.yaml` | Source document manifest |
| `scripts/verify_ingestion.py` | Verification script |

---

## 2. Root Cause Confirmed

**The persisted FAISS index (`data/faiss_index`) contained only 2 dummy/placeholder chunks instead of the full knowledge base.**

### Evidence

| Check | Before Fix | Expected |
|---|---|---|
| Chunks in persisted index | **2** | ~2000+ |
| Source IDs in index | `policy-1`, `policy-week6` | Manifest IDs like `health_policy_hdfcergo` |
| Chunks containing "day care" | **0** | 51+ |
| Insurance types in index | `{health}` | `{health, motor}` |
| FAISS vectors | 2 | 2066 |

### Why It Happened

The `FAISSStore` class has an auto-load mechanism in `__init__()`:
```python
def __init__(self, ...):
    ...
    self._load()  # Auto-loads existing index from disk
```

The ingestion pipeline did this:
```python
store = get_vector_store(backend=vector_backend, dimension=dimension)
store.add(all_chunks, embeddings)  # APPENDS to existing index
store.persist()
```

**Problem:** `store.add()` **appends** to the existing index. If the index already contained 2 dummy chunks from a previous test run, those 2 chunks would remain, and the new chunks would be added after them. Over multiple runs, the index would accumulate stale data.

The dummy chunks (`policy-1`, `policy-week6`) were from an earlier test script that used placeholder text instead of loading from `manifest.yaml`.

---

## 3. Code Changes Made

### `app/rag/ingest_basic.py`

Added `store.delete(ids=None)` before `store.add()` to clear any existing index data:

```python
# ── Step 4: Upsert into vector store ──
print("Step 4/4: Upserting into vector store...")
dimension = len(embeddings[0]) if embeddings else 1536
store = get_vector_store(backend=vector_backend, dimension=dimension)
# Clear any existing index data before adding fresh chunks.
# FAISSStore.__init__ auto-loads persisted data from disk, and add()
# appends to the existing index. Without this clear, old dummy-chunk
# data accumulates across repeated ingestion runs.
store.delete(ids=None)
store.add(all_chunks, embeddings)
store.persist()
```

### `app/rag/retriever_basic.py`

Applied the same fix to `build_basic_retriever()` and `get_retriever_with_stats()`:

```python
# Get vector store
store = get_vector_store(backend=vector_backend, dimension=dimension)

# Clear any existing index data before adding fresh chunks.
store.delete(ids=None)

# Upsert chunks
store.add(all_chunks, embeddings)

# Persist
store.persist()
```

### `scripts/verify_ingestion.py`

Rewrote the verification script to comprehensively check:
- FAISS index vector count
- Metadata JSON consistency
- Source IDs matching manifest.yaml
- Dummy/test data detection
- Day care keyword presence
- Insurance type coverage

---

## 4. Ingestion Command Executed

```bash
cd /home/kasm-user/Documents/claimprocessing
python -m app.rag.ingest_basic --vector-backend faiss --embedding-model text-embedding-3-small
```

**Output:**
```
============================================================
KNOWLEDGE BASE INGESTION
============================================================
Vector backend : faiss
Embedding model: text-embedding-3-small
Chunk size     : 800
Chunk overlap  : 100
Semantic       : True

Step 1/4: Loading documents from manifest...
  → 59 document(s) loaded

Step 2/4: Chunking documents...
  → 2066 chunk(s) created

Step 3/4: Generating embeddings...
  → 2066 embedding(s) generated (dim=1536)

Step 4/4: Upserting into vector store...
  → 2066 chunk(s) stored in faiss backend

============================================================
INGESTION SUMMARY BY DOCUMENT TYPE
============================================================
  exclusion_summary            1 document(s)      3 chunk(s)
  memo                        50 document(s)     50 chunk(s)
  network                      2 document(s)    305 chunk(s)
  policy_wording               5 document(s)   1305 chunk(s)
  regulation                   1 document(s)    403 chunk(s)
------------------------------------------------------------
  TOTAL                       59 document(s)   2066 chunk(s)
============================================================
Ingestion complete!
```

---

## 5. Chunk Count Before Fix

**2 chunks** (both dummy/placeholder text)

```
Chunk 0: "Coverage for hospital claims is available after the deductible."
Chunk 1: "Hospital claims are covered after the deductible and prior authorization is required for elective procedures."
```

Source IDs: `policy-1`, `policy-week6` (not from manifest.yaml)

---

## 6. Chunk Count After Fix

**2066 chunks** from 59 documents

| Document Type | Documents | Chunks |
|---|---|---|
| `policy_wording` | 5 | 1305 |
| `regulation` | 1 | 403 |
| `network` | 2 | 305 |
| `memo` | 50 | 50 |
| `exclusion_summary` | 1 | 3 |
| **TOTAL** | **59** | **2066** |

---

## 7. Unique Source IDs After Fix

**59 unique source_ids** matching manifest.yaml:

| Source ID | Document Type | Chunks |
|---|---|---|
| `health_policy_hdfcergo` | policy_wording | 296 |
| `health_policy_kotak` | policy_wording | 341 |
| `health_policy_sbi` | policy_wording | 349 |
| `motor_policy_sbi_private` | policy_wording | 237 |
| `motor_policy_sbi` | policy_wording | 82 |
| `irda_health_reg_2016` | regulation | 403 |
| `network_agreement_docx` | network | 205 |
| `network_agreement_pdf` | network | 100 |
| `health_exclusions_summary` | exclusion_summary | 3 |
| `prior_adjudication_memos_0` through `_49` | memo | 1 each |

---

## 8. Unique Insurance Type Values

```
{'health', 'motor'}
```

| Insurance Type | Chunks | Percentage |
|---|---|---|
| `health` | 1747 | 84.6% |
| `motor` | 319 | 15.4% |

---

## 9. Day Care Keyword Match Count

| Keyword | Chunks Found | Sample Source |
|---|---|---|
| `day care` | **51** | `health_policy_hdfcergo`, `health_policy_sbi` |
| `daycare` | 0 | — |
| `day care procedure` | **6** | `health_policy_hdfcergo` |
| `day care treatment` | **21** | `health_policy_hdfcergo`, `health_policy_kotak`, `health_policy_sbi` |
| `daycare treatment` | 0 | — |

**Sample matching chunk:**
```
source_id: health_policy_hdfcergo[25]
insurance_type: health
text: "Def. 12 Day Care Procedures means those medical treatment, and/or surgical procedure 
       i. which is undertaken under General or Local Anaesthesia ..."
```

---

## 10. Remaining Risks

### Risk 1: Metadata Filter Inference (Medium Priority)

The `_handle_knowledge_retrieval()` method in `app/chains/agent_chain.py` uses keyword matching to infer `insurance_type`:

```python
if "health" in message_lower or "medical" in message_lower:
    insurance_type = "health"
elif "motor" in message_lower or "car" in message_lower or "vehicle" in message_lower:
    insurance_type = "motor"
```

For the query "What is covered under day care procedures in this policy", none of these keywords appear, so `metadata_filter` should be `None`. However, the original trace showed `metadata_filter={"insurance_type":"motor"}`, which suggests either:
- The trace was from a different query
- The FAQ intent layer misclassified the query
- Some preprocessing added "motor" to the query

**Mitigation:** The ingestion fix alone won't resolve this. The metadata filter inference should be improved to use LLM-based classification or expanded keyword lists.

### Risk 2: Repeated Ingestion Without Clear

If future developers add new ingestion paths without the `store.delete(ids=None)` call, the same bug could recur.

**Mitigation:** Consider adding a safety check in `FAISSStore.add()` that warns if the store already contains data, or make the ingestion API explicitly require a "rebuild" flag.

### Risk 3: Verification Script False Positives

The verification script warns about `prior_adjudication_memos_0` through `_49` not being in `manifest.yaml` as individual IDs. This is expected behavior because the CSV loader creates per-row documents with suffixed IDs.

**Mitigation:** Update the verification script to recognize CSV-derived source IDs as valid.

---

## Validation Criteria Met

| Criterion | Status |
|---|---|
| FAISS index contains significantly more than 2 vectors | ✅ **2066 vectors** |
| source_ids match manifest.yaml | ✅ **All match** |
| insurance_type includes both health and motor | ✅ **{health, motor}** |
| At least one chunk contains "day care" or "daycare" | ✅ **51 chunks** |
| metadata count equals FAISS vector count | ✅ **2066 == 2066** |
| No production index contains only dummy chunks | ✅ **No dummy IDs** |

---

## Commands to Verify

```bash
# 1. Verify ingestion
python scripts/verify_ingestion.py

# 2. Run diagnostic script
PYTHONPATH=. python scripts/diagnose_retrieval_path.py

# 3. Test retrieval for day care query
python -c "
from app.rag.qa_chain import run_qa_chain
result = run_qa_chain('What is covered under day care procedures in this policy?', top_k=3)
print('Answer:', result.get('answer_text', '')[:200])
print('Citations:', len(result.get('citations', [])))
"