# Remediation Baseline — Phase R1

## Retriever Hybrid Behaviour Summary

File: `app/rag/retriever_hybrid.py`

### When `embedding_fn=None`

The function `hybrid_retrieve()` receives `embedding_fn=None` from the upstream call chain
(`knowledge_retrieval` → `run_qa_chain` → `_build_qa_payload` → `hybrid_retrieve`).  
It then calls `_get_default_embedding_fn()` which:

1. Tries to import `get_embedding_fn` from `app.rag.embeddings` and read `settings.OPENAI_EMBEDDING_MODEL`.
2. Calls `get_embedding_fn(model)` which creates an `OpenAIEmbeddings` wrapper.
3. If the OpenAI API call fails (e.g. missing/invalid key) or no config is available, returns `None`.

If `_get_default_embedding_fn()` returns a valid callable, dense embeddings are computed via OpenAI  
and `_compute_cosine_similarity` is used. **But in this environment, it fails with a 403 (model not found),**  
so the dense score branch (lines 110-115) is skipped and the code falls through to the  
`_token_overlap_score` fallback (lines 128-131).

**Bottom line:** When `embedding_fn=None` and OpenAI is unavailable, retrieval is **BM25-only** with a  
token-overlap dense signal — not real dense embeddings.

### Cross-Encoder Invocation

`rerank_results()` in `app/rag/reranker.py` is called with `rerank=True` (the default).  
It attempts, in order:

1. **Cohere rerank** — requires `COHERE_API_KEY` env var + `cohere` package. Skipped if missing.
2. **CrossEncoder** — requires `RERANK_CROSS_ENCODER_MODEL` env var + `sentence_transformers.CrossEncoder`. Skipped if missing.
3. **Fallback** (`_fallback_rerank`) — computes a lexical token-overlap score and blends it with `combined_score`.  
   This always runs and produces a `rerank_score` field.

**In this environment:** Neither Cohere API key nor `RERANK_CROSS_ENCODER_MODEL` are configured, so the  
fallback runs. However, the upstream `_build_qa_payload` uses `result.get("rerank_score", result["combined_score"])`  
(line 62 of `qa_chain.py`), meaning if `rerank_score` is absent, `combined_score` is used silently.

---

## "Before" Evidence — Diagnostic Script Output

```
======================================================================
PHASE R1 — RETRIEVAL PATH DIAGNOSTIC
======================================================================

--- [RETRIEVER_MODE] ---
RETRIEVER_MODE: BM25-only (embedding_fn errored: Error code: 403 - {'error': {'message': 'Project `proj_TZ6ozLyk481FbqXug4ZKcXJg` does not have access to model `text-embedding-3-small`', 'type': 'invalid_request_error', 'param': None, 'code': 'model_not_found'}})

--- [PERSISTENCE] ---
PERSISTENCE: rebuilt-in-memory (data/faiss_index does not exist; 2066 chunks rebuilt from manifest)

--- [RERANK] ---
RERANK: fallback-to-combined_score (no rerank_score field; score field = [0.8102, 0.782, 0.7484])

--- [E2E TRACE] ---
Query: Does health insurance cover pre-existing conditions with a waiting period?
Answer: The retrieved guidance says: For any health insurance policy, waiting period with respect to pre-existing diseases and time bound exclusions shall be taken into account as follows:-...
Confidence: 0.816
Citations count: 3
  [0] score=0.807 (combined_score (fallback)) source=irda_health_reg_2016
  [1] score=0.7616 (combined_score (fallback)) source=health_policy_kotak
  [2] score=0.7358 (combined_score (fallback)) source=health_policy_sbi
======================================================================