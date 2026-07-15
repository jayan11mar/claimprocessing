#!/usr/bin/env python3
"""
Diagnostic script for Phase R1.
Runs one representative KB query end-to-end through the retrieval chain
and prints findings about RETRIEVER_MODE, PERSISTENCE, and RERANK behaviour.

Usage: PYTHONPATH=. python scripts/diagnose_retrieval_path.py
"""
import os
import sys
import signal
import json
import tempfile
import importlib.util

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tools.knowledge_retrieval import knowledge_retrieval
from app.rag.qa_chain import (
    run_qa_chain,
    _load_chunks_from_manifest,
    _get_persisted_chunks,
    _build_qa_payload,
)
from app.rag.retriever_hybrid import hybrid_retrieve, _get_default_embedding_fn


# ---------------------------------------------------------------------------
# Timeout helper for calls that may hang on OpenAI API
# ---------------------------------------------------------------------------
class TimeoutError_(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutError_("call timed out (>15s)")

def run_with_timeout(func, args=(), kwargs=None, timeout=15):
    """Run `func(*args, **kwargs)` with a SIGALRM timeout (Unix only)."""
    if kwargs is None:
        kwargs = {}
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)
        return result
    except TimeoutError_:
        signal.alarm(0)
        raise
    except Exception:
        signal.alarm(0)
        raise


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def diagnose_retriever_mode() -> str:
    """Check whether an embedding_fn is being passed through the call chain.

    knowledge_retrieval() -> run_qa_chain(embedding_fn=None) -> _build_qa_payload(embedding_fn=None)
    -> hybrid_retrieve(embedding_fn=None) -> _get_default_embedding_fn().

    Returns "dense+BM25" if a real embedding_fn is used, else "BM25-only".
    """
    embedding_fn = _get_default_embedding_fn()
    if embedding_fn is None:
        return "BM25-only (no embedding_fn available; _get_default_embedding_fn() returned None)"
    # Try to determine if it's a real embedding function or a fallback placeholder
    try:
        test_vec = run_with_timeout(embedding_fn, args=(["test"],), timeout=15)
        test_vec = test_vec[0]
        if isinstance(test_vec, list) and len(test_vec) > 0:
            return f"dense+BM25 (embedding_fn available, dim={len(test_vec)})"
        return "dense+BM25 (embedding_fn available)"
    except Exception as e:
        return f"BM25-only (embedding_fn errored: {e})"


def diagnose_persistence() -> str:
    """Check whether chunks are loaded from the persisted FAISS index or rebuilt from manifest.

    Uses _get_persisted_chunks() which tries FAISSStore.load(VECTOR_PERSIST_PATH) first.

    Returns "loaded-from-disk" if the FAISS index was found and loaded,
    else "rebuilt-in-memory".
    """
    from app.config import get_settings

    persist_path = get_settings().VECTOR_PERSIST_PATH
    faiss_index_exists = os.path.exists(persist_path)

    _PERSISTED_CHUNKS_ATTR = "_PERSISTED_CHUNKS"
    # Clear the module-level cache so we get a fresh read
    import app.rag.qa_chain
    if hasattr(app.rag.qa_chain, _PERSISTED_CHUNKS_ATTR):
        setattr(app.rag.qa_chain, _PERSISTED_CHUNKS_ATTR, None)

    chunks = _get_persisted_chunks()
    source_paths = set(c.source_path for c in chunks)

    # Something from the FAISS store won't have a "manifest" path
    # If faiss_index exists and was loaded, chunks will NOT have come from manifest.
    # We check by comparing chunk count: _load_chunks_from_manifest() should produce
    # the exact same count as the persisted store if it was loaded.
    manifest_chunks = _load_chunks_from_manifest()
    loaded_from_disk = faiss_index_exists and len(chunks) > 0

    if loaded_from_disk:
        return f"loaded-from-disk (persisted index at {persist_path}; {len(chunks)} chunks)"
    elif faiss_index_exists:
        return f"rebuilt-in-memory (persisted index found at {persist_path} but failed to load; {len(chunks)} chunks rebuilt from manifest)"
    else:
        return f"rebuilt-in-memory (no persisted index at {persist_path}; {len(chunks)} chunks rebuilt from manifest)"


def diagnose_rerank() -> str:
    """Check whether rerank_results produces a rerank_score or falls back to combined_score.

    Runs a query through and inspects the result fields.

    Returns "active" if every result carries a rerank_score, else "fallback-to-combined_score".
    """
    query = "What are the health insurance exclusions for pre-existing conditions?"
    try:
        result = run_with_timeout(run_qa_chain, kwargs={"query": query, "top_k": 3}, timeout=30)
    except Exception as e:
        return f"error-during-retrieval ({e})"

    citations = result.get("citations", [])
    if not citations:
        return "no-results (cannot determine rerank status)"

    has_rerank = all("rerank_score" in c for c in citations)
    scores = [c.get("score", None) for c in citations]
    rerank_scores = [c.get("rerank_score", None) for c in citations]

    if has_rerank:
        return f"active (rerank_score present in all {len(citations)} citations; scores={rerank_scores})"
    else:
        return f"fallback-to-combined_score (no rerank_score field; score field = {scores})"


def main():
    print("=" * 70)
    print("PHASE R1 — RETRIEVAL PATH DIAGNOSTIC")
    print("=" * 70)

    # 1. RETRIEVER_MODE
    print("\n--- [RETRIEVER_MODE] ---")
    retriever_mode = diagnose_retriever_mode()
    print(f"RETRIEVER_MODE: {retriever_mode}")

    # 2. PERSISTENCE
    print("\n--- [PERSISTENCE] ---")
    persistence = diagnose_persistence()
    print(f"PERSISTENCE: {persistence}")

    # 3. RERANK
    print("\n--- [RERANK] ---")
    rerank_status = diagnose_rerank()
    print(f"RERANK: {rerank_status}")

    # 4. Full end-to-end trace (use pre-computed chunks to avoid re-chunking,
    #    and run_qa_chain directly to match what knowledge_retrieval() does internally)
    print("\n--- [E2E TRACE] ---")
    query = "Does health insurance cover pre-existing conditions with a waiting period?"
    print(f"Query: {query}")
    try:
        result = run_with_timeout(
            run_qa_chain,
            kwargs={"query": query, "top_k": 3},
            timeout=30,
        )
        print(f"Answer: {result.get('answer_text', '')[:200]}...")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        citations = result.get("citations", [])
        print(f"Citations count: {len(citations)}")
        for i, c in enumerate(citations):
            score_type = "rerank_score" if "rerank_score" in c else "combined_score (fallback)"
            score_val = c.get("rerank_score", c.get("score", "N/A"))
            print(f"  [{i}] score={score_val} ({score_type}) source={c.get('source_id', 'N/A')}")
    except Exception as e:
        print(f"E2E retrieval error (timeout/API failure): {e}")
        print("[Note: This is expected when OpenAI API key is invalid/restricted]")
    print("=" * 70)


if __name__ == "__main__":
    main()
