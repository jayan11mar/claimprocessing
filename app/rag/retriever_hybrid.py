"""Hybrid retrieval that fuses BM25 and persistent FAISS vector search.

Design:
  - BM25 retrieves a limited candidate set (top 20).
  - Persistent FAISS (or other VectorStore) retrieves top 20 via pre-computed
    embeddings — no per-query chunk re-embedding.
  - Candidates are merged and deduplicated by chunk_id (source_id + chunk_index).
  - Scores from both signals are normalized and combined.
  - A small candidate list is passed to the cross-encoder reranker.
  - Returns final top-k results.

Standalone mode (no vector_store):
  - BM25 only + token-overlap fallback. No API calls for embeddings.
"""

import logging
import math
import re
from typing import Any, Callable, Dict, List, Optional

from app.rag.chunkers import Chunk
from app.rag.query_transform import build_query_variants
from app.rag.retriever_bm25 import BM25Result, bm25_retrieve
from app.rag.reranker import rerank_results
from app.rag.vectorstores.base import VectorStore

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────

_CHUNK_ID_SEP = "_"


def _make_chunk_id(source_id: str, chunk_index: int) -> str:
    return f"{source_id}{_CHUNK_ID_SEP}{chunk_index}"


def _chunk_sort_key(item: Dict[str, Any]) -> float:
    return item.get("combined_score", 0.0)


def _deduplicate_merged(
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Deduplicate a list of scored items by chunk_id, keeping the entry with
    the higher combined_score."""
    seen: Dict[str, Dict[str, Any]] = {}
    for item in items:
        cid = item["chunk_id"]
        if cid in seen:
            if item["combined_score"] > seen[cid]["combined_score"]:
                seen[cid] = item
        else:
            seen[cid] = item
    return list(seen.values())


def _token_overlap_score(chunk_text: str, query: str) -> float:
    tokens = {token for token in re.findall(r"[a-z0-9]+", query.lower()) if token}
    if not tokens:
        return 0.0
    chunk_tokens = {token for token in re.findall(r"[a-z0-9]+", chunk_text.lower()) if token}
    if not chunk_tokens:
        return 0.0
    overlap = len(tokens & chunk_tokens)
    return overlap / max(1, len(tokens))


def _get_default_embedding_fn() -> Optional[Callable[[List[str]], List[List[float]]]]:
    """Get a default embedding function from config, or None if not available."""
    try:
        from app.rag.embeddings import get_embedding_fn
        from app.config import get_settings

        settings = get_settings()
        model = getattr(settings, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return get_embedding_fn(model)
    except Exception:
        return None


def _compute_cosine_similarity(
    query_embedding: List[float],
    chunk_embedding: List[float],
) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if not query_embedding or not chunk_embedding:
        return 0.0
    try:
        dot_product = sum(a * b for a, b in zip(query_embedding, chunk_embedding))
        query_mag = math.sqrt(sum(a * a for a in query_embedding))
        chunk_mag = math.sqrt(sum(b * b for b in chunk_embedding))
        return dot_product / (query_mag * chunk_mag) if (query_mag * chunk_mag) > 0 else 0.0
    except Exception:
        return 0.0


# ── Main entry point ─────────────────────────────────────────────────────────


def hybrid_retrieve(
    chunks: List[Chunk],
    query: str,
    k: int = 5,
    embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    rerank: bool = True,
    metadata_filter: Optional[Dict[str, Any]] = None,
    vector_store: Optional[VectorStore] = None,
) -> List[Dict[str, Any]]:
    """Fuse BM25 and dense (FAISS) signals into a ranked list.

    When a *vector_store* is provided, dense scores are obtained from the
    persistent index (pre-computed embeddings) — no per-query chunk
    re-embedding occurs.  Only the query itself is embedded (one API call).

    When *vector_store* is **None**, the dense path falls back to a fast
    token-overlap heuristic — no embedding API calls at all.

    BM25 always runs on the full *chunks* list (fast, no embeddings needed)
    and produces a limited candidate set.

    Args:
        chunks: List of Chunk objects to search.
        query: The query string.
        k: Number of results to return.
        embedding_fn: Optional embedding function for query embedding.
                      Required when *vector_store* is provided.  If None,
                      loaded from config automatically.
        rerank: Whether to apply cross-encoder reranking.
        metadata_filter: Optional dict of metadata fields to filter on.
        vector_store: Optional persistent VectorStore (e.g. FAISSStore) for
                      pre-computed dense search.  When provided, dense scoring
                      uses the store's ``search()`` method instead of
                      re-embedding chunk texts.

    Returns:
        Ranked list of result dicts. Each result dict may contain a
        ``fallback_used`` key in the metadata when the filter fallback
        was triggered.
    """
    if not chunks:
        return []

    # ── Apply metadata filter early ──────────────────────────────────────
    fallback_used = False
    original_filter = metadata_filter
    chunks_before_filter = len(chunks)
    chunks_after_filter = chunks_before_filter
    
    if metadata_filter:
        # Store original chunks before filtering for potential fallback
        original_chunks = chunks
        chunks = _apply_metadata_filter(chunks, metadata_filter)
        chunks_after_filter = len(chunks)
        
        # Check if fallback is needed
        if not chunks:
            from app.config import get_settings
            settings = get_settings()
            
            if settings.RETRIEVAL_FILTER_FALLBACK_ENABLED:
                # Log the fallback event
                logger.warning(
                    "Metadata filter returned zero chunks, applying fallback",
                    extra={
                        "query": query,
                        "original_metadata_filter": original_filter,
                        "chunks_before_filter": chunks_before_filter,
                        "chunks_after_filter": chunks_after_filter,
                        "fallback_used": True,
                        "reason": "metadata filter produced zero candidate chunks",
                    }
                )
                
                # Retry without filter - use original unfiltered chunks
                chunks = original_chunks
                fallback_used = True
            else:
                # Fallback disabled, return empty
                logger.warning(
                    "Metadata filter returned zero chunks, fallback disabled",
                    extra={
                        "query": query,
                        "original_metadata_filter": original_filter,
                        "chunks_before_filter": chunks_before_filter,
                        "chunks_after_filter": chunks_after_filter,
                        "fallback_used": False,
                    }
                )
                return []

    candidate_k = max(k * 4, 20)

    # ── BM25: fast lexical retrieval ─────────────────────────────────────
    bm25_results = bm25_retrieve(chunks, query, k=min(candidate_k, len(chunks)))
    bm25_by_cid: Dict[str, BM25Result] = {}
    for r in bm25_results:
        cid = _make_chunk_id(r.chunk.source_id, r.chunk.chunk_index)
        bm25_by_cid[cid] = r

    # ── Dense: query persistent vector store OR use token-overlap ────────
    if vector_store is not None:
        dense_results = _dense_from_vector_store(
            vector_store, query, candidate_k, embedding_fn
        )
    else:
        dense_results = _dense_token_overlap(chunks, query, candidate_k)

    # ── Merge → deduplicate → normalise → combine scores ─────────────────
    merged = _merge_scores(bm25_by_cid, dense_results)
    merged = _deduplicate_merged(merged)
    if not merged:
        return []

    # ── Sort by combined score ───────────────────────────────────────────
    merged.sort(key=_chunk_sort_key, reverse=True)

    # ── Rerank ────────────────────────────────────────────────────────────
    if rerank:
        rerank_candidates = merged[: max(k * 2, 20)]
        merged = rerank_results(query, rerank_candidates, top_k=k)

    # ── Add fallback metadata to results ─────────────────────────────────
    if fallback_used:
        for result in merged:
            result["fallback_used"] = True
            result["original_metadata_filter"] = original_filter
            result["filter_fallback_reason"] = "metadata filter produced zero candidate chunks"
        
        # Log final results
        logger.warning(
            "Filter fallback completed",
            extra={
                "query": query,
                "original_metadata_filter": original_filter,
                "chunks_before_filter": chunks_before_filter,
                "chunks_after_filter": 0,
                "fallback_used": True,
                "final_result_count": len(merged),
            }
        )
    else:
        # Log normal retrieval
        logger.info(
            "Retrieval completed",
            extra={
                "query": query,
                "metadata_filter": metadata_filter,
                "chunks_before_filter": chunks_before_filter if metadata_filter else len(chunks),
                "chunks_after_filter": chunks_after_filter if metadata_filter else len(chunks),
                "fallback_used": False,
                "final_result_count": len(merged),
            }
        )

    return merged[:k]


# ── Internal stages ──────────────────────────────────────────────────────────


def _apply_metadata_filter(
    chunks: List[Chunk], metadata_filter: Dict[str, Any]
) -> List[Chunk]:
    filtered = []
    for chunk in chunks:
        match = True
        for key, value in metadata_filter.items():
            chunk_val = getattr(chunk, key, None)
            if chunk_val is None:
                chunk_val = chunk.raw_metadata.get(key)
            if chunk_val != value:
                match = False
                break
        if match:
            filtered.append(chunk)
    return filtered




def _dense_from_vector_store(
    vector_store: VectorStore,
    query: str,
    candidate_k: int,
    embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
) -> List[Dict[str, Any]]:
    """Query the persistent vector store for dense candidates.

    Only the *query* is embedded — pre-computed chunk embeddings in the store
    are used for similarity search.  This eliminates per-query chunk
    re-embedding.
    """
    if embedding_fn is None:
        embedding_fn = _get_default_embedding_fn()
    if embedding_fn is None:
        logger.warning(
            "No embedding_fn available for vector_store query; "
            "falling back to empty dense results."
        )
        return []

    # ── Embed the query only (one API call) ──────────────────────────────
    try:
        query_embedding = embedding_fn([query])[0]
    except Exception as exc:
        logger.warning("Query embedding failed: %s", exc)
        return []

    # ── Search the persistent store ──────────────────────────────────────
    try:
        results = vector_store.search(
            query=query,
            query_embedding=query_embedding,
            k=candidate_k,
        )
    except Exception as exc:
        logger.warning("Vector store search failed: %s", exc)
        return []

    dense_list: List[Dict[str, Any]] = []
    for chunk, score in results:
        cid = _make_chunk_id(chunk.source_id, chunk.chunk_index)
        dense_list.append({
            "chunk_id": cid,
            "chunk": chunk,
            "source_id": chunk.source_id,
            "source_path": chunk.source_path,
            "dense_score": round(score, 4),
        })
    return dense_list


def _dense_token_overlap(
    chunks: List[Chunk], query: str, candidate_k: int
) -> List[Dict[str, Any]]:
    """Fast fallback dense signal using token overlap — no embeddings."""
    variants = build_query_variants(query)
    overlap_scores: List[Dict[str, Any]] = []
    for chunk in chunks:
        scores_per_variant = [
            _token_overlap_score(chunk.text, v) for v in variants
        ]
        best = max(scores_per_variant) if scores_per_variant else 0.0
        if best > 0.0:
            cid = _make_chunk_id(chunk.source_id, chunk.chunk_index)
            overlap_scores.append({
                "chunk_id": cid,
                "chunk": chunk,
                "source_id": chunk.source_id,
                "source_path": chunk.source_path,
                "dense_score": round(best, 4),
            })
    overlap_scores.sort(key=lambda x: x["dense_score"], reverse=True)
    return overlap_scores[:candidate_k]


def _merge_scores(
    bm25_by_cid: Dict[str, BM25Result],
    dense_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge BM25 and dense results into a combined score list.

    Scoring:
      - Each source's scores are normalised to [0, 1] independently.
      - Combined = 0.65 * normalized_BM25 + 0.35 * normalized_dense.
    """
    if not bm25_by_cid and not dense_list:
        return []

    # Build a lookup from chunk_id → dense entry
    dense_by_cid: Dict[str, Dict[str, Any]] = {}
    for d in dense_list:
        dense_by_cid[d["chunk_id"]] = d

    # All unique chunk_ids
    all_cids = set(bm25_by_cid.keys()) | set(dense_by_cid.keys())

    # Collect raw scores for normalisation
    raw_bm25: List[float] = []
    raw_dense: List[float] = []
    for cid in all_cids:
        if cid in bm25_by_cid:
            raw_bm25.append(bm25_by_cid[cid].score)
        if cid in dense_by_cid:
            raw_dense.append(dense_by_cid[cid]["dense_score"])

    max_bm25 = max(raw_bm25) if raw_bm25 else 0.0
    max_dense = max(raw_dense) if raw_dense else 0.0

    merged: List[Dict[str, Any]] = []
    for cid in all_cids:
        bm25_result = bm25_by_cid.get(cid)
        dense_entry = dense_by_cid.get(cid)

        # Determine which chunk object to use
        chunk: Optional[Chunk] = None
        source_id = ""
        source_path = ""
        if bm25_result is not None:
            chunk = bm25_result.chunk
            source_id = chunk.source_id
            source_path = chunk.source_path
        elif dense_entry is not None:
            chunk = dense_entry["chunk"]
            source_id = dense_entry.get("source_id", "")
            source_path = dense_entry.get("source_path", "")

        if chunk is None:
            continue

        bm25_score = bm25_result.score if bm25_result is not None else 0.0
        dense_score_val = (
            dense_entry["dense_score"] if dense_entry is not None else 0.0
        )

        normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
        normalized_dense = max(0.0, min(1.0, dense_score_val))
        combined_score = 0.65 * normalized_bm25 + 0.35 * normalized_dense

        merged.append({
            "chunk_id": cid,
            "chunk": chunk,
            "bm25_score": round(bm25_score, 4),
            "dense_score": round(normalized_dense, 4),
            "combined_score": round(combined_score, 4),
            "source_id": source_id,
            "source_path": source_path,
        })

    return merged