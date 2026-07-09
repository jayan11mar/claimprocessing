"""Hybrid retrieval that fuses BM25 and dense-style lexical overlap signals."""

import math
import re
from typing import Any, Callable, Dict, List, Optional

from app.rag.chunkers import Chunk
from app.rag.query_transform import build_query_variants
from app.rag.retriever_bm25 import bm25_retrieve
from app.rag.reranker import rerank_results


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


def hybrid_retrieve(
    chunks: List[Chunk],
    query: str,
    k: int = 5,
    embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    rerank: bool = True,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Fuse BM25 and dense embedding signals into a ranked list.

    Uses real dense embeddings via the provided embedding_fn (or loads one
    from config as fallback) rather than token-overlap approximations.

    Args:
        chunks: List of Chunk objects to search.
        query: The query string.
        k: Number of results to return.
        embedding_fn: Optional embedding function for dense scoring.
                      If None, loads from config automatically.
        rerank: Whether to apply cross-encoder reranking.
        metadata_filter: Optional dict of metadata fields to filter on.
                         Only chunks matching ALL filter criteria are returned.

    Returns:
        Ranked list of result dicts.
    """
    if not chunks:
        return []

    # Apply metadata filter early to reduce the search space
    if metadata_filter:
        filtered_chunks = []
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
                filtered_chunks.append(chunk)
        chunks = filtered_chunks
        if not chunks:
            return []

    # Load embedding function from config if not provided
    if embedding_fn is None:
        embedding_fn = _get_default_embedding_fn()

    variants = build_query_variants(query)
    bm25_results = bm25_retrieve(chunks, query, k=len(chunks))

    # Pre-compute embeddings for efficiency if embedding_fn is available
    chunk_texts = [r.chunk.text for r in bm25_results]
    chunk_embeddings = None
    query_embedding_vec = None
    if embedding_fn is not None:
        try:
            query_embedding_vec = embedding_fn([query])[0]
            chunk_embeddings = embedding_fn(chunk_texts)
        except Exception:
            pass

    scored_results: List[Dict[str, Any]] = []
    for idx, bm25_result in enumerate(bm25_results):
        chunk = bm25_result.chunk

        # Compute dense score using real embeddings when available
        dense_score = 0.0
        if chunk_embeddings is not None and query_embedding_vec is not None and idx < len(chunk_embeddings):
            try:
                dense_score = _compute_cosine_similarity(query_embedding_vec, chunk_embeddings[idx])
            except Exception:
                pass
        else:
            # Fall back to token overlap only when embeddings are unavailable
            dense_scores = [_token_overlap_score(chunk.text, variant) for variant in variants]
            dense_score = max(dense_scores) if dense_scores else 0.0

        max_bm25 = max(result.score for result in bm25_results) if bm25_results else 0.0
        normalized_bm25 = bm25_result.score / max_bm25 if max_bm25 else 0.0
        normalized_dense = max(0.0, min(1.0, dense_score))
        combined_score = 0.65 * normalized_bm25 + 0.35 * normalized_dense

        scored_results.append(
            {
                "chunk_id": f"{chunk.source_id}_{chunk.chunk_index}",
                "chunk": chunk,
                "bm25_score": round(bm25_result.score, 4),
                "dense_score": round(normalized_dense, 4),
                "combined_score": round(combined_score, 4),
                "source_id": chunk.source_id,
                "source_path": chunk.source_path,
                "query_variants": variants,
            }
        )

    scored_results.sort(key=lambda item: item["combined_score"], reverse=True)
    if rerank:
        scored_results = rerank_results(query, scored_results, top_k=k)
    return scored_results[:k]
