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


def hybrid_retrieve(
    chunks: List[Chunk],
    query: str,
    k: int = 5,
    embedding_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    rerank: bool = True,
) -> List[Dict[str, Any]]:
    """Fuse BM25 and a lightweight dense-style score into a ranked list."""
    if not chunks:
        return []

    variants = build_query_variants(query)
    bm25_results = bm25_retrieve(chunks, query, k=len(chunks))

    scored_results: List[Dict[str, Any]] = []
    for bm25_result in bm25_results:
        chunk = bm25_result.chunk
        dense_scores = [_token_overlap_score(chunk.text, variant) for variant in variants]
        dense_score = max(dense_scores) if dense_scores else 0.0

        if embedding_fn is not None:
            try:
                query_embedding = embedding_fn([query])[0]
                chunk_embedding = embedding_fn([chunk.text])[0]
                if query_embedding and chunk_embedding:
                    dot_product = sum(a * b for a, b in zip(query_embedding, chunk_embedding))
                    magnitude = math.sqrt(sum(a * a for a in query_embedding)) * math.sqrt(sum(b * b for b in chunk_embedding))
                    dense_score = dot_product / magnitude if magnitude else dense_score
            except Exception:
                pass

        max_bm25 = max(result.score for result in bm25_results) if bm25_results else 0.0
        normalized_bm25 = bm25_result.score / max_bm25 if max_bm25 else 0.0
        normalized_dense = dense_score
        combined_score = 0.65 * normalized_bm25 + 0.35 * normalized_dense

        scored_results.append(
            {
                "chunk_id": f"{chunk.source_id}_{chunk.chunk_index}",
                "chunk": chunk,
                "bm25_score": round(bm25_result.score, 4),
                "dense_score": round(dense_score, 4),
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
