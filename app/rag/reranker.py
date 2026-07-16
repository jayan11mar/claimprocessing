"""Reranking utilities for hybrid retrieval using a cross-encoder model.

The cross-encoder is loaded once at module level (cached) and reused across
queries.  No API key is required — the model runs fully offline.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# ── Module-level cross-encoder cache (loaded once) ──────────────────────
_CROSS_ENCODER: Optional[CrossEncoder] = None
_CROSS_ENCODER_MODEL_NAME: Optional[str] = None


def _get_cross_encoder(model_name: str) -> CrossEncoder:
    """Return a cached CrossEncoder instance, loading it on first call."""
    global _CROSS_ENCODER, _CROSS_ENCODER_MODEL_NAME
    if _CROSS_ENCODER is not None and _CROSS_ENCODER_MODEL_NAME == model_name:
        return _CROSS_ENCODER
    logger.info("Loading cross-encoder model: %s", model_name)
    _CROSS_ENCODER = CrossEncoder(model_name)
    _CROSS_ENCODER_MODEL_NAME = model_name
    return _CROSS_ENCODER


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int = 5,
    model: Optional[str] = None,
    max_rerank_candidates: int = 20,
) -> List[Dict[str, Any]]:
    """Rerank candidate chunks with a cross-encoder.

    Every result in the returned list carries a ``rerank_score`` field.
    The function logs the top-*k* ordering *before* and *after* reranking
    for verifiability.

    The cross-encoder is loaded **once** at module level and reused across
    all requests (lazy singleton).  This function also caps the number of
    candidates sent to the model to *max_rerank_candidates* (default 20)
    so that inference cost stays bounded.

    Args:
        query: The user query string.
        results: List of result dicts from hybrid fusion (must have ``chunk`` key).
        top_k: Number of results to return after reranking.
        model: Cross-encoder model name.  Falls back to the
               ``RERANKER_MODEL`` env var, then to
               ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
        max_rerank_candidates: Maximum number of candidates to score with
                               the cross-encoder.  Results beyond this count
                               are discarded before scoring.  Default 20.

    Returns:
        Reranked list of result dicts (each with ``rerank_score``).

    Raises:
        RuntimeError: If the cross-encoder cannot be loaded or scores
                      cannot be computed.
    """
    if not results:
        return []

    # ── Cap candidates to keep inference bounded ────────────────────────
    if len(results) > max_rerank_candidates:
        logger.info(
            "Trimming reranker input from %d to %d candidates (max_rerank_candidates=%d).",
            len(results), max_rerank_candidates, max_rerank_candidates,
        )
    candidates = results[:max_rerank_candidates]

    # ── Resolve model name ──────────────────────────────────────────────
    if model is None:
        model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ── Log pre-rerank ordering ─────────────────────────────────────────
    pre_order = [
        {
            "chunk_id": r.get("chunk_id", "?"),
            "combined_score": r.get("combined_score", 0.0),
        }
        for r in candidates[:top_k]
    ]
    logger.info("PRE-RERANK top-%d order: %s", top_k, pre_order)

    # ── Load cross-encoder ──────────────────────────────────────────────
    try:
        encoder = _get_cross_encoder(model)
    except Exception as exc:
        logger.error("Failed to load cross-encoder '%s': %s", model, exc)
        raise RuntimeError(
            f"Cross-encoder '{model}' could not be loaded. "
            "Install sentence-transformers and ensure the model name is correct."
        ) from exc

    # ── Build (query, text) pairs ───────────────────────────────────────
    pairs: List[tuple] = []
    for result in candidates:
        chunk = result.get("chunk")
        text = chunk.text if chunk is not None else ""
        pairs.append((query, text))

    # ── Predict relevance scores ────────────────────────────────────────
    try:
        scores = encoder.predict(pairs)
    except Exception as exc:
        logger.error("Cross-encoder prediction failed: %s", exc)
        raise RuntimeError("Cross-encoder prediction failed.") from exc

    # ── Attach rerank_score to each result ──────────────────────────────
    reranked: List[Dict[str, Any]] = []
    for result, score in zip(candidates, scores):
        candidate = dict(result)
        candidate["rerank_score"] = round(float(score), 4)
        candidate["reranker"] = "cross-encoder"
        reranked.append(candidate)

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

    # ── Log post-rerank ordering ────────────────────────────────────────
    post_order = [
        {
            "chunk_id": r.get("chunk_id", "?"),
            "rerank_score": r.get("rerank_score", 0.0),
        }
        for r in reranked[:top_k]
    ]
    logger.info("POST-RERANK top-%d order: %s", top_k, post_order)

    return reranked[:top_k]
