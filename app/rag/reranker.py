"""Reranking utilities for hybrid retrieval."""

import math
import os
import re
from typing import Any, Dict, List, Optional

try:
    import cohere
except ImportError:  # pragma: no cover - optional dependency guard
    cohere = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover - optional dependency guard
    CrossEncoder = None


def _token_overlap_score(query: str, text: str) -> float:
    tokens = {token for token in re.findall(r"[a-z0-9]+", query.lower()) if token}
    if not tokens:
        return 0.0
    text_tokens = {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}
    if not text_tokens:
        return 0.0
    overlap = len(tokens & text_tokens)
    return overlap / max(1, len(tokens))


def _fallback_rerank(query: str, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    reranked: List[Dict[str, Any]] = []
    for result in results:
        chunk = result.get("chunk")
        text = chunk.text if chunk is not None else ""
        lexical_score = _token_overlap_score(query, text)
        combined_score = result.get("combined_score", 0.0)
        adjusted_score = (0.7 * float(combined_score)) + (0.3 * lexical_score)
        reranked.append({**result, "rerank_score": round(adjusted_score, 4), "reranker": "fallback"})

    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return reranked[:top_k]


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int = 5,
    model: str = "rerank-english-v3.0",
) -> List[Dict[str, Any]]:
    """Rerank candidate chunks with Cohere when available, otherwise fall back to a cross-encoder-like lexical rerank."""
    if not results:
        return []

    api_key = os.getenv("COHERE_API_KEY")
    if api_key and cohere is not None:
        try:
            client = cohere.Client(api_key)
            documents = [result.get("chunk").text if result.get("chunk") is not None else "" for result in results]
            response = client.rerank(
                model=model,
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
            )
            reranked: List[Dict[str, Any]] = []
            for item in getattr(response, "results", []) or []:
                idx = getattr(item, "index", None)
                if idx is None and isinstance(item, dict):
                    idx = item.get("index")
                if idx is None:
                    continue
                if idx < len(results):
                    candidate = dict(results[idx])
                    score = getattr(item, "relevance_score", None)
                    if score is None and isinstance(item, dict):
                        score = item.get("relevance_score")
                    candidate["rerank_score"] = round(float(score or 0.0), 4)
                    candidate["reranker"] = "cohere"
                    reranked.append(candidate)
            if reranked:
                return reranked
        except Exception:
            pass

    cross_encoder_model = os.getenv("RERANK_CROSS_ENCODER_MODEL")
    if CrossEncoder is not None and cross_encoder_model:
        try:
            encoder = CrossEncoder(cross_encoder_model)
            pairs = [(query, result.get("chunk").text if result.get("chunk") is not None else "") for result in results]
            scores = encoder.predict(pairs)
            reranked = []
            for result, score in zip(results, scores):
                candidate = dict(result)
                candidate["rerank_score"] = round(float(score), 4)
                candidate["reranker"] = "cross-encoder"
                reranked.append(candidate)
            reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
            return reranked[:top_k]
        except Exception:
            pass

    return _fallback_rerank(query, results, top_k)
