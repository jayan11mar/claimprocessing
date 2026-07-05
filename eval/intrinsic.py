"""Intrinsic retrieval metrics for RAG evaluation."""

import math
import re
from typing import Iterable, List, Sequence


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = set(_normalize(left).split())
    right_tokens = set(_normalize(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    return round(len(overlap) / max(1, len(right_tokens)), 3)


def compute_intrinsic_metrics(retrieved_chunks: Sequence[str], expected_chunks: Sequence[str], k: int = 5) -> dict:
    """Compute hit@k, MRR, NDCG, context precision, and context recall."""
    retrieved = [chunk for chunk in retrieved_chunks[:k] if chunk]
    expected = [chunk for chunk in expected_chunks if chunk]

    hit_at_k = 0.0
    mrr = 0.0
    precision_hits = 0
    matched_expected = set()

    for position, chunk in enumerate(retrieved, start=1):
        for expected_index, expected_chunk in enumerate(expected):
            if expected_index in matched_expected:
                continue
            if _token_overlap_score(chunk, expected_chunk) >= 0.5:
                precision_hits += 1
                if hit_at_k == 0.0:
                    hit_at_k = 1.0
                if mrr == 0.0:
                    mrr = 1.0 / position
                matched_expected.add(expected_index)
                break

    context_precision = round(precision_hits / max(1, len(retrieved)), 3)
    context_recall = round(len(matched_expected) / max(1, len(expected)), 3) if expected else 1.0

    dcg = 0.0
    for position, chunk in enumerate(retrieved, start=1):
        if any(_token_overlap_score(chunk, expected_chunk) >= 0.5 for expected_chunk in expected):
            dcg += 1.0 / math.log2(position + 1)

    ideal_relevant = min(len(expected), len(retrieved))
    idcg = 0.0
    for position in range(1, ideal_relevant + 1):
        idcg += 1.0 / math.log2(position + 1)

    ndcg = round(dcg / idcg, 3) if idcg > 0 else 1.0

    return {
        "hit_at_k": hit_at_k,
        "mrr": round(mrr, 3),
        "ndcg": ndcg,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
