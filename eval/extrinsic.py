"""Extrinsic answer quality metrics for RAG evaluation."""

import re
from typing import Optional, Sequence


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = set(_normalize(left).split())
    right_tokens = set(_normalize(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    return round(len(overlap) / max(1, len(right_tokens)), 3)


def compute_extrinsic_metrics(answer: str, expected_answer: str, retrieved_chunks: Optional[Sequence[str]] = None) -> dict:
    """Compute faithfulness, answer correctness, and answer relevance."""
    retrieved_chunks = retrieved_chunks or []
    answer_score = _token_overlap_score(answer, expected_answer)
    retrieved_text = " ".join(retrieved_chunks)
    faithfulness = _token_overlap_score(answer, retrieved_text)
    relevance = _token_overlap_score(answer, expected_answer)
    if retrieved_chunks and answer_score == 0.0:
        relevance = 0.0

    return {
        "faithfulness": round(max(answer_score, faithfulness), 3),
        "answer_correctness": round(answer_score, 3),
        "answer_relevance": round(max(answer_score, relevance), 3),
    }
