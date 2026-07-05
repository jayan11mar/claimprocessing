"""Bucket evaluation failures by likely root cause."""

from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def bucket_failures(results: Sequence[Mapping[str, object]]) -> dict:
    """Return a simple root-cause bucket for observed failures."""
    buckets = {
        "retrieval": [],
        "answer_quality": [],
        "citations": [],
        "other": [],
    }

    for result in results:
        if not result.get("passed", True):
            reason = str(result.get("reason", "")).lower()
            if any(token in reason for token in ["evidence", "chunk", "retrieval", "missing"]):
                buckets["retrieval"].append(reason)
            elif any(token in reason for token in ["relevance", "answer", "quality", "correctness"]):
                buckets["answer_quality"].append(reason)
            elif any(token in reason for token in ["citation", "quote"]):
                buckets["citations"].append(reason)
            else:
                buckets["other"].append(reason)

    return buckets
