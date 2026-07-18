"""Baseline management for drift detection.

Stores and retrieves baseline evaluation results for comparison
against current system performance.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.rag.embeddings import get_embedding_fn
from app.rag.qa_chain import run_qa_chain

logger = logging.getLogger(__name__)

# Regex to count inline citations like [chunk_id], [12], [doc_3], etc.
_CITATION_RE = re.compile(r"\[[\w_.-]+\]")


def _count_citations(answer_text: str) -> int:
    """Count the number of citation markers in an answer string."""
    return len(_CITATION_RE.findall(answer_text))


def _check_format_ok(answer_text: str) -> bool:
    """Heuristic check that the answer is well-formed.

    Returns ``True`` when the answer contains at least one word and is
    not merely a boilerplate fallback message.
    """
    text = answer_text.strip()
    if not text:
        return False
    # Reject trivial fallback / empty-responses
    if len(text.split()) < 3:
        return False
    return True


def _get_embedding(texts: List[str]) -> List[List[float]]:
    """Obtain embeddings for a list of strings using the app's embedding model.

    Falls back to a zero vector of the default dimension on failure.
    """
    try:
        embed_fn = get_embedding_fn()
        return embed_fn(texts)
    except Exception:
        logger.warning("Embedding call failed; returning zero vectors", exc_info=True)
        return [[0.0] * 1536 for _ in texts]


def _compute_embedding_stats(
    embeddings: List[List[float]],
) -> Dict[str, Any]:
    """Compute aggregate embedding distribution statistics.

    Returns a dict containing:
        - ``mean_vector`` (list[float]): per-dimension mean.
        - ``std_vector`` (list[float]): per-dimension standard deviation
          (population std; zero for single-sample baselines).
        - ``dimension`` (int): embedding dimensionality.
        - ``count`` (int): number of embeddings aggregated.
    """
    if not embeddings:
        return {"mean_vector": [], "std_vector": [], "dimension": 0, "count": 0}

    arr = np.array(embeddings, dtype=np.float64)
    mean_vec = arr.mean(axis=0).tolist()
    std_vec = arr.std(axis=0, ddof=0).tolist()  # population std

    return {
        "mean_vector": mean_vec,
        "std_vector": std_vec,
        "dimension": arr.shape[1],
        "count": arr.shape[0],
    }


def snapshot_baseline(
    cases: List[Dict[str, Any]],
    out_path: str = "reports/drift_baseline.json",
) -> Dict[str, Any]:
    """Run the QA chain on each case and persist a baseline snapshot.

    For each case dict the ``question`` key is used as the query.
    Stores per-case results together with aggregate embedding-distribution
    statistics (used later by the embedding-drift step).

    Args:
        cases: List of case dicts. Each must contain at least ``question``.
        out_path: Filesystem path for the output JSON baseline.

    Returns:
        The full baseline dict that was written to disk.
    """
    results: List[Dict[str, Any]] = []
    answer_texts: List[str] = []

    for idx, case in enumerate(cases):
        question: str = case.get("question", "")
        if not question:
            logger.warning("Case %d has no 'question' key; skipping.", idx)
            continue

        # ── Run the existing QA chain ────────────────────────────────────
        try:
            qa_result = run_qa_chain(query=question)
        except Exception:
            logger.exception("QA chain failed for question %r", question)
            qa_result = {
                "answer_text": "",
                "citations": [],
                "confidence": 0.0,
            }

        answer_text: str = qa_result.get("answer_text", "")
        citations: List[Any] = qa_result.get("citations", [])
        citation_count: int = _count_citations(answer_text) or len(citations)
        format_ok: bool = _check_format_ok(answer_text)

        # ── Embed the answer text ────────────────────────────────────────
        emb = _get_embedding([answer_text])[0]

        results.append({
            "question": question,
            "answer_text": answer_text,
            "answer_embedding": emb,
            "citation_count": citation_count,
            "format_ok": format_ok,
        })
        answer_texts.append(answer_text)

    # ── Aggregate embedding distribution stats ───────────────────────────
    all_embeddings = [r["answer_embedding"] for r in results]
    embedding_stats = _compute_embedding_stats(all_embeddings)

    baseline = {
        "snapshot_version": "1.0",
        "total_cases": len(results),
        "embedding_stats": embedding_stats,
        "results": results,
    }

    # ── Persist to disk ──────────────────────────────────────────────────
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline, indent=2, default=str), encoding="utf-8")
    logger.info("Drift baseline written to %s (%d cases).", out_path, len(results))

    return baseline


def load_baseline(path: str) -> Dict[str, Any]:
    """Load a previously persisted drift baseline from a JSON file.

    Args:
        path: Filesystem path to the baseline JSON.

    Returns:
        The baseline dict (same structure as returned by :func:`snapshot_baseline`).
    """
    raw = Path(path).read_text(encoding="utf-8")
    return json.loads(raw)