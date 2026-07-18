"""Prompt drift detection.

Monitors changes in prompt effectiveness and response patterns
over time to detect semantic or behavioral drift.

Computes three signals by comparing current evaluation results against
a previously persisted baseline (loaded via :func:`~app.drift.baseline.load_baseline`):

1. **semantic_shift** — mean cosine similarity between current and baseline
   answer embeddings (per case, then averaged).
2. **format_compliance** — fraction of current answers passing the same
   ``format_ok`` heuristic used during baseline creation.
3. **citation_density_change** — mean(current citation count) minus
   mean(baseline citation count).
"""

import logging
import math
from typing import Any, Dict, List, Optional

from app.drift.baseline import _check_format_ok, load_baseline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors.

    Returns a value in ``[-1.0, 1.0]`` (or ``0.0`` if either vector is
    all-zeros).
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        norm_a += ai * ai
        norm_b += bi * bi

    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom == 0.0:
        return 0.0
    return dot / denom


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_prompt_drift(
    current_results: List[Dict[str, Any]],
    baseline_path: str = "reports/drift_baseline.json",
) -> Dict[str, Any]:
    """Compute prompt-drift signals by comparing current results to a baseline.

    Each item in *current_results* must contain the same keys produced by
    :func:`~app.drift.baseline.snapshot_baseline`:

    - ``answer_embedding`` (list[float])
    - ``answer_text`` (str)
    - ``citation_count`` (int)

    Args:
        current_results: List of per-case result dicts from a fresh
            evaluation run.
        baseline_path: Path to the baseline JSON written by
            :func:`~app.drift.baseline.snapshot_baseline`.

    Returns:
        A dict with keys ``semantic_shift``, ``format_compliance``,
        ``citation_density_change``.
    """
    baseline = load_baseline(baseline_path)
    baseline_results: List[Dict[str, Any]] = baseline.get("results", [])

    if not baseline_results or not current_results:
        logger.warning(
            "Insufficient data for prompt-drift comparison "
            "(baseline=%d, current=%d).",
            len(baseline_results),
            len(current_results),
        )
        return {
            "semantic_shift": 0.0,
            "format_compliance": 0.0,
            "citation_density_change": 0.0,
        }

    # Align by index (assumes same ordering as baseline)
    n = min(len(baseline_results), len(current_results))

    cosine_sims: List[float] = []
    format_oks: List[bool] = []
    current_citations: List[int] = []
    baseline_citations: List[int] = []

    for i in range(n):
        b = baseline_results[i]
        c = current_results[i]

        # 1. Semantic shift — cosine similarity of answer embeddings
        sim = _cosine_similarity(
            b.get("answer_embedding", []),
            c.get("answer_embedding", []),
        )
        cosine_sims.append(sim)

        # 2. Format compliance
        answer_text = c.get("answer_text", "")
        format_oks.append(_check_format_ok(answer_text))

        # 3. Citation density
        current_citations.append(c.get("citation_count", 0))
        baseline_citations.append(b.get("citation_count", 0))

    semantic_shift = sum(cosine_sims) / len(cosine_sims) if cosine_sims else 0.0
    format_compliance = sum(format_oks) / len(format_oks) if format_oks else 0.0
    citation_density_change = (
        (sum(current_citations) / len(current_citations))
        - (sum(baseline_citations) / len(baseline_citations))
        if current_citations and baseline_citations
        else 0.0
    )

    return {
        "semantic_shift": round(semantic_shift, 6),
        "format_compliance": round(format_compliance, 6),
        "citation_density_change": round(citation_density_change, 6),
    }


# ---------------------------------------------------------------------------
# Inline self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Verify cosine similarity ≈ 1.0 for identical vectors
    v = [0.1, 0.3, 0.7, 0.2, 0.9]
    sim = _cosine_similarity(v, v)
    assert abs(sim - 1.0) < 1e-9, f"Expected 1.0, got {sim}"
    print(f"Self-test passed: cosine(identical) = {sim}")

    # Verify cosine similarity ≈ 0.0 for orthogonal vectors
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    sim_orth = _cosine_similarity(a, b)
    assert abs(sim_orth) < 1e-9, f"Expected 0.0, got {sim_orth}"
    print(f"Self-test passed: cosine(orthogonal) = {sim_orth}")

    print("All prompt-drift self-tests passed.")