"""Embedding drift detection.

Monitors embedding space shifts over time to detect
distributional changes in the retrieval corpus.

Compares a current set of answer embeddings (produced by the same
:func:`~app.drift.baseline.snapshot_baseline` pipeline) against a
previously persisted baseline.  Three complementary signals are computed:

1. **kl_divergence** — KL divergence between binned distributions of
   all embedding values (flattened across cases), with a small epsilon
   to avoid division by zero.
2. **spearman_rank_corr** — Spearman rank correlation between the
   per-dimension mean vector of the baseline and the per-dimension mean
   vector of the current set.
3. **nn_stability** — For each item in the smaller set, the fraction of
   its top-k nearest neighbours (by cosine distance) that overlap with
   those found in the other set, averaged over all items.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

from app.drift.baseline import load_baseline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPS = 1e-12


def _flatten_embeddings(results: List[Dict[str, Any]]) -> np.ndarray:
    """Extract and stack all answer_embedding vectors into a 2-D array.

    Returns shape ``(n_cases, dim)``.
    """
    vecs = [r["answer_embedding"] for r in results if r.get("answer_embedding")]
    if not vecs:
        return np.empty((0, 0), dtype=np.float64)
    return np.array(vecs, dtype=np.float64)


def _compute_kl_divergence(
    baseline_flat: np.ndarray,
    current_flat: np.ndarray,
    n_bins: int = 50,
) -> float:
    """KL divergence between binned distributions of *all* embedding values.

    Both arrays are flattened to 1-D, binned into *n_bins* equally-spaced
    bins spanning the combined range, and the KL divergence
    ``D_KL(P || Q)`` is computed where:

    - *P* = baseline distribution
    - *Q* = current distribution

    A small epsilon (``1e-12``) is added to both histograms before
    normalisation to avoid division-by-zero.
    """
    if baseline_flat.size == 0 or current_flat.size == 0:
        return 0.0

    combined = np.concatenate([baseline_flat, current_flat])
    lo, hi = combined.min(), combined.max()
    if hi - lo < _EPS:
        return 0.0

    bins = np.linspace(lo, hi, n_bins + 1)

    p_hist, _ = np.histogram(baseline_flat, bins=bins)
    q_hist, _ = np.histogram(current_flat, bins=bins)

    p_hist = p_hist.astype(np.float64) + _EPS
    q_hist = q_hist.astype(np.float64) + _EPS

    p_dist = p_hist / p_hist.sum()
    q_dist = q_hist / q_hist.sum()

    # D_KL(P || Q) = sum(P * log(P / Q))
    kl = np.sum(p_dist * np.log(p_dist / q_dist))
    return float(kl)


def _compute_spearman_rank_corr(
    baseline_mean: List[float],
    current_mean: List[float],
) -> float:
    """Spearman rank correlation between two per-dimension mean vectors."""
    if not baseline_mean or not current_mean:
        return 0.0
    min_len = min(len(baseline_mean), len(current_mean))
    if min_len < 2:
        return 0.0
    corr, _ = spearmanr(baseline_mean[:min_len], current_mean[:min_len])
    if isinstance(corr, (np.floating, float)):
        return float(corr)
    return 0.0


def _cosine_sim_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix for a 2-D array (rows = items)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < _EPS, 1.0, norms)
    unit = vectors / norms
    return unit @ unit.T


def _top_k_indices(sim_row: np.ndarray, k: int, exclude_self: bool = True) -> List[int]:
    """Return indices of top-k largest values in *sim_row* (excluding self)."""
    n = len(sim_row)
    if k >= n:
        cand = list(range(n))
    else:
        idx = np.argpartition(sim_row, n - k)[-k:]
        cand = idx.tolist()
    cand.sort(key=lambda i: sim_row[i], reverse=True)
    if exclude_self:
        cand = [i for i in cand if i != int(sim_row.argmax())]
    return cand[:k]


def _nn_stability(
    baseline_vecs: np.ndarray,
    current_vecs: np.ndarray,
    k: int = 5,
) -> float:
    """Nearest-neighbour stability score.

    For each item (using the **smaller** set as the probe set), finds its
    top-k neighbours within its own set (excluding itself) and its top-k
    neighbours in the *other* set (excluding the probe item's counterpart
    match).  Returns the average Jaccard-like overlap ratio across all
    probe items.

    This mirrors: "overlap ratio of top-k nearest neighbors (baseline set
    vs current set), averaged".
    """
    n_b = baseline_vecs.shape[0]
    n_c = current_vecs.shape[0]

    if n_b == 0 or n_c == 0:
        return 0.0

    k = min(k, n_b, n_c)

    # ── Normalise all vectors first (ensures consistent dot-product
    #    similarity across same-set and cross-set comparisons) ───────────
    b_norms = np.linalg.norm(baseline_vecs, axis=1, keepdims=True)
    c_norms = np.linalg.norm(current_vecs, axis=1, keepdims=True)
    b_norms = np.where(b_norms < _EPS, 1.0, b_norms)
    c_norms = np.where(c_norms < _EPS, 1.0, c_norms)
    b_unit = baseline_vecs / b_norms   # (n_b, dim)
    c_unit = current_vecs / c_norms    # (n_c, dim)

    sim_bb = b_unit @ b_unit.T  # (n_b, n_b) — diagonal is exactly 1.0
    sim_cc = c_unit @ c_unit.T  # (n_c, n_c)
    sim_bc = b_unit @ c_unit.T  # (n_b, n_c)

    def _top_k_excluding(sim_row: np.ndarray, exclude_idx: int, kk: int) -> List[int]:
        """Return indices of top-k values in *sim_row*, excluding *exclude_idx*."""
        # Set excluded index to -inf so it can never be in the top-k
        row = sim_row.copy()
        row[exclude_idx] = -np.inf
        if kk >= len(row):
            return list(range(len(row)))
        idx = np.argpartition(row, len(row) - kk)[-kk:]
        # Sort descending by similarity
        idx = idx[np.argsort(row[idx])[::-1]]
        return idx.tolist()[:kk]

    # Use the smaller set as the probe set
    if n_b <= n_c:
        overlap_ratios = []
        for i in range(n_b):
            # Own neighbourhood: exclude self (index i)
            nn_own = set(_top_k_excluding(sim_bb[i], exclude_idx=i, kk=k))
            # Cross neighbourhood: find the probe's best match in C, exclude that
            match_idx = int(np.argmax(sim_bc[i]))
            nn_cross = set(_top_k_excluding(sim_bc[i], exclude_idx=match_idx, kk=k))
            overlap = len(nn_own & nn_cross)
            overlap_ratios.append(overlap / k)
    else:
        overlap_ratios = []
        sim_cb = sim_bc.T  # (n_c, n_b)
        for i in range(n_c):
            nn_own = set(_top_k_excluding(sim_cc[i], exclude_idx=i, kk=k))
            match_idx = int(np.argmax(sim_cb[i]))
            nn_cross = set(_top_k_excluding(sim_cb[i], exclude_idx=match_idx, kk=k))
            overlap = len(nn_own & nn_cross)
            overlap_ratios.append(overlap / k)

    return float(np.mean(overlap_ratios)) if overlap_ratios else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_embedding_drift(
    current_results: List[Dict[str, Any]],
    baseline_path: str = "reports/drift_baseline.json",
    kl_bins: int = 50,
    nn_k: int = 5,
) -> Dict[str, Any]:
    """Compute embedding-drift signals by comparing current to baseline.

    Each item in *current_results* must contain ``answer_embedding``
    (a ``list[float]``).

    Args:
        current_results: List of per-case result dicts from a fresh
            evaluation run (same structure as baseline results).
        baseline_path: Path to the baseline JSON written by
            :func:`~app.drift.baseline.snapshot_baseline`.
        kl_bins: Number of bins for the KL-divergence histogram.
        nn_k: ``k`` for top-k nearest-neighbour stability.

    Returns:
        A dict with keys ``kl_divergence``, ``spearman_rank_corr``,
        ``nn_stability``.
    """
    baseline = load_baseline(baseline_path)
    baseline_results: List[Dict[str, Any]] = baseline.get("results", [])

    if not baseline_results or not current_results:
        logger.warning(
            "Insufficient data for embedding-drift comparison "
            "(baseline=%d, current=%d).",
            len(baseline_results),
            len(current_results),
        )
        return {
            "kl_divergence": 0.0,
            "spearman_rank_corr": 0.0,
            "nn_stability": 0.0,
        }

    # ── Flatten embedding arrays ────────────────────────────────────────
    b_vecs = _flatten_embeddings(baseline_results)
    c_vecs = _flatten_embeddings(current_results)

    if b_vecs.size == 0 or c_vecs.size == 0:
        return {
            "kl_divergence": 0.0,
            "spearman_rank_corr": 0.0,
            "nn_stability": 0.0,
        }

    # 1. KL divergence (flatten all values into one distribution)
    kl = _compute_kl_divergence(
        b_vecs.ravel(),
        c_vecs.ravel(),
        n_bins=kl_bins,
    )

    # 2. Spearman rank correlation on per-dimension means
    b_mean = b_vecs.mean(axis=0).tolist()
    c_mean = c_vecs.mean(axis=0).tolist()
    sp = _compute_spearman_rank_corr(b_mean, c_mean)

    # 3. NN stability
    nn = _nn_stability(b_vecs, c_vecs, k=nn_k)

    return {
        "kl_divergence": round(kl, 6),
        "spearman_rank_corr": round(sp, 6),
        "nn_stability": round(nn, 6),
    }


# ---------------------------------------------------------------------------
# Inline self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Identical distributions → KL ≈ 0, Spearman ≈ 1.0, nn_stability ≈ 1.0
    rng = np.random.default_rng(42)
    vecs = rng.normal(loc=0.0, scale=1.0, size=(20, 8)).tolist()

    # Build fake baseline + current results with the same vectors
    fake_results = [
        {"answer_embedding": v, "answer_text": "x", "citation_count": 1}
        for v in vecs
    ]

    b_vecs = _flatten_embeddings(fake_results)
    c_vecs = _flatten_embeddings(fake_results)

    # KL
    kl = _compute_kl_divergence(b_vecs.ravel(), c_vecs.ravel(), n_bins=50)
    assert abs(kl) < 1e-6, f"Expected KL ≈ 0, got {kl}"

    # Spearman
    b_mean = b_vecs.mean(axis=0).tolist()
    c_mean = c_vecs.mean(axis=0).tolist()
    sp = _compute_spearman_rank_corr(b_mean, c_mean)
    assert abs(sp - 1.0) < 1e-9, f"Expected Spearman ≈ 1.0, got {sp}"

    # NN stability
    nn = _nn_stability(b_vecs, c_vecs, k=5)
    assert abs(nn - 1.0) < 1e-6, f"Expected NN stability ≈ 1.0, got {nn}"

    print(f"Self-test: KL = {kl:.6f}  (expect ≈0)")
    print(f"Self-test: Spearman = {sp:.6f}  (expect ≈1.0)")
    print(f"Self-test: NN stability = {nn:.6f}  (expect ≈1.0)")
    print("All embedding-drift self-tests passed.")