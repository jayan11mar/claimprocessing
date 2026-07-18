"""Tests for drift detection module.

Uses small SYNTHETIC embeddings/fixtures — does NOT call the real LLM
or embedding model.

All tests seed ``np.random`` for determinism.
"""

import json
import os
import tempfile
from typing import Any, Dict, List
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.drift.alerts import raise_alerts
from app.drift.embedding_drift import compute_embedding_drift
from app.drift.prompt_drift import compute_prompt_drift
from app.drift.detector import run_drift


# =========================================================================
# Helpers
# =========================================================================


def _build_fake_baseline(
    n: int = 5,
    dim: int = 8,
    seed: int = 42,
) -> Dict[str, Any]:
    """Build a synthetic baseline dict with random embeddings.

    Every answer is well-formed (format_ok=True) and carries a positive
    citation count.  All vectors are drawn from the same seeded normal
    distribution for reproducibility.
    """
    rng = np.random.default_rng(seed)
    vectors = rng.normal(loc=0.0, scale=1.0, size=(n, dim)).tolist()

    results: List[Dict[str, Any]] = []
    for i in range(n):
        results.append({
            "question": f"test_question_{i}",
            "answer_text": (
                f"This is a well-formed answer number {i} with "
                f"sufficient length for testing purposes."
            ),
            "answer_embedding": vectors[i],
            "citation_count": 2,
            "format_ok": True,
        })

    return {
        "snapshot_version": "1.0",
        "total_cases": n,
        "embedding_stats": {
            "mean_vector": list(np.mean(vectors, axis=0)),
            "std_vector": list(np.std(vectors, axis=0, ddof=0)),
            "dimension": dim,
            "count": n,
        },
        "results": results,
    }


def _identical_results(baseline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a deep-copied list of results identical to the baseline."""
    return [dict(r) for r in baseline["results"]]


def _write_baseline(tmp_path_str: str, baseline: Dict[str, Any]) -> str:
    """Persist a baseline dict to a temporary JSON file and return its path."""
    path = os.path.join(tmp_path_str, "drift_baseline.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(baseline, fh)
    return path


# =========================================================================
# Test 1 – No-drift (identical baseline and current)
# =========================================================================


class TestNoDriftIdentical:
    """1. test_no_drift_identical

    Build a fake baseline dict and an identical "current" set, then verify
    all drift signals indicate no drift.
    """

    def test_no_drift_identical(self, tmp_path: Any) -> None:
        np.random.seed(42)

        baseline = _build_fake_baseline(n=5, dim=8, seed=42)
        current = _identical_results(baseline)
        baseline_path = _write_baseline(str(tmp_path), baseline)

        # ── Prompt drift ────────────────────────────────────────────────
        prompt = compute_prompt_drift(current, baseline_path=baseline_path)
        assert prompt["semantic_shift"] >= 0.99, (
            f"semantic_shift={prompt['semantic_shift']} (expected >= 0.99)"
        )
        assert prompt["format_compliance"] >= 0.99, (
            f"format_compliance={prompt['format_compliance']} (expected >= 0.99)"
        )
        assert abs(prompt["citation_density_change"]) <= 0.01, (
            f"citation_density_change={prompt['citation_density_change']} "
            f"(expected ≈ 0)"
        )

        # ── Embedding drift ─────────────────────────────────────────────
        embedding = compute_embedding_drift(current, baseline_path=baseline_path)
        assert embedding["kl_divergence"] <= 0.01, (
            f"kl_divergence={embedding['kl_divergence']} (expected <= 0.01)"
        )
        assert embedding["spearman_rank_corr"] >= 0.99, (
            f"spearman_rank_corr={embedding['spearman_rank_corr']} "
            f"(expected >= 0.99)"
        )
        assert embedding["nn_stability"] >= 0.99, (
            f"nn_stability={embedding['nn_stability']} (expected >= 0.99)"
        )


# =========================================================================
# Test 2 – Drift detected (degraded / perturbed)
# =========================================================================


class TestDriftDetectedDegraded:
    """2. test_drift_detected_degraded

    Take the baseline and build a "current" set with shuffled/perturbed
    embeddings, then assert that ``run_drift`` reports at least one breach.
    """

    def test_drift_detected_degraded(self, tmp_path: Any) -> None:
        np.random.seed(42)

        baseline = _build_fake_baseline(n=5, dim=8, seed=42)
        baseline_path = _write_baseline(str(tmp_path), baseline)

        # Build "current" results by adding heavy noise + shuffling dims
        rng = np.random.default_rng(99)
        current_results: List[Dict[str, Any]] = []
        for r in baseline["results"]:
            emb = np.array(r["answer_embedding"], dtype=np.float64)
            noise = rng.normal(loc=0.0, scale=2.0, size=emb.shape)
            perturbed = emb + noise
            rng.shuffle(perturbed)
            entry = dict(r)
            entry["answer_embedding"] = perturbed.tolist()
            current_results.append(entry)

        # Patch ``snapshot_baseline`` so ``run_drift`` never calls the real
        # QA chain or embedding model.
        with patch("app.drift.detector.snapshot_baseline") as mock_snapshot:
            mock_snapshot.return_value = {"results": current_results}

            result = run_drift(
                cases=[{"question": r["question"]} for r in current_results],
                baseline_path=baseline_path,
                thresholds_path="config/drift_thresholds.yaml",
            )

        assert result["any_breach"] is True, (
            f"Expected any_breach=True, got {result['any_breach']}"
        )
        assert len(result["breaches"]) >= 1, (
            f"Expected >= 1 breach, got {len(result['breaches'])}: "
            f"{result['breaches']}"
        )


# =========================================================================
# Test 3 – Alerting (conservative behaviour)
# =========================================================================


class TestAlertsConservative:
    """3. test_alerts_conservative

    - ``raise_alerts`` on a within-threshold result returns [] (empty).
    - ``raise_alerts`` on a breached result returns a non-empty list of
      strings.
    """

    def test_alerts_empty_when_no_breach(self) -> None:
        result: Dict[str, Any] = {
            "scores": {"semantic_shift": 0.95, "kl_divergence": 0.05},
            "breaches": [],
            "any_breach": False,
        }
        alerts = raise_alerts(result)
        assert alerts == [], f"Expected empty list, got {alerts}"

    def test_alerts_nonempty_when_breach(self) -> None:
        result: Dict[str, Any] = {
            "scores": {"semantic_shift": 0.5},
            "breaches": [
                {
                    "metric": "semantic_shift",
                    "value": 0.5,
                    "threshold": 0.85,
                    "direction": "below",
                },
                {
                    "metric": "kl_divergence",
                    "value": 0.25,
                    "threshold": 0.10,
                    "direction": "above",
                },
            ],
            "any_breach": True,
        }
        alerts = raise_alerts(result)
        assert len(alerts) >= 1, f"Expected >= 1 alert, got {alerts}"
        assert all(isinstance(a, str) for a in alerts), (
            f"All alerts should be strings; got {alerts}"
        )


# =========================================================================
# Test 4 – Endpoint disabled by default
# =========================================================================


class TestEndpointDisabledByDefault:
    """4. test_endpoint_disabled_by_default

    POST /eval/drift with body {} while ENABLE_DRIFT is OFF (default
    ``False``).  Must return HTTP 200 with ``{"enabled": false}``.
    """

    def test_endpoint_disabled_by_default(self) -> None:
        # Patch get_settings on the server module (the reference the
        # endpoint actually calls at runtime).
        from app.api.server import app  # safe import; no real chain call

        with patch("app.api.server.get_settings") as mock_get_settings:
            fake_settings = mock_get_settings.return_value
            fake_settings.ENABLE_DRIFT = False
            fake_settings.ENABLE_MCP = False
            fake_settings.ENABLE_RBAC = False

            with TestClient(app) as client:
                response = client.post("/eval/drift", json={})

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data.get("enabled") is False, (
            f"Expected enabled=False, got {data}"
        )


# =========================================================================
# Test 5 – Endpoint accepts optional body
# =========================================================================


class TestEndpointOptionalBody:
    """5. test_endpoint_optional_body

    POST /eval/drift with NO body at all -> HTTP 200 (not 422),
    confirming the optional-body contract.
    """

    def test_endpoint_optional_body(self) -> None:
        # Patch get_settings on the server module (the reference the
        # endpoint actually calls at runtime).
        from app.api.server import app

        with patch("app.api.server.get_settings") as mock_get_settings:
            fake_settings = mock_get_settings.return_value
            fake_settings.ENABLE_DRIFT = False
            fake_settings.ENABLE_MCP = False
            fake_settings.ENABLE_RBAC = False

            with TestClient(app) as client:
                # Send POST without any body / content-type.
                response = client.post("/eval/drift")

        assert response.status_code == 200, (
            f"Expected 200 (not 422) for missing body, "
            f"got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data.get("enabled") is False, (
            f"Expected enabled=False, got {data}"
        )
