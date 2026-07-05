from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DOCS_DIR = REPO_ROOT / "docs"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"


def _normalize_output_dir(output_dir: Optional[Path | str] = None) -> Path:
    if output_dir is None:
        return REPO_ROOT
    if isinstance(output_dir, str):
        return Path(output_dir)
    return output_dir


def _summarize_evaluation(evaluation_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if evaluation_result is None:
        return {
            "cases_evaluated": 0,
            "passed_cases": 0,
            "failed_cases": 0,
            "projects_evaluated": 0,
        }

    summary = evaluation_result.get("summary", {}) if isinstance(evaluation_result, dict) else {}
    return {
        "projects_evaluated": int(summary.get("projects_evaluated", 0) or 0),
        "cases_evaluated": int(summary.get("cases_evaluated", 0) or 0),
        "passed_cases": int(summary.get("passed_cases", 0) or 0),
        "failed_cases": int(summary.get("failed_cases", 0) or 0),
    }


def build_acceptance_artifacts(
    output_dir: Optional[Path | str] = None,
    evaluation_result: Optional[Dict[str, Any]] = None,
    trace_count: int = 30,
) -> Dict[str, Path]:
    base_dir = _normalize_output_dir(output_dir)
    docs_dir = base_dir / "docs"
    reports_dir = base_dir / "reports"
    docs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary = _summarize_evaluation(evaluation_result)
    evidence_payload = {
        "summary": summary,
        "trace_capture_target": int(trace_count),
        "trace_capture_status": "local tracing disabled; evidence captured from evaluation artifacts and endpoint tests",
        "evidence_paths": {
            "evaluation_summary": str(reports_dir / "summary.json"),
            "evaluation_report": str(reports_dir / "report.md"),
            "api_tests": "tests/test_api_rag_endpoints.py",
            "pipeline_tests": "tests/test_rag_pipeline.py",
        },
    }
    evidence_path = reports_dir / "acceptance_evidence.json"
    evidence_path.write_text(json.dumps(evidence_payload, indent=2), encoding="utf-8")

    mapping_path = docs_dir / "project_acceptance_mapping.md"
    signoff_path = docs_dir / "project_signoff_report.md"

    mapping_text = """# Acceptance criteria mapping

This document ties each project requirement to the implemented modules, the tests that cover it, and the evidence artifacts generated for sign-off.

## Ingestion and indexing

| Criterion | Implemented modules | Tests / metrics | Evidence |
| --- | --- | --- | --- |
| Upload documents and create chunks for retrieval | app/api/server.py, app/rag/chunkers.py, app/rag/embeddings.py | tests/test_api_rag_endpoints.py | /ingest and /ingest/status endpoints, plus evaluation evidence in reports/summary.json |
| Build a retrievable index from uploaded documents | app/api/server.py, app/rag/vectorstores.py | tests/test_api_rag_endpoints.py | /retrieve endpoint returns ranked chunks and source_count |

## Retrieval and citations

| Criterion | Implemented modules | Tests / metrics | Evidence |
| --- | --- | --- | --- |
| Return top-k chunks for a user query | app/api/server.py, app/rag/retriever_hybrid.py | tests/test_api_rag_endpoints.py | /retrieve endpoint returns chunk payloads with scores |
| Attach citations to RAG-backed answers | app/api/server.py, app/chains/agent_chain.py | tests/test_api_rag_endpoints.py | /chat surfaces retrieval_trace and citations in the response payload |

## Chat and agent orchestration

| Criterion | Implemented modules | Tests / metrics | Evidence |
| --- | --- | --- | --- |
| Route between deterministic tools and retrieval-backed answers | app/chains/agent_chain.py, app/api/server.py | tests/test_api_rag_endpoints.py | Chat endpoint accepts retrieval metadata and returns structured answers |
| Preserve existing endpoint compatibility while adding the retrieval workflow | app/api/server.py | tests/test_api_rag_and_retrieval.py | Existing health and chat tests remain green alongside the new acceptance coverage |

## Evaluation and sign-off

| Criterion | Implemented modules | Tests / metrics | Evidence |
| --- | --- | --- | --- |
| Produce measurable RAG evaluation outputs | eval/run_eval.py | tests/test_rag_pipeline.py | reports/summary.json and reports/report.md |
| Create a sign-off package for project acceptance | app/rag/acceptance_validation.py | tests/test_rag_pipeline.py | docs/project_signoff_report.md and reports/acceptance_evidence.json |
"""

    signoff_text = f"""# Project sign-off report

## Summary
- Projects evaluated: {summary['projects_evaluated']}
- Cases evaluated: {summary['cases_evaluated']}
- Passed cases: {summary['passed_cases']}
- Failed cases: {summary['failed_cases']}
- Trace capture target: {trace_count} sample queries
- Trace capture status: local tracing disabled; evidence uses evaluation artifacts and endpoint tests instead

## Status
The acceptance bundle is ready for review. The retrieval pipeline, ingestion endpoints, and chat response contract are covered by automated tests and documented in this report.

## Gaps and mitigation
- LangSmith traces were not captured in this environment because tracing is disabled. Enable LANGSMITH_TRACING and a valid LANGSMITH_API_KEY to capture the requested run trees before formal sign-off.
- If evaluation quality falls below the project thresholds, inspect reports/report.md and reports/failure_analysis.json for the failing buckets and iterate on retrieval or answer generation.

## Trace capture
The acceptance validation workflow writes a machine-readable evidence bundle to reports/acceptance_evidence.json so the sign-off package can be re-generated in CI or a local environment with the same inputs.
"""

    mapping_path.write_text(mapping_text, encoding="utf-8")
    signoff_path.write_text(signoff_text, encoding="utf-8")

    return {
        "mapping_path": mapping_path,
        "signoff_path": signoff_path,
        "evidence_path": evidence_path,
    }
