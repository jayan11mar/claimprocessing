from pathlib import Path

from app.rag.acceptance_validation import build_acceptance_artifacts
from eval.run_eval import run_evaluation


def test_build_week6_acceptance_artifacts_generates_mapping_and_signoff(tmp_path):
    evaluation_result = run_evaluation(output_dir=str(tmp_path / "reports"))

    artifacts = build_acceptance_artifacts(
        output_dir=tmp_path,
        evaluation_result=evaluation_result,
        trace_count=30,
    )

    mapping_path = tmp_path / "docs" / "project_acceptance_mapping.md"
    signoff_path = tmp_path / "docs" / "project_signoff_report.md"

    assert artifacts["mapping_path"] == mapping_path
    assert artifacts["signoff_path"] == signoff_path
    assert mapping_path.exists()
    assert signoff_path.exists()

    mapping_text = mapping_path.read_text(encoding="utf-8")
    assert "Acceptance criteria mapping" in mapping_text
    assert "Ingestion and indexing" in mapping_text
    assert "Retrieval and citations" in mapping_text
    assert "Chat and agent orchestration" in mapping_text

    signoff_text = signoff_path.read_text(encoding="utf-8")
    assert "Project sign-off report" in signoff_text
    assert "Summary" in signoff_text
    assert "Trace capture" in signoff_text
