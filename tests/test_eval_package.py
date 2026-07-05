import json
from pathlib import Path

from eval.extrinsic import compute_extrinsic_metrics
from eval.failure_analysis import bucket_failures
from eval.intrinsic import compute_intrinsic_metrics
from eval.llm_judge import judge_answer


def test_intrinsic_metrics_compute_expected_values():
    metrics = compute_intrinsic_metrics(
        retrieved_chunks=["policy coverage", "claim documents", "hospital claim"],
        expected_chunks=["policy coverage", "claim documents"],
        k=3,
    )

    assert metrics["hit_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg"] > 0.8
    assert metrics["context_precision"] > 0.0
    assert metrics["context_recall"] == 1.0


def test_extrinsic_metrics_compute_expected_values():
    metrics = compute_extrinsic_metrics(
        answer="Coverage applies to hospital claims and requires supporting documents.",
        expected_answer="Coverage applies to hospital claims and requires supporting documents.",
        retrieved_chunks=["Coverage applies to hospital claims.", "Supporting documents are required."],
    )

    assert metrics["faithfulness"] >= 0.7
    assert metrics["answer_correctness"] >= 0.8
    assert metrics["answer_relevance"] >= 0.8


def test_judge_answer_and_failure_bucketing_work():
    judgment = judge_answer(
        query="What documents are required?",
        answer="You need the itemized bill and discharge summary.",
        expected_answer="You need the itemized bill and discharge summary.",
        retrieved_chunks=["Itemized bill", "Discharge summary"],
    )

    assert judgment["overall_score"] >= 0.0
    assert judgment["criteria"]["correctness"] >= 0.0

    buckets = bucket_failures(
        [
            {"id": "case-1", "passed": False, "reason": "missing evidence"},
            {"id": "case-2", "passed": False, "reason": "poor answer relevance"},
            {"id": "case-3", "passed": True, "reason": ""},
        ]
    )

    assert "missing evidence" in buckets["retrieval"]
    assert "poor answer relevance" in buckets["answer_quality"]


def test_run_eval_creates_reports(tmp_path):
    output_dir = tmp_path / "eval_output"
    output_dir.mkdir()

    from eval.run_eval import run_evaluation

    report = run_evaluation(output_dir=str(output_dir), project_name="claims / insurance")

    assert report["summary"]["projects_evaluated"] == 1
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "failure_analysis.json").exists()
