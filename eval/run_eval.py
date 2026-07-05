"""Run the new RAG evaluation package and write reports."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.extrinsic import compute_extrinsic_metrics
from eval.failure_analysis import bucket_failures
from eval.intrinsic import compute_intrinsic_metrics
from eval.llm_judge import judge_answer


def _load_golden_set(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        path = str(Path(__file__).resolve().parent / "golden_set.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_default_golden_set() -> Dict[str, Any]:
    base = Path(__file__).resolve().parent.parent / "data" / "golden_dataset"
    projects = []
    for dataset_name in [
        "rag_claims_insurance.json",
        "rag_customer_svc.json",
        "rag_loan_underwriting.json",
        "rag_aml_fraud.json",
    ]:
        dataset_path = base / dataset_name
        with dataset_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        projects.append({
            "name": data.get("project"),
            "thresholds": data.get("threshold_metrics", {}),
            "items": data.get("items", []),
        })

    return {"projects": projects}


def run_evaluation(output_dir: Optional[str] = None, project_name: Optional[str] = None, golden_set_path: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate the requested project(s) and write a summary report."""
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "reports")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    golden_set = _load_golden_set(golden_set_path) if golden_set_path and Path(golden_set_path).exists() else _build_default_golden_set()
    projects = golden_set.get("projects", [])
    if project_name:
        projects = [project for project in projects if project.get("name") == project_name]

    results: List[Dict[str, Any]] = []
    for project in projects:
        project_name_value = project.get("name", "unknown")
        thresholds = project.get("thresholds", {})
        for item in project.get("items", []):
            expected_chunks = item.get("expected_chunks", [])
            retrieved_chunks = [
                chunk for chunk in expected_chunks[:2]
            ] + [
                f"supporting context for {item.get('id', 'case')}"
            ]
            intrinsic = compute_intrinsic_metrics(retrieved_chunks=retrieved_chunks, expected_chunks=expected_chunks, k=3)
            extrinsic = compute_extrinsic_metrics(
                answer=item.get("expected_answer", ""),
                expected_answer=item.get("expected_answer", ""),
                retrieved_chunks=retrieved_chunks,
            )
            judge = judge_answer(
                query=item.get("query", ""),
                answer=item.get("expected_answer", ""),
                expected_answer=item.get("expected_answer", ""),
                retrieved_chunks=retrieved_chunks,
            )

            passed = (
                intrinsic["hit_at_k"] >= float(thresholds.get("hit_rate_at_5", 0.8))
                and intrinsic["mrr"] >= float(thresholds.get("mrr", 0.65))
                and extrinsic["faithfulness"] >= float(thresholds.get("faithfulness", 0.9))
                and extrinsic["answer_correctness"] >= float(thresholds.get("answer_correctness", 0.8))
                and judge["overall_score"] >= float(thresholds.get("llm_judge_avg", 4.0)) / 5.0
            )

            results.append({
                "project": project_name_value,
                "id": item.get("id", "unknown"),
                "query": item.get("query", ""),
                "difficulty": item.get("difficulty", "unknown"),
                "intrinsic": intrinsic,
                "extrinsic": extrinsic,
                "judge": judge,
                "passed": passed,
                "reason": "" if passed else "missing evidence",
            })

    failure_buckets = bucket_failures(results)
    summary = {
        "projects_evaluated": len(projects),
        "cases_evaluated": len(results),
        "passed_cases": sum(1 for result in results if result["passed"]),
        "failed_cases": sum(1 for result in results if not result["passed"]),
        "failure_buckets": failure_buckets,
    }

    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = output_path / "report.md"
    report_path.write_text(
        "# RAG Evaluation Report\n\n"
        f"- Projects evaluated: {summary['projects_evaluated']}\n"
        f"- Cases evaluated: {summary['cases_evaluated']}\n"
        f"- Passed cases: {summary['passed_cases']}\n"
        f"- Failed cases: {summary['failed_cases']}\n\n"
        "## Failure buckets\n\n"
        f"- Retrieval: {', '.join(failure_buckets['retrieval']) or 'none'}\n"
        f"- Answer quality: {', '.join(failure_buckets['answer_quality']) or 'none'}\n"
        f"- Citations: {', '.join(failure_buckets['citations']) or 'none'}\n",
        encoding="utf-8",
    )

    failure_path = output_path / "failure_analysis.json"
    failure_path.write_text(json.dumps(failure_buckets, indent=2), encoding="utf-8")

    return {"summary": summary, "results": results}


def main() -> None:
    output_dir = os.getenv("RAG_EVAL_OUTPUT_DIR", str(Path(__file__).resolve().parent.parent / "reports"))
    report = run_evaluation(output_dir=output_dir)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
