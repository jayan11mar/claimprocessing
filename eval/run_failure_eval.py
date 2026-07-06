"""Run evaluation on failure test cases to generate failure analysis document."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval.extrinsic import compute_extrinsic_metrics
from eval.failure_analysis import bucket_failures
from eval.intrinsic import compute_intrinsic_metrics
from eval.llm_judge import judge_answer


def _load_failure_dataset(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the failure test cases dataset."""
    if path is None:
        path = str(Path(__file__).resolve().parent.parent / "data" / "golden_dataset" / "rag_failure_cases.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_failure_evaluation(output_dir: Optional[str] = None, dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate the failure test cases and generate failure analysis report."""
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent.parent / "reports")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = _load_failure_dataset(dataset_path)
    items = dataset.get("items", [])
    thresholds = dataset.get("threshold_metrics", {})

    results: List[Dict[str, Any]] = []
    for item in items:
        # Use retrieved_chunks from the test case (simulating actual retrieval results)
        retrieved_chunks = item.get("retrieved_chunks", [])
        expected_chunks = item.get("expected_chunks", [])
        
        # Use answer_override if present (simulating actual generated answer), otherwise use expected_answer
        actual_answer = item.get("answer_override", item.get("expected_answer", ""))
        expected_answer = item.get("expected_answer", "")
        
        # Compute intrinsic metrics
        intrinsic = compute_intrinsic_metrics(
            retrieved_chunks=retrieved_chunks,
            expected_chunks=expected_chunks,
            k=3
        )
        
        # Compute extrinsic metrics
        extrinsic = compute_extrinsic_metrics(
            answer=actual_answer,
            expected_answer=expected_answer,
            retrieved_chunks=retrieved_chunks,
        )
        
        # LLM judge evaluation
        judge = judge_answer(
            query=item.get("query", ""),
            answer=actual_answer,
            expected_answer=expected_answer,
            retrieved_chunks=retrieved_chunks,
        )

        # Check if test case passes thresholds
        passed = (
            intrinsic["hit_at_k"] >= float(thresholds.get("hit_rate_at_5", 0.8))
            and intrinsic["mrr"] >= float(thresholds.get("mrr", 0.65))
            and extrinsic["faithfulness"] >= float(thresholds.get("faithfulness", 0.9))
            and extrinsic["answer_correctness"] >= float(thresholds.get("answer_correctness", 0.8))
            and judge["overall_score"] >= float(thresholds.get("llm_judge_avg", 4.0)) / 5.0
        )

        # Use the predefined failure reason if available, otherwise generate one
        if not passed:
            predefined_reason = item.get("failure_reason", "")
            if predefined_reason:
                reason = predefined_reason
            else:
                # Generate reason based on which metrics failed
                reasons = []
                if intrinsic["hit_at_k"] < float(thresholds.get("hit_rate_at_5", 0.8)):
                    reasons.append("missing evidence")
                if extrinsic["faithfulness"] < float(thresholds.get("faithfulness", 0.9)):
                    reasons.append("answer not grounded in context")
                if extrinsic["answer_correctness"] < float(thresholds.get("answer_correctness", 0.8)):
                    reasons.append("incorrect answer")
                if judge["overall_score"] < float(thresholds.get("llm_judge_avg", 4.0)) / 5.0:
                    reasons.append("poor answer quality")
                reason = ", ".join(reasons) if reasons else "missing evidence"
        else:
            reason = ""

        results.append({
            "project": dataset.get("project", "failure_analysis_test_cases"),
            "id": item.get("id", "unknown"),
            "query": item.get("query", ""),
            "difficulty": item.get("difficulty", "unknown"),
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
            "judge": judge,
            "passed": passed,
            "reason": reason,
            "source": item.get("source", ""),
        })

    failure_buckets = bucket_failures(results)
    summary = {
        "projects_evaluated": 1,
        "cases_evaluated": len(results),
        "passed_cases": sum(1 for result in results if result["passed"]),
        "failed_cases": sum(1 for result in results if not result["passed"]),
        "failure_buckets": failure_buckets,
    }

    # Write summary.json
    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Write report.md
    report_path = output_path / "report.md"
    report_content = f"""# RAG Evaluation Report - Failure Analysis

## Summary

- Projects evaluated: {summary['projects_evaluated']}
- Cases evaluated: {summary['cases_evaluated']}
- Passed cases: {summary['passed_cases']}
- Failed cases: {summary['failed_cases']}

## Failure Buckets

- Retrieval: {len(failure_buckets['retrieval'])} cases
- Answer quality: {len(failure_buckets['answer_quality'])} cases
- Citations: {len(failure_buckets['citations'])} cases
- Other: {len(failure_buckets['other'])} cases

## Detailed Failure Analysis

### Top Failed Queries

"""
    # Add top 10 failed queries (or all if less than 10)
    failed_results = [r for r in results if not r["passed"]]
    for i, result in enumerate(failed_results[:10], 1):
        report_content += f"""#### {i}. {result['id']}: {result['query'][:100]}{'...' if len(result['query']) > 100 else ''}

**Failure Reason:** {result['reason']}

**Metrics:**
- Hit@K: {result['intrinsic']['hit_at_k']} (threshold: {thresholds.get('hit_rate_at_5', 0.8)})
- MRR: {result['intrinsic']['mrr']} (threshold: {thresholds.get('mrr', 0.65)})
- Faithfulness: {result['extrinsic']['faithfulness']} (threshold: {thresholds.get('faithfulness', 0.9)})
- Answer Correctness: {result['extrinsic']['answer_correctness']} (threshold: {thresholds.get('answer_correctness', 0.8)})
- LLM Judge Score: {result['judge']['overall_score']} (threshold: {float(thresholds.get('llm_judge_avg', 4.0)) / 5.0})

**Source:** {result.get('source', 'N/A')}

---

"""
    
    report_path.write_text(report_content, encoding="utf-8")

    # Write failure_analysis.json
    failure_path = output_path / "failure_analysis.json"
    failure_analysis = {
        "summary": summary,
        "failed_queries": [
            {
                "id": r["id"],
                "query": r["query"],
                "reason": r["reason"],
                "metrics": {
                    "hit_at_k": r["intrinsic"]["hit_at_k"],
                    "mrr": r["intrinsic"]["mrr"],
                    "faithfulness": r["extrinsic"]["faithfulness"],
                    "answer_correctness": r["extrinsic"]["answer_correctness"],
                    "llm_judge_score": r["judge"]["overall_score"],
                },
                "difficulty": r["difficulty"],
                "source": r.get("source", ""),
            }
            for r in failed_results[:10]
        ],
        "failure_buckets": failure_buckets,
        "root_causes": {
            "retrieval": "RAG system failed to retrieve relevant context chunks for the query",
            "answer_quality": "Generated answer was incorrect, incomplete, or hallucinated",
            "citations": "Answer was not properly grounded in the retrieved context",
            "other": "Other failure modes not categorized above",
        },
        "proposed_fixes": {
            "retrieval": [
                "Improve embedding model for better semantic matching",
                "Add query expansion techniques",
                "Implement hybrid search (BM25 + vector)",
                "Tune chunk size and overlap parameters",
                "Add metadata filtering for domain-specific retrieval",
            ],
            "answer_quality": [
                "Fine-tune LLM on domain-specific Q&A pairs",
                "Add answer validation against retrieved context",
                "Implement chain-of-thought prompting",
                "Add post-processing to verify factual accuracy",
                "Use larger or more capable LLM for generation",
            ],
            "citations": [
                "Implement stricter grounding checks",
                "Add citation verification step",
                "Use extractive QA for factual claims",
                "Implement fact-checking against retrieved context",
                "Add penalties for unsupported claims in training",
            ],
            "other": [
                "Review edge cases and add specific handling",
                "Improve error handling and fallback mechanisms",
                "Add human-in-the-loop for ambiguous queries",
            ],
        },
    }
    failure_path.write_text(json.dumps(failure_analysis, indent=2), encoding="utf-8")

    return {"summary": summary, "results": results}


def main() -> None:
    output_dir = os.getenv("RAG_EVAL_OUTPUT_DIR", str(Path(__file__).resolve().parent.parent / "reports"))
    report = run_failure_evaluation(output_dir=output_dir)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()