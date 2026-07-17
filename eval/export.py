"""JSON/CSV export for CI pipeline integration.

Provides:
  - JSON export of evaluation results (machine-readable)
  - CSV export for spreadsheet analysis
  - JUnit XML export for CI test reporting
  - Markdown summary for PR comments
  - CI-friendly exit codes (0 = all pass, 1 = failures)
"""

import csv
import io
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_json(
    report: Dict[str, Any],
    output_path: Optional[str] = None,
    pretty: bool = True,
) -> str:
    """Export evaluation report as JSON.

    Args:
        report: The evaluation report dict.
        output_path: Optional file path to write to.
        pretty: If True, indent the JSON.

    Returns:
        The JSON string.
    """
    indent = 2 if pretty else None
    json_str = json.dumps(report, indent=indent, default=str)

    if output_path:
        Path(output_path).write_text(json_str, encoding="utf-8")

    return json_str


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> str:
    """Export per-case evaluation results as CSV.

    Args:
        results: List of evaluation result dicts.
        output_path: Optional file path to write to.

    Returns:
        The CSV string.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    writer.writerow([
        "id",
        "query",
        "difficulty",
        "project",
        "passed",
        "hit_at_k",
        "mrr",
        "ndcg",
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_correctness",
        "answer_relevance",
        "judge_overall",
        "judge_correctness",
        "judge_completeness",
        "judge_citation_quality",
        "judge_clarity",
    ])

    for r in results:
        intrinsic = r.get("intrinsic", {})
        extrinsic = r.get("extrinsic", {})
        judge = r.get("judge", {})
        judge_criteria = judge.get("criteria", {})

        writer.writerow([
            r.get("id", ""),
            r.get("query", ""),
            r.get("difficulty", ""),
            r.get("project", ""),
            "PASS" if r.get("passed") else "FAIL",
            intrinsic.get("hit_at_k", ""),
            intrinsic.get("mrr", ""),
            intrinsic.get("ndcg", ""),
            intrinsic.get("context_precision", ""),
            intrinsic.get("context_recall", ""),
            extrinsic.get("faithfulness", ""),
            extrinsic.get("answer_correctness", ""),
            extrinsic.get("answer_relevance", ""),
            judge.get("overall_score", ""),
            judge_criteria.get("correctness", ""),
            judge_criteria.get("completeness", ""),
            judge_criteria.get("citation_quality", ""),
            judge_criteria.get("clarity", ""),
        ])

    csv_str = output.getvalue()
    output.close()

    if output_path:
        Path(output_path).write_text(csv_str, encoding="utf-8")

    return csv_str


# ---------------------------------------------------------------------------
# JUnit XML export (for CI integration)
# ---------------------------------------------------------------------------

def export_junit_xml(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    suite_name: str = "rag-evaluation",
) -> str:
    """Export evaluation results as JUnit XML for CI pipeline integration.

    Each test case in the golden set becomes a JUnit test case.
    A case passes if all metric thresholds are met.

    Args:
        results: List of evaluation result dicts.
        output_path: Optional file path to write to.
        suite_name: Name for the test suite.

    Returns:
        The XML string.
    """
    # Create the root element
    testsuite = ET.Element("testsuite")
    testsuite.set("name", suite_name)
    testsuite.set("timestamp", datetime.now(timezone.utc).isoformat())

    passed = sum(1 for r in results if r.get("passed"))
    failed = len(results) - passed
    testsuite.set("tests", str(len(results)))
    testsuite.set("failures", str(failed))
    testsuite.set("errors", "0")

    for r in results:
        case_id = r.get("id", "unknown")
        query = r.get("query", "")[:100]

        testcase = ET.SubElement(testsuite, "testcase")
        testcase.set("name", f"{case_id}: {query}")
        testcase.set("classname", r.get("project", "unknown"))

        if not r.get("passed"):
            failure = ET.SubElement(testcase, "failure")
            failure.set("type", "threshold_check")
            intrinsic = r.get("intrinsic", {})
            extrinsic = r.get("extrinsic", {})
            judge = r.get("judge", {})
            message = (
                f"hit_at_k={intrinsic.get('hit_at_k')}, "
                f"mrr={intrinsic.get('mrr')}, "
                f"faithfulness={extrinsic.get('faithfulness')}, "
                f"correctness={extrinsic.get('answer_correctness')}, "
                f"judge={judge.get('overall_score')}"
            )
            failure.set("message", message)
            failure.text = r.get("reason", "threshold check failed")

        # Add properties with detailed metrics
        properties = ET.SubElement(testcase, "properties")
        for key, value in _flatten_metrics(r).items():
            prop = ET.SubElement(properties, "property")
            prop.set("name", key)
            prop.set("value", str(value))

    xml_str = ET.tostring(testsuite, encoding="unicode", short_empty_elements=True)

    if output_path:
        Path(output_path).write_text(xml_str, encoding="utf-8")

    return xml_str


def _flatten_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested metric dicts into a single level."""
    flat: Dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    for sub_sub_key, sub_sub_value in sub_value.items():
                        flat[f"{key}_{sub_key}_{sub_sub_key}"] = sub_sub_value
                else:
                    flat[f"{key}_{sub_key}"] = sub_value
        else:
            flat[key] = value
    return flat


# ---------------------------------------------------------------------------
# Markdown summary (for PR comments)
# ---------------------------------------------------------------------------

def export_markdown_summary(
    summary: Dict[str, Any],
    comparison: Optional[Dict[str, Any]] = None,
    custom_metrics: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> str:
    """Export a human-readable Markdown summary of evaluation results.

    Suitable for posting as a PR comment or including in CI output.

    Args:
        summary: The evaluation summary dict.
        comparison: Optional baseline comparison dict.
        custom_metrics: Optional custom metrics results (Spec 3.5).
        output_path: Optional file path to write to.

    Returns:
        The Markdown string.
    """
    lines: List[str] = []
    lines.append("# RAG Evaluation Report")
    lines.append("")
    lines.append(f"- **Timestamp**: {summary.get('timestamp', 'unknown')}")
    lines.append(f"- **Elapsed**: {summary.get('elapsed_seconds', 0)}s")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Cases | {summary.get('total_cases', 0)} |")
    lines.append(f"| Passed | {summary.get('passed_cases', 0)} |")
    lines.append(f"| Failed | {summary.get('failed_cases', 0)} |")
    lines.append(f"| Pass Rate | {summary.get('pass_rate', 0):.2%} |")
    lines.append(f"| Projects | {summary.get('projects_evaluated', 0)} |")
    lines.append("")

    # Custom metrics (Spec 3.5)
    if custom_metrics:
        lines.append("## Custom Metrics (Spec 3.5)")
        lines.append("")
        lines.append("| Metric | Value | Required | Status |")
        lines.append("|--------|-------|----------|--------|")
        overall = custom_metrics.get("overall", {})
        required = custom_metrics.get("required_thresholds", {})
        for key in ["golden_set_pass_rate", "answer_stability", "regulatory_compliance",
                     "role_appropriateness", "hitl_trigger_precision"]:
            value = overall.get(key, 0)
            req = required.get(key, 0)
            status = "✅ PASS" if value >= req else "❌ FAIL"
            lines.append(f"| {key.replace('_', ' ').title()} | {value:.2%} | {req:.0%} | {status} |")
        lines.append("")
        lines.append(f"**Overall: {'✅ ALL PASSED' if custom_metrics.get('all_metrics_passed') else '❌ SOME FAILED'}**")
        lines.append("")

    # Failure buckets
    failure_buckets = summary.get("failure_buckets", {})
    if any(failure_buckets.values()):
        lines.append("## Failure Analysis")
        lines.append("")
        for bucket, cases in failure_buckets.items():
            if cases:
                lines.append(f"- **{bucket.replace('_', ' ').title()}**: {', '.join(cases[:10])}")
        lines.append("")

    # Baseline comparison
    if comparison and "error" not in comparison:
        lines.append("## Baseline Comparison")
        lines.append("")
        lines.append(f"- **Baseline**: {comparison.get('baseline_timestamp', 'unknown')}")
        lines.append(f"- **Pass Rate Delta**: {comparison.get('pass_rate_delta', 0):+.2%}")
        lines.append(f"- **Regressions**: {comparison.get('regression_count', 0)}")
        lines.append(f"- **Improvements**: {comparison.get('improvement_count', 0)}")
        lines.append("")

        if comparison.get("regressions"):
            lines.append("### Regressions")
            lines.append("")
            for reg in comparison["regressions"][:10]:
                lines.append(
                    f"- {reg['id']}: judge score "
                    f"{reg.get('baseline_judge', 'N/A')} → {reg.get('current_judge', 'N/A')}"
                )
            lines.append("")

    # All-important CI status
    all_passed = summary.get("all_metrics_passed", summary.get("failed_cases", 0) == 0)
    lines.append("---")
    lines.append("")
    lines.append(f"**CI Status: {'✅ PASSED' if all_passed else '❌ FAILED'}**")
    lines.append("")

    md_str = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(md_str, encoding="utf-8")

    return md_str


# ---------------------------------------------------------------------------
# CI-friendly runner (exits with non-zero on failure)
# ---------------------------------------------------------------------------

def ci_run(
    report: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> int:
    """Run all exports suitable for CI pipeline integration.

    Writes JSON, CSV, JUnit XML, and Markdown files to the output directory.
    Returns 0 if all tests pass, 1 if any fail.

    Args:
        report: The evaluation report dict.
        output_dir: Directory to write export files.

    Returns:
        0 for pass, 1 for failure.
    """
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    summary = report.get("summary", {})
    results = report.get("results", [])
    comparison = report.get("comparison")
    custom_metrics = report.get("custom_metrics")

    # JSON export
    json_path = str(output_path / "evaluation_report.json") if output_path else None
    export_json(report, output_path=json_path)

    # CSV export
    csv_path = str(output_path / "evaluation_results.csv") if output_path else None
    export_csv(results, output_path=csv_path)

    # JUnit XML export
    junit_path = str(output_path / "evaluation_results.xml") if output_path else None
    export_junit_xml(results, output_path=junit_path)

    # Markdown summary
    md_path = str(output_path / "evaluation_summary.md") if output_path else None
    export_markdown_summary(
        summary,
        comparison=comparison,
        custom_metrics=custom_metrics,
        output_path=md_path,
    )

    # Determine exit code
    all_passed = summary.get("all_metrics_passed", summary.get("failed_cases", 0) == 0)
    return 0 if all_passed else 1


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for exporting evaluation results."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export evaluation results in various formats."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to evaluation report JSON file.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Output directory for export files.",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "csv", "junit", "md", "all"],
        default="all",
        help="Export format (default: all).",
    )

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        report = json.load(f)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = report.get("results", [])
    summary = report.get("summary", {})
    comparison = report.get("comparison")
    custom_metrics = report.get("custom_metrics")

    if args.format in ("json", "all"):
        export_json(report, output_path=str(output_path / "evaluation_report.json"))
        print(f"  JSON: {output_path / 'evaluation_report.json'}")

    if args.format in ("csv", "all"):
        export_csv(results, output_path=str(output_path / "evaluation_results.csv"))
        print(f"  CSV: {output_path / 'evaluation_results.csv'}")

    if args.format in ("junit", "all"):
        export_junit_xml(results, output_path=str(output_path / "evaluation_results.xml"))
        print(f"  JUnit XML: {output_path / 'evaluation_results.xml'}")

    if args.format in ("md", "all"):
        export_markdown_summary(
            summary,
            comparison=comparison,
            custom_metrics=custom_metrics,
            output_path=str(output_path / "evaluation_summary.md"),
        )
        print(f"  Markdown: {output_path / 'evaluation_summary.md'}")


if __name__ == "__main__":
    main()