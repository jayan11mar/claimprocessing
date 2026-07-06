"""Utilities for evaluating RAG retrieval quality against configurable acceptance thresholds."""

import json
import re
from typing import Any, Callable, Dict, List, Optional

from app.config import get_settings
from app.rag.chunkers import Chunk
from app.rag.retriever_hybrid import hybrid_retrieve


DEFAULT_CONTEXT_THRESHOLDS = {
    "loan underwriting": {
        "hit_rate_at_5": 0.85,
        "mrr": 0.65,
        "faithfulness": 0.9,
        "answer_correctness": 0.8,
        "llm_judge_avg": 4.0,
        "citation_coverage": 1.0,
    },
    "customer svc": {
        "hit_rate_at_5": 0.85,
        "mrr": 0.65,
        "faithfulness": 0.85,
        "answer_correctness": 0.8,
        "llm_judge_avg": 4.0,
        "citation_coverage": 1.0,
    },
    "aml / fraud": {
        "hit_rate_at_5": 0.8,
        "mrr": 0.6,
        "faithfulness": 0.95,
        "answer_correctness": 0.85,
        "llm_judge_avg": 4.2,
        "citation_coverage": 1.0,
    },
    "claims / insurance": {
        "hit_rate_at_5": 0.85,
        "mrr": 0.65,
        "faithfulness": 0.9,
        "answer_correctness": 0.8,
        "llm_judge_avg": 4.0,
        "citation_coverage": 1.0,
    },
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _keyword_overlap_score(text: str, keywords: Optional[List[str]] = None) -> float:
    if not text:
        return 0.0
    if not keywords:
        return 0.0
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return 0.0
    matches = 0
    for keyword in keywords:
        if not keyword:
            continue
        if _normalize_text(keyword) in normalized_text:
            matches += 1
    return round(matches / max(1, len(keywords)), 3)


def _build_default_cases() -> List[Dict[str, Any]]:
    return [
        {
            "name": "coverage lookup",
            "query": "Is pre-hospitalization covered under this policy?",
            "expected_keywords": ["pre-hospitalization", "coverage"],
            "top_k": 3,
        },
        {
            "name": "claim document request",
            "query": "What documents are required for a hospital claim?",
            "expected_keywords": ["documents", "hospital", "claim"],
            "top_k": 3,
        },
    ]


def _get_thresholds(settings: Optional[Any] = None) -> Dict[str, float]:
    if settings is None:
        settings = get_settings()

    context = str(getattr(settings, "RAG_EVALUATION_CONTEXT", "claims / insurance") or "claims / insurance").strip().lower()
    profile = DEFAULT_CONTEXT_THRESHOLDS.get(context, DEFAULT_CONTEXT_THRESHOLDS["claims / insurance"])

    return {
        "hit_rate_at_5": float(getattr(settings, "RAG_EVALUATION_HIT_RATE_THRESHOLD", profile["hit_rate_at_5"])),
        "mrr": float(getattr(settings, "RAG_EVALUATION_MRR_THRESHOLD", profile["mrr"])),
        "faithfulness": float(getattr(settings, "RAG_EVALUATION_FAITHFULNESS_THRESHOLD", profile["faithfulness"])),
        "answer_correctness": float(getattr(settings, "RAG_EVALUATION_ANSWER_CORRECTNESS_THRESHOLD", profile["answer_correctness"])),
        "llm_judge_avg": float(getattr(settings, "RAG_EVALUATION_LLM_JUDGE_AVG_THRESHOLD", profile["llm_judge_avg"])),
        "citation_coverage": float(getattr(settings, "RAG_EVALUATION_CITATION_COVERAGE_THRESHOLD", profile["citation_coverage"])),
    }


def evaluate_rag_queries(
    cases: Optional[List[Dict[str, Any]]] = None,
    chunks: Optional[List[Chunk]] = None,
    retrieval_fn: Optional[Callable[[List[Chunk], str, int], List[Dict[str, Any]]]] = None,
    answer_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate a set of RAG queries and return a structured pass/fail report."""
    if cases is None:
        cases = _build_default_cases()

    settings = get_settings()
    thresholds = _get_thresholds(settings)
    retrieval_threshold = thresholds["hit_rate_at_5"]
    answer_threshold = thresholds["answer_correctness"]
    overall_threshold = max(thresholds["hit_rate_at_5"], thresholds["answer_correctness"], thresholds["faithfulness"])
    min_citations = int(getattr(settings, "RAG_EVALUATION_MIN_CITATIONS", 1))

    if retrieval_fn is None:
        retrieval_fn = lambda chunks_to_search, query, k=3: hybrid_retrieve(chunks_to_search, query, k=k)
    if answer_fn is None:
        def answer_fn(query: str, chunks: Optional[List[Chunk]] = None, claim_context: Optional[str] = None, top_k: int = 3, embedding_fn: Optional[Any] = None) -> Dict[str, Any]:
            from app.rag.qa_chain import run_qa_chain

            return run_qa_chain(query, chunks=chunks, claim_context=claim_context, top_k=top_k, embedding_fn=embedding_fn)

    results: List[Dict[str, Any]] = []
    passed_cases = 0
    for case in cases:
        query = case.get("query", "")
        expected_keywords = case.get("expected_keywords", [])
        top_k = int(case.get("top_k", 3))

        retrieval_results = []
        if chunks is not None:
            retrieval_results = retrieval_fn(chunks, query, k=max(5, top_k))

        chunk_texts = []
        if retrieval_results:
            for result in retrieval_results[:5]:
                chunk = result.get("chunk")
                chunk_texts.append(getattr(chunk, "text", "") if chunk is not None else "")

        best_chunk_text = chunk_texts[0] if chunk_texts else ""
        retrieval_score = _keyword_overlap_score(best_chunk_text, expected_keywords)
        hit_rate_at_5 = 1.0 if any(_keyword_overlap_score(chunk_text, expected_keywords) > 0 for chunk_text in chunk_texts[:5]) else 0.0
        mrr = 0.0
        for index, chunk_text in enumerate(chunk_texts[:5], start=1):
            if _keyword_overlap_score(chunk_text, expected_keywords) > 0:
                mrr = 1.0 / index
                break

        answer_payload = answer_fn(
            query,
            chunks=chunks,
            claim_context=case.get("claim_context"),
            top_k=top_k,
            embedding_fn=case.get("embedding_fn"),
        )
        answer_text = answer_payload.get("answer_text", "") if isinstance(answer_payload, dict) else ""
        citations = answer_payload.get("citations", []) if isinstance(answer_payload, dict) else []
        answer_score = _keyword_overlap_score(answer_text, expected_keywords)
        citation_count = len(citations or [])
        citation_coverage = min(1.0, citation_count / max(1, min_citations))
        faithfulness = 0.0
        if citations and answer_text:
            shared_tokens = set(_normalize_text(answer_text).split()) & set(_normalize_text(best_chunk_text).split())
            faithfulness = min(1.0, round(len(shared_tokens) / max(1, len(expected_keywords)), 3))
        if not faithfulness and answer_score > 0:
            faithfulness = min(1.0, answer_score)
        llm_judge_avg = round(((answer_score * 5) + (faithfulness * 5) + (citation_coverage * 5)) / 3, 3)
        overall_score = round((hit_rate_at_5 + mrr + faithfulness + answer_score + (llm_judge_avg / 5.0) + citation_coverage) / 6, 3)

        passed = (
            hit_rate_at_5 >= thresholds["hit_rate_at_5"]
            and mrr >= thresholds["mrr"]
            and faithfulness >= thresholds["faithfulness"]
            and answer_score >= thresholds["answer_correctness"]
            and llm_judge_avg >= thresholds["llm_judge_avg"]
            and citation_coverage >= thresholds["citation_coverage"]
        )
        if passed:
            passed_cases += 1

        results.append(
            {
                "name": case.get("name", query),
                "query": query,
                "expected_keywords": expected_keywords,
                "retrieval_score": retrieval_score,
                "hit_rate_at_5": hit_rate_at_5,
                "mrr": mrr,
                "faithfulness": faithfulness,
                "answer_score": answer_score,
                "llm_judge_avg": llm_judge_avg,
                "citation_count": citation_count,
                "citation_coverage": citation_coverage,
                "overall_score": overall_score,
                "passed": passed,
                "thresholds": {
                    "retrieval": retrieval_threshold,
                    "answer": answer_threshold,
                    "overall": overall_threshold,
                    "hit_rate_at_5": thresholds["hit_rate_at_5"],
                    "mrr": thresholds["mrr"],
                    "faithfulness": thresholds["faithfulness"],
                    "answer_correctness": thresholds["answer_correctness"],
                    "llm_judge_avg": thresholds["llm_judge_avg"],
                    "citation_coverage": thresholds["citation_coverage"],
                    "min_citations": min_citations,
                },
                "answer_text": answer_text,
            }
        )

    return {
        "summary": {
            "total_cases": len(results),
            "passed_cases": passed_cases,
            "failed_cases": len(results) - passed_cases,
            "overall_passed": passed_cases == len(results),
            "thresholds": {
                "retrieval": retrieval_threshold,
                "answer": answer_threshold,
                "overall": overall_threshold,
                "hit_rate_at_5": thresholds["hit_rate_at_5"],
                "mrr": thresholds["mrr"],
                "faithfulness": thresholds["faithfulness"],
                "answer_correctness": thresholds["answer_correctness"],
                "llm_judge_avg": thresholds["llm_judge_avg"],
                "citation_coverage": thresholds["citation_coverage"],
                "min_citations": min_citations,
            },
        },
        "cases": results,
    }


def run_rag_evaluation(
    cases: Optional[List[Dict[str, Any]]] = None,
    chunks: Optional[List[Chunk]] = None,
    output_path: Optional[str] = None,
    retrieval_fn: Optional[Callable[[List[Chunk], str, int], List[Dict[str, Any]]]] = None,
    answer_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run an evaluation and optionally persist the JSON report to disk."""
    report = evaluate_rag_queries(cases=cases, chunks=chunks, retrieval_fn=retrieval_fn, answer_fn=answer_fn)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
    return report
