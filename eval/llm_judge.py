"""LLM-as-judge wrapper for RAG answers with cross-family judging support.

Uses a different model family for judging than for generation to avoid
self-evaluation bias. When the generation model is OpenAI, the judge uses
Anthropic Claude, and vice versa.

Requires:
  - JUDGE_MODEL_NAME: The Anthropic Claude model to use (e.g., "claude-sonnet-4-20250514")
  - JUDGE_ANTHROPIC_API_KEY: API key for the judge Claude model

When JUDGE_ANTHROPIC_API_KEY is absent, falls back to a Python-based
semantic-similarity scorer using sentence-transformers cosine similarity.
NEVER falls back to the generation model (gpt-4o-mini).

Supports A/B randomization for pairwise comparisons to reduce position bias.
"""

import json
import logging
import os
import re
import random
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from eval.extrinsic import compute_extrinsic_metrics


logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _get_judge_config() -> Tuple[Optional[str], Optional[str]]:
    """Read judge configuration from environment variables.

    Returns:
        Tuple of (judge_model_name, anthropic_api_key).
        Both may be None if not configured.
    """
    judge_model = os.getenv("JUDGE_MODEL_NAME")
    api_key = os.getenv("JUDGE_ANTHROPIC_API_KEY")
    return judge_model, api_key


def _call_anthropic_judge(prompt: str, model: str, api_key: str) -> Optional[str]:
    """Call an Anthropic Claude model as judge with the provided API key."""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=512,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception:
        return None


def _build_judge_prompt(query: str, answer: str, expected_answer: str) -> str:
    """Build a structured prompt for the LLM judge."""
    return f"""You are an expert evaluator of RAG system outputs. Assess the quality of the generated answer against the expected answer.

## Query
{query}

## Generated Answer
{answer}

## Expected Answer
{expected_answer}

## Evaluation Criteria
Score each criterion on a scale of 1-5 (5 = best):

1. **correctness**: How factually correct is the answer compared to the expected answer?
2. **completeness**: Does the answer cover all key points from the expected answer?
3. **citation_quality**: Does the answer appear grounded in retrieved context?
4. **clarity**: Is the answer clear, well-structured, and easy to understand?

## Output Format
Return ONLY a valid JSON object with no additional text:
{{
  "correctness": <1-5>,
  "completeness": <1-5>,
  "citation_quality": <1-5>,
  "clarity": <1-5>,
  "reason": "<brief explanation of scores>"
}}"""


def _build_pairwise_judge_prompt(query: str, answer_a: str, answer_b: str, expected_answer: str) -> str:
    """Build a pairwise comparison prompt with randomized A/B labels."""
    return f"""You are an expert evaluator of RAG system outputs. Compare two answers against the expected answer.

## Query
{query}

## Answer A
{answer_a}

## Answer B
{answer_b}

## Expected Answer
{expected_answer}

## Evaluation Criteria
For each answer, score on a scale of 1-5 (5 = best):

1. **correctness**: How factually correct is the answer compared to the expected answer?
2. **completeness**: Does the answer cover all key points from the expected answer?
3. **citation_quality**: Does the answer appear grounded in retrieved context?
4. **clarity**: Is the answer clear, well-structured, and easy to understand?

## Output Format
Return ONLY a valid JSON object with no additional text:
{{
  "answer_a": {{
    "correctness": <1-5>,
    "completeness": <1-5>,
    "citation_quality": <1-5>,
    "clarity": <1-5>
  }},
  "answer_b": {{
    "correctness": <1-5>,
    "completeness": <1-5>,
    "citation_quality": <1-5>,
    "clarity": <1-5>
  }},
  "reason": "<brief explanation of scores>"
}}"""


def _parse_judge_response(response_text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse the JSON response from the LLM judge."""
    if not response_text:
        return None
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def _parse_pairwise_judge_response(response_text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse the JSON response from a pairwise LLM judge."""
    if not response_text:
        return None
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def _get_semantic_similarity_scorer():
    """Get or create a SemanticSimilarityScorer instance.

    Uses sentence-transformers for cosine similarity.
    Returns None if sentence-transformers is not available.
    """
    try:
        from eval.custom_metrics import SemanticSimilarityScorer
        return SemanticSimilarityScorer()
    except Exception:
        return None


def _compute_fallback_scores(
    query: str,
    answer: str,
    expected_answer: str,
    retrieved_chunks: Optional[Sequence[str]] = None,
    judge_model: str = "fallback",
    generation_model: str = "unknown",
) -> dict:
    """Compute fallback evaluation scores using semantic similarity.

    Uses sentence-transformers cosine similarity when available, falling
    back to token-overlap-based heuristic scoring only as last resort.

    NEVER uses the generation model for judging.
    """
    scorer = _get_semantic_similarity_scorer()

    if scorer is not None:
        sim = scorer.similarity(answer, expected_answer)
        correctness = round(max(0.0, min(1.0, sim)), 4)
        completeness = round(max(0.0, min(1.0, sim)), 4)

        if retrieved_chunks:
            chunk_sims = [
                scorer.similarity(answer, chunk) for chunk in retrieved_chunks if chunk
            ]
            citation_quality = round(float(np.mean(chunk_sims)), 4) if chunk_sims else 0.0
        else:
            citation_quality = 0.0

        clarity = min(1.0, len(_normalize(answer).split()) / 20.0) if answer else 0.0
        clarity = round(max(0.0, min(1.0, clarity)), 4)
        reason = "semantic-similarity fallback (sentence-transformers cosine)"
    else:
        metrics = compute_extrinsic_metrics(
            answer=answer,
            expected_answer=expected_answer,
            retrieved_chunks=retrieved_chunks,
        )
        correctness = metrics["answer_correctness"]
        completeness = min(1.0, correctness + 0.1)
        citation_quality = 1.0 if retrieved_chunks else 0.0
        clarity = 5.0 if answer and len(_normalize(answer).split()) >= 4 else 3.0
        reason = "token-overlap fallback (sentence-transformers unavailable)"

    criteria = {
        "correctness": round(5 * correctness, 3),
        "completeness": round(5 * completeness, 3),
        "citation_quality": round(5 * citation_quality, 3),
        "clarity": round(5 * clarity, 3),
    }

    overall_score = round(sum(criteria.values()) / len(criteria), 3)
    return {
        "overall_score": overall_score,
        "criteria": criteria,
        "judge_model": judge_model,
        "generation_model": generation_model,
        "reason": reason,
        "is_fallback": True,
    }


def judge_answer(
    query: str,
    answer: str,
    expected_answer: str,
    retrieved_chunks: Optional[Sequence[str]] = None,
) -> dict:
    """Score an answer using an independent Claude judge when available.

    Uses JUDGE_MODEL_NAME + JUDGE_ANTHROPIC_API_KEY for the judge.
    When the judge API key is absent, falls back to a Python semantic-similarity
    scorer using sentence-transformers cosine similarity.

    NEVER falls back to the generation model (gpt-4o-mini).

    Args:
        query: The original query string.
        answer: The generated answer to evaluate.
        expected_answer: The expected/reference answer.
        retrieved_chunks: Optional list of retrieved chunk texts for citation quality.

    Returns:
        Dict with overall_score, criteria, judge_model, generation_model, and reason.
    """
    judge_model_name, judge_api_key = _get_judge_config()
    generation_model = os.getenv(
        "LLM_GENERATION_MODEL",
        os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    )

    # Try independent Claude judge when API key is present
    if judge_api_key and judge_model_name:
        prompt = _build_judge_prompt(query, answer, expected_answer)
        response_text = _call_anthropic_judge(prompt, judge_model_name, judge_api_key)
        parsed = _parse_judge_response(response_text)

        if parsed and all(k in parsed for k in ("correctness", "completeness", "citation_quality", "clarity")):
            criteria = {
                "correctness": float(parsed["correctness"]),
                "completeness": float(parsed["completeness"]),
                "citation_quality": float(parsed["citation_quality"]),
                "clarity": float(parsed["clarity"]),
            }
            overall_score = round(sum(criteria.values()) / len(criteria), 3)
            return {
                "overall_score": overall_score,
                "criteria": criteria,
                "judge_model": judge_model_name,
                "generation_model": generation_model,
                "reason": parsed.get("reason", "LLM judge evaluation"),
                "is_fallback": False,
            }

    # Fallback to semantic-similarity scorer (sentence-transformers cosine)
    fallback_model = judge_model_name or "no-judge-configured"
    logger.info(
        "llm_judge_fallback_triggered",
        extra={
            "judge_model": fallback_model,
            "has_api_key": bool(judge_api_key),
            "has_model_name": bool(judge_model_name),
            "reason": "JUDGE_ANTHROPIC_API_KEY absent" if not judge_api_key else "LLM judge call failed",
        },
    )

    return _compute_fallback_scores(
        query=query,
        answer=answer,
        expected_answer=expected_answer,
        retrieved_chunks=retrieved_chunks,
        judge_model=fallback_model,
        generation_model=generation_model,
    )


def judge_pairwise(
    query: str,
    answer_a: str,
    answer_b: str,
    expected_answer: str,
    retrieved_chunks: Optional[Sequence[str]] = None,
    randomize_labels: bool = True,
) -> dict:
    """Compare two answers using an independent Claude judge with A/B randomization.

    Uses JUDGE_MODEL_NAME + JUDGE_ANTHROPIC_API_KEY for the judge.
    When the judge API key is absent, falls back to a Python semantic-similarity
    scorer using sentence-transformers cosine similarity.

    NEVER falls back to the generation model (gpt-4o-mini).

    Args:
        query: The original query string.
        answer_a: The first answer to evaluate.
        answer_b: The second answer to evaluate.
        expected_answer: The expected/reference answer.
        retrieved_chunks: Optional list of retrieved chunk texts for citation quality.
        randomize_labels: If True, randomly swap A/B labels to reduce position bias.

    Returns:
        Dict with scores for both answers, judge_model, generation_model, randomization info, and reason.
    """
    judge_model_name, judge_api_key = _get_judge_config()
    generation_model = os.getenv(
        "LLM_GENERATION_MODEL",
        os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    )

    # Randomize A/B labels to reduce position bias
    labels_swapped = False
    if randomize_labels:
        if random.random() < 0.5:
            answer_a, answer_b = answer_b, answer_a
            labels_swapped = True

    # Try independent Claude judge when API key is present
    if judge_api_key and judge_model_name:
        prompt = _build_pairwise_judge_prompt(query, answer_a, answer_b, expected_answer)
        response_text = _call_anthropic_judge(prompt, judge_model_name, judge_api_key)
        parsed = _parse_pairwise_judge_response(response_text)

        if parsed and "answer_a" in parsed and "answer_b" in parsed:
            criteria_a = {
                "correctness": float(parsed["answer_a"]["correctness"]),
                "completeness": float(parsed["answer_a"]["completeness"]),
                "citation_quality": float(parsed["answer_a"]["citation_quality"]),
                "clarity": float(parsed["answer_a"]["clarity"]),
            }
            overall_a = round(sum(criteria_a.values()) / len(criteria_a), 3)

            criteria_b = {
                "correctness": float(parsed["answer_b"]["correctness"]),
                "completeness": float(parsed["answer_b"]["completeness"]),
                "citation_quality": float(parsed["answer_b"]["citation_quality"]),
                "clarity": float(parsed["answer_b"]["clarity"]),
            }
            overall_b = round(sum(criteria_b.values()) / len(criteria_b), 3)

            if labels_swapped:
                overall_a, overall_b = overall_b, overall_a
                criteria_a, criteria_b = criteria_b, criteria_a

            return {
                "answer_a": {"overall_score": overall_a, "criteria": criteria_a},
                "answer_b": {"overall_score": overall_b, "criteria": criteria_b},
                "judge_model": judge_model_name,
                "generation_model": generation_model,
                "labels_swapped": labels_swapped,
                "reason": parsed.get("reason", "LLM pairwise judge evaluation"),
                "is_fallback": False,
            }

    # Fallback to semantic-similarity scorer
    fallback_model = judge_model_name or "no-judge-configured"
    logger.info(
        "llm_judge_pairwise_fallback_triggered",
        extra={
            "judge_model": fallback_model,
            "has_api_key": bool(judge_api_key),
            "has_model_name": bool(judge_model_name),
            "reason": "JUDGE_ANTHROPIC_API_KEY absent" if not judge_api_key else "LLM judge call failed",
        },
    )

    result_a = _compute_fallback_scores(
        query=query, answer=answer_a, expected_answer=expected_answer,
        retrieved_chunks=retrieved_chunks, judge_model=fallback_model,
        generation_model=generation_model,
    )
    result_b = _compute_fallback_scores(
        query=query, answer=answer_b, expected_answer=expected_answer,
        retrieved_chunks=retrieved_chunks, judge_model=fallback_model,
        generation_model=generation_model,
    )

    if labels_swapped:
        result_a, result_b = result_b, result_a

    return {
        "answer_a": {"overall_score": result_a["overall_score"], "criteria": result_a["criteria"]},
        "answer_b": {"overall_score": result_b["overall_score"], "criteria": result_b["criteria"]},
        "judge_model": fallback_model,
        "generation_model": generation_model,
        "labels_swapped": labels_swapped,
        "reason": result_a["reason"],
        "is_fallback": True,
    }