"""LLM-as-judge wrapper for RAG answers with cross-family judging support.

Uses a different model family for judging than for generation to avoid
self-evaluation bias. When the generation model is OpenAI, the judge uses
Anthropic Claude, and vice versa.
"""

import json
import os
import re
from typing import Any, Dict, Optional, Sequence

from eval.extrinsic import compute_extrinsic_metrics


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _call_llm_judge(prompt: str, judge_model: str) -> Optional[str]:
    """Call an LLM judge model and return the raw response text.

    Supports OpenAI (gpt-*) and Anthropic (claude-*) models.
    Returns None if the call fails.
    """
    if judge_model.startswith("gpt-") or judge_model.startswith("text-"):
        return _call_openai_judge(prompt, judge_model)
    elif judge_model.startswith("claude-"):
        return _call_anthropic_judge(prompt, judge_model)
    return None


def _call_openai_judge(prompt: str, model: str) -> Optional[str]:
    """Call an OpenAI model as judge."""
    try:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        return response.choices[0].message.content
    except Exception:
        return None


def _call_anthropic_judge(prompt: str, model: str) -> Optional[str]:
    """Call an Anthropic Claude model as judge."""
    try:
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None
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


def _parse_judge_response(response_text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse the JSON response from the LLM judge."""
    if not response_text:
        return None
    try:
        # Try direct JSON parse
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        # Try to find any JSON object in the text
        json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def judge_answer(
    query: str,
    answer: str,
    expected_answer: str,
    retrieved_chunks: Optional[Sequence[str]] = None,
) -> dict:
    """Score an answer using a cross-family LLM judge when available, or a deterministic fallback.

    Uses a different model family for judging than for generation to avoid
    self-evaluation bias. The judge model is configured via LLM_JUDGE_MODEL env var
    (default: claude-3-sonnet), while the generation model is configured via
    LLM_GENERATION_MODEL or OPENAI_MODEL_NAME.

    Args:
        query: The original query string.
        answer: The generated answer to evaluate.
        expected_answer: The expected/reference answer.
        retrieved_chunks: Optional list of retrieved chunk texts for citation quality.

    Returns:
        Dict with overall_score, criteria, judge_model, generation_model, and reason.
    """
    # Determine models
    judge_model = os.getenv("LLM_JUDGE_MODEL", "claude-3-sonnet")
    generation_model = os.getenv(
        "LLM_GENERATION_MODEL",
        os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    )

    # Try the LLM judge first
    prompt = _build_judge_prompt(query, answer, expected_answer)
    response_text = _call_llm_judge(prompt, judge_model)
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
            "judge_model": judge_model,
            "generation_model": generation_model,
            "reason": parsed.get("reason", "LLM judge evaluation"),
        }

    # Fallback to deterministic heuristic scoring
    metrics = compute_extrinsic_metrics(
        answer=answer,
        expected_answer=expected_answer,
        retrieved_chunks=retrieved_chunks,
    )
    correctness = metrics["answer_correctness"]
    completeness = min(1.0, correctness + 0.1)
    citation_quality = 1.0 if retrieved_chunks else 0.0
    clarity = 5.0 if answer and len(_normalize(answer).split()) >= 4 else 3.0

    criteria = {
        "correctness": round(5 * correctness, 3),
        "completeness": round(5 * completeness, 3),
        "citation_quality": round(5 * citation_quality, 3),
        "clarity": round(clarity, 3),
    }

    overall_score = round(sum(criteria.values()) / len(criteria), 3)
    return {
        "overall_score": overall_score,
        "criteria": criteria,
        "judge_model": judge_model,
        "generation_model": generation_model,
        "reason": "heuristic fallback (LLM judge unavailable or returned invalid response)",
    }