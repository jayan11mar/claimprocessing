"""Custom evaluation dimensions for production-grade RAG monitoring.

Implements the five new metrics required by Spec 3.5:
  - Golden Set Pass Rate (>=95%)
  - Answer Stability (>=0.90)
  - Regulatory Compliance (>=0.90)
  - Role Appropriateness (100%)
  - HITL Trigger Precision (>=0.85)

Each metric is computed independently and can be used as a standalone scorer
or composed into a multi-dimensional evaluation report.
"""

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _tokenize(text: str) -> set:
    return set(_normalize(text).split())


def _jaccard_similarity(a: str, b: str) -> float:
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a, dtype=np.float64)
    b = np.array(vec_b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Semantic similarity scorer (offline fallback for LLM-as-judge)
# ---------------------------------------------------------------------------

class SemanticSimilarityScorer:
    """Python-based semantic similarity scorer using sentence embeddings.

    Uses sentence-transformers if available; falls back to a character-level
    n-gram Jaccard similarity when the model cannot be loaded.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._fallback = True
        self._load_model()

    def _load_model(self) -> None:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._fallback = False
        except Exception:
            self._fallback = True

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of texts into embedding vectors."""
        if self._fallback or self._model is None:
            # Fallback: return dummy vectors based on character n-gram hashes
            return [self._ngram_vector(t) for t in texts]
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception:
            return [self._ngram_vector(t) for t in texts]

    def _ngram_vector(self, text: str, n: int = 3, dim: int = 384) -> List[float]:
        """Character n-gram based vector as fallback embedding."""
        normalized = _normalize(text)
        ngrams = set()
        for i in range(len(normalized) - n + 1):
            ngrams.add(normalized[i:i + n])
        vec = [0.0] * dim
        for i, ng in enumerate(sorted(ngrams)[:dim]):
            vec[i % dim] = 1.0
        return vec

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute semantic similarity between two texts."""
        if self._fallback:
            return _jaccard_similarity(text_a, text_b)
        try:
            emb = self._model.encode([text_a, text_b], convert_to_numpy=True)
            return _cosine_similarity(emb[0].tolist(), emb[1].tolist())
        except Exception:
            return _jaccard_similarity(text_a, text_b)


# ---------------------------------------------------------------------------
# Metric 1: Golden Set Pass Rate
# ---------------------------------------------------------------------------

def compute_golden_set_pass_rate(
    results: List[Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute the Golden Set Pass Rate metric.

    A case passes if ALL of the following hold:
      - hit_rate_at_k >= threshold (default 0.85)
      - mrr >= threshold (default 0.65)
      - faithfulness >= threshold (default 0.90)
      - answer_correctness >= threshold (default 0.80)
      - llm_judge_score >= threshold (default 4.0/5.0)

    Args:
        results: List of evaluation result dicts, each containing
            'intrinsic', 'extrinsic', 'judge' sub-dicts.
        thresholds: Optional per-metric thresholds to override defaults.

    Returns:
        Dict with pass_rate, passed_count, total_count, per_case_results.
    """
    if not results:
        return {
            "pass_rate": 0.0,
            "passed_count": 0,
            "total_count": 0,
            "per_case_results": [],
        }

    t = thresholds or {}
    hit_rate_thresh = t.get("hit_rate_at_5", 0.85)
    mrr_thresh = t.get("mrr", 0.65)
    faithfulness_thresh = t.get("faithfulness", 0.90)
    correctness_thresh = t.get("answer_correctness", 0.80)
    judge_thresh = t.get("llm_judge_avg", 4.0) / 5.0

    passed = 0
    per_case = []
    for r in results:
        intrinsic = r.get("intrinsic", {})
        extrinsic = r.get("extrinsic", {})
        judge = r.get("judge", {})

        case_pass = (
            intrinsic.get("hit_at_k", 0) >= hit_rate_thresh
            and intrinsic.get("mrr", 0) >= mrr_thresh
            and extrinsic.get("faithfulness", 0) >= faithfulness_thresh
            and extrinsic.get("answer_correctness", 0) >= correctness_thresh
            and judge.get("overall_score", 0) >= judge_thresh
        )
        if case_pass:
            passed += 1
        per_case.append({
            "id": r.get("id", "unknown"),
            "query": r.get("query", ""),
            "passed": case_pass,
            "hit_at_k": intrinsic.get("hit_at_k", 0),
            "mrr": intrinsic.get("mrr", 0),
            "faithfulness": extrinsic.get("faithfulness", 0),
            "answer_correctness": extrinsic.get("answer_correctness", 0),
            "judge_score": judge.get("overall_score", 0),
        })

    total = len(results)
    pass_rate = round(passed / total, 4) if total > 0 else 0.0

    return {
        "pass_rate": pass_rate,
        "passed_count": passed,
        "total_count": total,
        "per_case_results": per_case,
    }


# ---------------------------------------------------------------------------
# Metric 2: Answer Stability
# ---------------------------------------------------------------------------

def compute_answer_stability(
    answers_a: List[str],
    answers_b: List[str],
    queries: Optional[List[str]] = None,
    scorer: Optional[SemanticSimilarityScorer] = None,
) -> Dict[str, Any]:
    """Compute Answer Stability between two runs of the same queries.

    Measures the semantic similarity of answers produced by two different
    runs (e.g., before/after a prompt change, or two model versions).

    Args:
        answers_a: List of answers from the first run.
        answers_b: List of answers from the second run (same order).
        queries: Optional list of queries for per-case reporting.
        scorer: Optional SemanticSimilarityScorer instance.

    Returns:
        Dict with stability_score (mean pairwise similarity), per_pair list.
    """
    if not answers_a or not answers_b:
        return {"stability_score": 0.0, "per_pair": []}

    if len(answers_a) != len(answers_b):
        raise ValueError(
            f"Answer list length mismatch: {len(answers_a)} vs {len(answers_b)}"
        )

    if scorer is None:
        scorer = SemanticSimilarityScorer()

    similarities = []
    per_pair = []
    for i, (a, b) in enumerate(zip(answers_a, answers_b)):
        sim = scorer.similarity(a, b)
        similarities.append(sim)
        entry = {
            "index": i,
            "similarity": round(sim, 4),
        }
        if queries and i < len(queries):
            entry["query"] = queries[i]
        per_pair.append(entry)

    stability = round(float(np.mean(similarities)), 4) if similarities else 0.0

    return {
        "stability_score": stability,
        "per_pair": per_pair,
        "min_similarity": round(float(np.min(similarities)), 4) if similarities else 0.0,
        "max_similarity": round(float(np.max(similarities)), 4) if similarities else 0.0,
        "std_similarity": round(float(np.std(similarities)), 4) if len(similarities) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Metric 3: Regulatory Compliance
# ---------------------------------------------------------------------------

# Known regulatory keywords / phrases for insurance domain
_REG_TARGET_WEIGHT = 1.5
_REGULATORY_PATTERNS: List[Dict[str, Any]] = [
    {"pattern": r"irda[ai]?", "label": "IRDAI reference", "weight": 1.0},
    {"pattern": r"irda[ai]?\s+regulation", "label": "IRDAI regulation", "weight": 1.0},
    {"pattern": r"regulatory\s+guideline", "label": "Regulatory guideline", "weight": 0.9},
    {"pattern": r"as\s+per\s+regulation", "label": "As per regulation", "weight": 0.9},
    {"pattern": r"compliance", "label": "Compliance mention", "weight": 0.8},
    {"pattern": r"regulatory\s+requirement", "label": "Regulatory requirement", "weight": 0.9},
    {"pattern": r"insurance\s+regulatory", "label": "Insurance regulatory", "weight": 0.9},
    {"pattern": r"policy\s+holder\s+protection", "label": "Policyholder protection", "weight": 0.8},
    {"pattern": r"grievance\s+redressal", "label": "Grievance redressal", "weight": 0.8},
    {"pattern": r"portability", "label": "Portability mention", "weight": 0.7},
    {"pattern": r"waiting\s+period", "label": "Waiting period mention", "weight": 0.7},
    {"pattern": r"pre-existing\s+disease", "label": "Pre-existing disease", "weight": 0.7},
    {"pattern": r"sum\s+insured", "label": "Sum insured mention", "weight": 0.6},
    {"pattern": r"claim\s+settlement", "label": "Claim settlement", "weight": 0.6},
    {"pattern": r"cashless", "label": "Cashless mention", "weight": 0.6},
    {"pattern": r"reimbursement", "label": "Reimbursement mention", "weight": 0.5},
    {"pattern": r"network\s+hospital", "label": "Network hospital", "weight": 0.5},
    {"pattern": r"deductible", "label": "Deductible mention", "weight": 0.5},
    {"pattern": r"co-payment", "label": "Co-payment mention", "weight": 0.5},
    {"pattern": r"sub[- ]?limit", "label": "Sub-limit mention", "weight": 0.5},
    {"pattern": r"exclusion", "label": "Exclusion mention", "weight": 0.5},
    {"pattern": r"grace\s+period", "label": "Grace period", "weight": 0.5},
    {"pattern": r"free[- ]?look[- ]?period", "label": "Free look period", "weight": 0.5},
    {"pattern": r"nomination", "label": "Nomination mention", "weight": 0.4},
    {"pattern": r"disclosure", "label": "Disclosure mention", "weight": 0.4},
    {"pattern": r"utmost\s+good\s+faith", "label": "Utmost good faith", "weight": 0.4},
]


def compute_regulatory_compliance(
    answers: List[str],
    queries: Optional[List[str]] = None,
    regulatory_patterns: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute Regulatory Compliance score for a set of answers.

    Measures how well answers reference or align with regulatory language
    expected in the insurance domain.

    Args:
        answers: List of answer texts to evaluate.
        queries: Optional list of queries for per-case reporting.
        regulatory_patterns: Optional custom list of regulatory patterns.
            Each entry: {"pattern": regex_str, "label": str, "weight": float}.

    Returns:
        Dict with compliance_score, per_case list, matched_patterns summary.
    """
    if not answers:
        return {"compliance_score": 0.0, "per_case": [], "matched_patterns": {}}

    patterns = regulatory_patterns or _REGULATORY_PATTERNS
    compiled = [(re.compile(p["pattern"], re.IGNORECASE), p["label"], p["weight"]) for p in patterns]

    per_case = []
    all_matched: Dict[str, int] = {}

    for i, answer in enumerate(answers):
        normalized = _normalize(answer)
        case_score = 0.0
        max_possible = 0.0
        case_matched: Dict[str, float] = {}

        for pattern_re, label, weight in compiled:
            if pattern_re.search(normalized):
                case_score += weight
                case_matched[label] = weight
                all_matched[label] = all_matched.get(label, 0) + 1

        compliance = round(min(1.0, case_score / _REG_TARGET_WEIGHT), 4)
        entry = {
            "index": i,
            "compliance_score": compliance,
            "matched_patterns": case_matched,
        }
        if queries and i < len(queries):
            entry["query"] = queries[i]
        per_case.append(entry)

    overall = round(
        float(np.mean([c["compliance_score"] for c in per_case])), 4
    ) if per_case else 0.0

    return {
        "compliance_score": overall,
        "per_case": per_case,
        "matched_patterns": dict(
            sorted(all_matched.items(), key=lambda x: -x[1])
        ),
    }


# ---------------------------------------------------------------------------
# Metric 4: Role Appropriateness
# ---------------------------------------------------------------------------

def compute_role_appropriateness(
    answers: List[str],
    role_contexts: List[str],
    queries: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute Role Appropriateness metric.

    Verifies that answers respect role-based access constraints:
      - A role with limited access should NOT produce answers containing
        restricted document type information.
      - 100% compliance required (no leakage).

    Args:
        answers: List of answer texts.
        role_contexts: List of role names for each answer (e.g., "agent",
            "underwriter", "compliance_officer", "customer").
        queries: Optional list of queries for per-case reporting.

    Returns:
        Dict with appropriateness_score, per_case list, violations.
    """
    if not answers or not role_contexts:
        return {
            "appropriateness_score": 0.0,
            "per_case": [],
            "violations": [],
            "total_violations": 0,
        }

    if len(answers) != len(role_contexts):
        raise ValueError(
            f"Answer/role list length mismatch: {len(answers)} vs {len(role_contexts)}"
        )

    # Define restricted content per role
    # These are keywords that should NOT appear in answers for certain roles
    role_restrictions: Dict[str, List[str]] = {
        "customer": [
            "internal", "underwriting guideline", "risk score",
            "fraud indicator", "settlement authority", "reserve amount",
            "claims adjuster note", "internal memo",
        ],
        "agent": [
            "underwriting guideline", "risk score",
            "fraud indicator", "settlement authority",
        ],
        "underwriter": [],
        "compliance_officer": [],
        "claims_adjuster": [],
        "manager": [],
        "admin": [],
        "service": [
            "underwriting guideline", "risk score",
            "fraud indicator", "settlement authority",
        ],
    }

    violations = []
    per_case = []

    for i, (answer, role) in enumerate(zip(answers, role_contexts)):
        normalized = _normalize(answer)
        restricted_terms = role_restrictions.get(role, [])
        case_violations = []

        for term in restricted_terms:
            if term.lower() in normalized:
                case_violations.append(term)

        if case_violations:
            violations.append({
                "index": i,
                "role": role,
                "violations": case_violations,
                "answer_snippet": answer[:200],
            })

        entry = {
            "index": i,
            "role": role,
            "appropriate": len(case_violations) == 0,
            "violations": case_violations,
        }
        if queries and i < len(queries):
            entry["query"] = queries[i]
        per_case.append(entry)

    total_violations = len(violations)
    total_cases = len(answers)
    appropriateness = round(
        (total_cases - total_violations) / total_cases, 4
    ) if total_cases > 0 else 0.0

    return {
        "appropriateness_score": appropriateness,
        "per_case": per_case,
        "violations": violations,
        "total_violations": total_violations,
    }


# ---------------------------------------------------------------------------
# Metric 5: HITL Trigger Precision
# ---------------------------------------------------------------------------

def compute_hitl_trigger_precision(
    hitl_decisions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute HITL Trigger Precision metric.

    Measures how often a HITL (human-in-the-loop) trigger was appropriate:
      Precision = True Positives / (True Positives + False Positives)

    A trigger is a True Positive if the human reviewer confirmed the action
    (approved). A trigger is a False Positive if the human reviewer rejected
    the action or it was unnecessary.

    Args:
        hitl_decisions: List of dicts, each with at least:
            - 'triggered': bool (whether HITL was triggered)
            - 'approved': Optional[bool] (True if approved, False if rejected,
              None if pending/unreviewed)
            - 'task_id': str (optional)

    Returns:
        Dict with precision, true_positives, false_positives, total_triggers,
        per_decision breakdown.
    """
    if not hitl_decisions:
        return {
            "precision": 0.0,
            "true_positives": 0,
            "false_positives": 0,
            "total_triggers": 0,
            "per_decision": [],
        }

    tp = 0
    fp = 0
    per_decision = []

    for d in hitl_decisions:
        triggered = d.get("triggered", False)
        approved = d.get("approved")

        if not triggered:
            per_decision.append({
                "task_id": d.get("task_id", "unknown"),
                "triggered": False,
                "classification": "not_triggered",
            })
            continue

        if approved is True:
            tp += 1
            classification = "true_positive"
        elif approved is False:
            fp += 1
            classification = "false_positive"
        else:
            # Unreviewed triggers are not counted in precision
            classification = "unreviewed"

        per_decision.append({
            "task_id": d.get("task_id", "unknown"),
            "triggered": True,
            "approved": approved,
            "classification": classification,
        })

    total_triggers = tp + fp
    precision = round(tp / total_triggers, 4) if total_triggers > 0 else 0.0

    return {
        "precision": precision,
        "true_positives": tp,
        "false_positives": fp,
        "total_triggers": total_triggers,
        "per_decision": per_decision,
    }


# ---------------------------------------------------------------------------
# Composite evaluation: run all five metrics
# ---------------------------------------------------------------------------

def compute_all_custom_metrics(
    results: List[Dict[str, Any]],
    answers_a: Optional[List[str]] = None,
    answers_b: Optional[List[str]] = None,
    role_contexts: Optional[List[str]] = None,
    hitl_decisions: Optional[List[Dict[str, Any]]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute all five custom evaluation metrics in one call.

    Args:
        results: Evaluation results for Golden Set Pass Rate.
        answers_a: First-run answers for Answer Stability.
        answers_b: Second-run answers for Answer Stability.
        role_contexts: Role names for Role Appropriateness.
        hitl_decisions: HITL decision records for HITL Trigger Precision.
        thresholds: Optional threshold overrides.

    Returns:
        Dict with all five metric results and an overall summary.
    """
    queries = [r.get("query", "") for r in results] if results else None

    # 1. Golden Set Pass Rate
    golden = compute_golden_set_pass_rate(results, thresholds)

    # 2. Answer Stability
    stability = compute_answer_stability(
        answers_a or [],
        answers_b or [],
        queries=queries,
    )

    # 3. Regulatory Compliance — only evaluate regulatory-relevant cases
    reg_cases = [
        r for r in (results or [])
        if str(r.get("category", r.get("project", ""))).lower()
           in {"policy", "regulatory", "compliance", "coverage"}
    ]
    # Only evaluate if we have actual generated answers, not just expected_answer templates
    reg_answers = [r.get("answer", "") for r in reg_cases]
    reg_queries = [r.get("query", "") for r in reg_cases]
    
    # Check if we have actual generated answers (not just templates)
    has_real_answers = any(r.get("answer") for r in reg_cases)
    
    compliance = {"compliance_score": 0.0, "per_case": [], "matched_patterns": {}}
    if has_real_answers:
        compliance = compute_regulatory_compliance(
            reg_answers,
            queries=reg_queries,
        )
        overall_reg = compliance["compliance_score"]
    else:
        # No real answers generated, mark as not-evaluable
        overall_reg = None

    # 4. Role Appropriateness
    all_answers = [r.get("answer", r.get("expected_answer", "")) for r in (results or [])]
    appropriateness = compute_role_appropriateness(
        all_answers,
        role_contexts or ["customer"] * len(all_answers),
        queries=queries,
    )

    # 5. HITL Trigger Precision
    hitl_precision = compute_hitl_trigger_precision(hitl_decisions or [])

    # Overall summary
    overall = {
        "golden_set_pass_rate": golden["pass_rate"],
        "answer_stability": stability["stability_score"] if stability["per_pair"] else None,
        "regulatory_compliance": overall_reg,
        "role_appropriateness": appropriateness["appropriateness_score"],
        "hitl_trigger_precision": hitl_precision["precision"] if hitl_precision["total_triggers"] > 0 else None,
    }

    # Check against required thresholds
    t = thresholds or {}
    required = {
        "golden_set_pass_rate": t.get("golden_set_pass_rate", 0.95),
        "answer_stability": t.get("answer_stability", 0.90),
        "regulatory_compliance": t.get("regulatory_compliance", 0.90),
        "role_appropriateness": t.get("role_appropriateness", 1.0),
        "hitl_trigger_precision": t.get("hitl_trigger_precision", 0.85),
    }

    all_passed = all(
        overall[k] >= required[k] for k in required if overall[k] is not None
    )

    return {
        "golden_set_pass_rate": golden,
        "answer_stability": stability,
        "regulatory_compliance": compliance,
        "role_appropriateness": appropriateness,
        "hitl_trigger_precision": hitl_precision,
        "overall": overall,
        "required_thresholds": required,
        "all_metrics_passed": all_passed,
    }