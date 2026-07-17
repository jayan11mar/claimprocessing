"""HITL trigger rule evaluation engine.

Loads rules from ``config/hitl_rules.yaml`` and evaluates them against
a context dict to determine whether a HITL pause is required.
"""

import os
from typing import Any, Dict, List, Optional

import yaml

from app.config import get_settings
from app.hitl.models import HITLTask, HITLTriggerResult
from app.logging.json_logger import get_logger

logger = get_logger("app.hitl.triggers")

# ── Rule loading ──────────────────────────────────────────────────────────

_RULES_CACHE: Optional[List[Dict[str, Any]]] = None


def load_rules(rules_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load trigger rules from the YAML config file.

    Results are cached after the first load.
    """
    global _RULES_CACHE
    if _RULES_CACHE is not None:
        return _RULES_CACHE

    path = rules_path or get_settings().HITL_RULES_PATH
    if not os.path.exists(path):
        logger.warning("hitl_rules_file_not_found", {"path": path})
        _RULES_CACHE = []
        return _RULES_CACHE

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rules = (data or {}).get("trigger_rules", [])
    _RULES_CACHE = rules
    logger.info("hitl_rules_loaded", {"rule_count": len(rules)})
    return rules


def clear_rules_cache() -> None:
    """Clear the cached rules (used in tests)."""
    global _RULES_CACHE
    _RULES_CACHE = None


# ── Evaluation ────────────────────────────────────────────────────────────


def _evaluate_rule(rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Evaluate a single rule against the given context.

    Supported operators: ``>``, ``>=``, ``<``, ``<=``, ``equals``,
    ``is_true``, ``is_false``.
    """
    field = rule.get("field", "")
    operator = rule.get("operator", "equals")
    expected = rule.get("value")

    actual = context.get(field)

    if operator == "is_true":
        return bool(actual) is True
    if operator == "is_false":
        return bool(actual) is False

    if operator == "equals":
        return actual == expected

    # Numeric comparisons
    try:
        actual_num = float(actual) if actual is not None else 0.0
        expected_num = float(expected) if expected is not None else 0.0
    except (TypeError, ValueError):
        return False

    if operator == ">":
        return actual_num > expected_num
    if operator == ">=":
        return actual_num >= expected_num
    if operator == "<":
        return actual_num < expected_num
    if operator == "<=":
        return actual_num <= expected_num

    logger.warning("hitl_unknown_operator", {"operator": operator, "rule_id": rule.get("rule_id")})
    return False


def evaluate_triggers(
    context: Dict[str, Any],
    rules: Optional[List[Dict[str, Any]]] = None,
) -> HITLTriggerResult:
    """Evaluate all trigger rules against the given context.

    Args:
        context: A dict with fields like ``claim_amount``, ``decision``,
            ``fraud_flag``, ``policy_exclusion``, etc.
        rules: Optional list of rules.  If ``None``, rules are loaded from
            the YAML config file.

    Returns:
        A ``HITLTriggerResult`` with ``triggered`` set to ``True`` if any
        rule matched, along with the matched rules and a populated task.
    """
    if rules is None:
        rules = load_rules()

    matched: List[Dict[str, Any]] = []
    for rule in rules:
        if _evaluate_rule(rule, context):
            matched.append(rule)

    if not matched:
        return HITLTriggerResult(triggered=False)

    # Use the first matched rule for the task
    primary = matched[0]
    task = HITLTask(
        session_id=context.get("session_id", ""),
        rule_id=primary.get("rule_id", "unknown"),
        rule_reason=primary.get("reason", "Trigger rule matched"),
        retrieved_chunks=context.get("retrieved_chunks", []),
        reasoning_trace=context.get("reasoning_trace", ""),
        confidence=context.get("confidence", 0.0),
        recommendation=context.get("recommendation", {}),
        user_message=context.get("user_message", ""),
        agent_response=context.get("agent_response", ""),
    )

    return HITLTriggerResult(
        triggered=True,
        matched_rules=matched,
        task=task,
    )