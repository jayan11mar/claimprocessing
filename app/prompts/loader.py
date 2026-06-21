"""Template loader for prompt and guardrail templates."""

import json
from pathlib import Path
from typing import Any, Dict

_TEMPLATES_CACHE: Dict[str, Any] = {}


def load_templates() -> Dict[str, Any]:
    """Load templates from templates.json into cache and return them.
    
    Returns:
        Dict containing all templates (system, guardrails, few_shot_examples).
    """
    global _TEMPLATES_CACHE
    
    if _TEMPLATES_CACHE:
        return _TEMPLATES_CACHE
    
    templates_path = Path(__file__).parent / "templates.json"
    
    if not templates_path.exists():
        raise FileNotFoundError(f"Templates file not found at {templates_path}")
    
    with open(templates_path, "r") as f:
        _TEMPLATES_CACHE = json.load(f)
    
    return _TEMPLATES_CACHE


def get_system_template(template_name: str = "main_faq_assistant") -> str:
    """Get a system template by name.
    
    Args:
        template_name: Name of the system template (default: main_faq_assistant).
    
    Returns:
        The system template string.
    """
    templates = load_templates()
    return templates["system"].get(template_name, "")


def get_guardrail_response(rule_name: str) -> str:
    """Get a guardrail response template by rule name.
    
    Args:
        rule_name: Name of the guardrail rule (e.g., "pii_warning", "off_topic_response").
    
    Returns:
        The guardrail response text.
    """
    templates = load_templates()
    return templates["guardrails"].get(rule_name, "I cannot assist with that request.")


def get_few_shot_examples() -> list:
    """Get all few-shot examples.
    
    Returns:
        List of example dicts with 'user' and 'assistant' keys.
    """
    templates = load_templates()
    examples_dict = templates.get("few_shot_examples", {})
    return list(examples_dict.values())


def get_json_format_instruction() -> str:
    """Get the JSON format instruction for LLM output.
    
    Returns:
        The JSON format instruction.
    """
    templates = load_templates()
    return templates["system"].get("json_format_instruction", "")
