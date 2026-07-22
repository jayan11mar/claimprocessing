"""
Prompt loader — delegates to the versioned PromptRegistry.

This module maintains backward compatibility with existing code that calls
get_system_template(), get_guardrail_response(), get_few_shot_examples(),
and get_json_format_instruction(). Under the hood it reads from the
versioned YAML prompt files via the PromptRegistry.
"""
import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.prompt_manager.registry import get_registry, initialize_prompts

logger = logging.getLogger(__name__)

_TEMPLATES_CACHE: Dict[str, Any] = {}
_REGISTRY_INITIALIZED = False

# Absolute last-resort safety net for guardrail fallback.
# This is intentionally defined ONCE at module top so it is auditable
# rather than scattered as an ad-hoc literal. The primary path loads
# "default_refusal" from guardrail.yaml via the registry; this is only
# reached if the registry itself raises an exception (e.g. YAML parse
# error, missing file, or malformed document).
_LAST_RESORT_REFUSAL = "I cannot assist with that request."


def _ensure_registry():
    global _REGISTRY_INITIALIZED
    if not _REGISTRY_INITIALIZED:
        try:
            initialize_prompts()
        except Exception as exc:
            logger.warning("Failed to initialize prompt registry: %s", exc)
        _REGISTRY_INITIALIZED = True


def load_templates() -> Dict[str, Any]:
    """Load templates from the versioned YAML prompt files via registry.

    Returns a dict with the same structure as the old templates.json for
    backward compatibility.
    """
    global _TEMPLATES_CACHE

    if _TEMPLATES_CACHE:
        return _TEMPLATES_CACHE

    _ensure_registry()
    registry = get_registry()

    result: Dict[str, Any] = {
        "system": {},
        "guardrails": {},
        "few_shot_examples": {},
    }

    # Load FAQ system prompt
    try:
        result["system"]["main_faq_assistant"] = registry.get_template("faq_system")
    except Exception:
        pass

    # Load JSON format instruction
    try:
        result["system"]["json_format_instruction"] = registry.get_template("faq_json_instruction")
    except Exception:
        pass

    # Load guardrail responses
    try:
        guardrail_doc = registry.get_prompt("guardrail")
        if guardrail_doc and guardrail_doc.active_version:
            version = guardrail_doc.versions.get(guardrail_doc.active_version)
            if version and version.templates:
                result["guardrails"] = dict(version.templates)
    except Exception:
        pass

    # Load few-shot examples from the old JSON file for backward compatibility
    try:
        templates_path = Path(os.path.dirname(__file__)) / "templates.json"
        if templates_path.exists():
            with open(templates_path, "r") as f:
                old_data = json.load(f)
            result["few_shot_examples"] = old_data.get("few_shot_examples", {})
    except Exception:
        pass

    _TEMPLATES_CACHE.clear()
    _TEMPLATES_CACHE.update(result)
    return _TEMPLATES_CACHE


def get_system_template(template_name: str = "main_faq_assistant") -> str:
    """Get a system template by name from the registry."""
    _ensure_registry()
    registry = get_registry()
    try:
        return registry.get_template("faq_system")
    except Exception:
        templates = load_templates()
        return templates["system"].get(template_name, "")


def get_guardrail_response(rule_name: str) -> str:
    """Get a guardrail response by rule name from the registry."""
    _ensure_registry()
    registry = get_registry()
    try:
        guardrail_doc = registry.get_prompt("guardrail")
        if guardrail_doc and guardrail_doc.active_version:
            version = guardrail_doc.versions.get(guardrail_doc.active_version)
            if version and rule_name in version.templates:
                return version.templates[rule_name]
    except Exception:
        pass

    templates = load_templates()
    # Prefer the YAML-sourced "default_refusal" template; fall back to the
    # module-level safety net if the loaded guardrails dict lacks the key.
    return templates["guardrails"].get(rule_name, templates["guardrails"].get("default_refusal", _LAST_RESORT_REFUSAL))


def get_few_shot_examples() -> list:
    """Get few-shot examples from the old templates.json."""
    templates = load_templates()
    examples_dict = templates.get("few_shot_examples", {})
    return list(examples_dict.values())


def get_json_format_instruction() -> str:
    """Get the JSON format instruction from the registry."""
    _ensure_registry()
    registry = get_registry()
    try:
        return registry.get_template("faq_json_instruction")
    except Exception:
        templates = load_templates()
        return templates["system"].get("json_format_instruction", "")